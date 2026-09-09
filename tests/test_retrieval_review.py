from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tescogpt.evaluation.retrieval_review import (
    evaluate_retrieval_review,
    freeze_retrieval_review,
    initialize_retrieval_review,
    validate_retrieval_review,
)


def _registry(path: Path) -> Path:
    pd.DataFrame(
        [
            {
                "case_id": "query-damaged",
                "conversation_id": "query-damaged",
                "message": "damaged yoghurt pot leaking",
                "prior_context": "",
                "sample_slice": "natural",
            },
            {
                "case_id": "query-delivery",
                "conversation_id": "query-delivery",
                "message": "delivery order is late",
                "prior_context": "",
                "sample_slice": "challenge",
            },
        ]
    ).to_csv(path, index=False)
    return path


def _corpus(path: Path) -> Path:
    rows = [
        ("damaged-z-positive", "damaged yoghurt pot leaking", True, False),
        ("damaged-a-unresolved", "damaged yoghurt pot leaking", False, True),
        ("delivery-z-positive", "delivery order late missing", True, False),
        ("delivery-a-unresolved", "delivery order late missing", False, True),
    ]
    pd.DataFrame(
        [
            {
                "case_id": case_id,
                "conversation_id": case_id,
                "message": message,
                "historical_reply": f"Historical reply for {case_id}.",
                "customer_followup": "thanks sorted" if positive else "still broken",
                "private_handoff_proxy": False,
                "positive_outcome_proxy": positive,
                "unresolved_outcome_proxy": unresolved,
                "split": "train",
            }
            for case_id, message, positive, unresolved in rows
        ]
    ).to_csv(path, index=False)
    return path


def test_review_is_pooled_stratified_and_system_blinded(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    key = tmp_path / "key.csv"
    manifest = initialize_retrieval_review(
        _registry(tmp_path / "registry.csv"),
        _corpus(tmp_path / "corpus.csv"),
        review,
        key,
        tmp_path / "manifest.json",
        query_count=2,
        top_k=2,
        candidate_pool=4,
        seed=7,
    )

    review_frame = pd.read_csv(review, keep_default_na=False)
    key_frame = pd.read_csv(key, keep_default_na=False)
    assert manifest["slice_query_counts"] == {"challenge": 1, "natural": 1}
    assert review_frame["query_case_id"].nunique() == 2
    assert set(review_frame["retrieval_review_id"]) == set(key_frame["retrieval_review_id"])
    assert not {
        "retrieved_by",
        "bm25_rank",
        "outcome_rank",
        "outcome_tier",
    } & set(review_frame)
    assert {"retrieved_by", "bm25_rank", "outcome_rank", "outcome_tier"} <= set(
        key_frame
    )
    assert validate_retrieval_review(review)["completed_count"] == 0


def test_validator_rejects_partial_and_invalid_ratings(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    initialize_retrieval_review(
        _registry(tmp_path / "registry.csv"),
        _corpus(tmp_path / "corpus.csv"),
        review,
        tmp_path / "key.csv",
        tmp_path / "manifest.json",
        query_count=1,
        top_k=1,
        candidate_pool=4,
    )
    frame = pd.read_csv(review, dtype="string", keep_default_na=False)
    frame.loc[0, "relevance_grade"] = "2"
    frame.to_csv(review, index=False)
    with pytest.raises(ValueError, match="partially completed"):
        validate_retrieval_review(review)

    frame.loc[0, "reviewer_id"] = "human_a"
    frame.loc[0, "relevance_grade"] = "3"
    frame.to_csv(review, index=False)
    with pytest.raises(ValueError, match="Invalid relevance grades"):
        validate_retrieval_review(review)


def test_complete_review_requires_borderline_reasons_and_one_reviewer(
    tmp_path: Path,
) -> None:
    review = tmp_path / "review.csv"
    initialize_retrieval_review(
        _registry(tmp_path / "registry.csv"),
        _corpus(tmp_path / "corpus.csv"),
        review,
        tmp_path / "key.csv",
        tmp_path / "manifest.json",
        query_count=1,
        top_k=2,
        candidate_pool=4,
    )
    frame = pd.read_csv(review, dtype="string", keep_default_na=False)
    frame["relevance_grade"] = "1"
    frame["reviewer_id"] = "human_a"
    frame.to_csv(review, index=False)

    with pytest.raises(ValueError, match="borderline grades lack a reason"):
        validate_retrieval_review(review, require_complete=True)

    frame["relevance_reason"] = "Related issue, but the support action differs."
    frame.loc[frame.index[0], "reviewer_id"] = "human_b"
    frame.to_csv(review, index=False)
    with pytest.raises(ValueError, match="exactly one reviewer ID"):
        validate_retrieval_review(review, require_complete=True)


def test_evaluation_reports_pooled_metrics_and_paired_result(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    key = tmp_path / "key.csv"
    manifest_path = tmp_path / "manifest.json"
    initialize_retrieval_review(
        _registry(tmp_path / "registry.csv"),
        _corpus(tmp_path / "corpus.csv"),
        review,
        key,
        manifest_path,
        query_count=2,
        top_k=2,
        candidate_pool=4,
        seed=7,
    )
    review_frame = pd.read_csv(review, dtype="string", keep_default_na=False)
    key_frame = pd.read_csv(key, dtype="string", keep_default_na=False)
    positive_ids = set(
        key_frame.loc[
            key_frame["candidate_case_id"].str.endswith("positive"),
            "retrieval_review_id",
        ]
    )
    review_frame["relevance_grade"] = review_frame["retrieval_review_id"].map(
        lambda review_id: "2" if review_id in positive_ids else "0"
    )
    review_frame["reviewer_id"] = "human_a"
    review_frame.to_csv(review, index=False)
    manifest = freeze_retrieval_review(review, key, manifest_path)

    report = evaluate_retrieval_review(
        review,
        key,
        tmp_path / "metrics.json",
        top_k=2,
        manifest_path=manifest_path,
    )

    assert manifest["label_status"] == "HUMAN_LABELED"
    assert manifest["reviewer_ids"] == ["human_a"]
    assert report["systems"]["bm25"]["query_count"] == 2
    assert "pooled_precision_at_2" in report["systems"]["outcome"]
    assert "pooled_ndcg_at_2" in report["systems"]["outcome"]
    assert report["paired_comparison"]["outcome_wins"] == 2
    assert "recall is not estimable" in report["important_limitation"]
