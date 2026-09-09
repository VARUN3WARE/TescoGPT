from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from tescogpt.evaluation.agreement import (
    annotation_agreement,
    cohen_kappa,
    weighted_kappa,
)
from tescogpt.evaluation.judge import (
    JudgeRating,
    OpenAIReplyJudge,
    judge_human_agreement,
    judge_review_sheet,
    validate_judge_output,
)
from tescogpt.evaluation.labels import LABEL_COLUMNS
from tescogpt.evaluation.reply_review import (
    RATING_DIMENSIONS,
    evaluate_reply_quality,
    freeze_reply_review,
    initialize_reply_review,
    validate_reply_ratings,
    validate_reply_review_sources,
)


def _gold(path: Path) -> Path:
    rows = []
    for index in range(4):
        rows.append(
            {
                "case_id": f"case-{index}",
                "conversation_id": str(index),
                "tweet_id": str(index),
                "message": f"message {index}",
                "prior_context": "",
                **{column: "" for column in LABEL_COLUMNS},
                "intent_label": "feedback_praise_or_suggestion",
                "handling_label": "AUTO_HANDLE",
                "reason_code": "NO_ACTION_NEEDED",
                "must_include": "acknowledge feedback",
                "must_avoid": "invented action",
                "annotator_id": "human_a",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _prediction(path: Path, system: str) -> Path:
    rows = []
    for index in range(4):
        rows.append(
            {
                "case_id": f"case-{index}",
                "system_name": system,
                "draft_reply": f"Thanks for the feedback {index}.",
                "handling_decision": "AUTO_HANDLE",
                "decision_reason": "NO_ACTION_NEEDED",
                "evidence_quotes": "[]",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_reply_review_is_stratified_and_system_blinded(tmp_path: Path) -> None:
    gold = _gold(tmp_path / "gold.csv")
    registry = tmp_path / "registry.csv"
    pd.DataFrame(
        [
            {"case_id": f"case-{index}", "sample_slice": "natural" if index < 3 else "challenge"}
            for index in range(4)
        ]
    ).to_csv(registry, index=False)
    review = tmp_path / "review.csv"
    key = tmp_path / "key.csv"
    predictions = [
        _prediction(tmp_path / "a.csv", "a"),
        _prediction(tmp_path / "b.csv", "b"),
    ]

    manifest = initialize_reply_review(
        gold,
        registry,
        predictions,
        review,
        key,
        tmp_path / "manifest.json",
        case_count=3,
        decoy_count=1,
        seed=4,
    )

    review_frame = pd.read_csv(review)
    key_frame = pd.read_csv(key)
    assert len(review_frame) == 7
    assert not {"system_name", "sample_slice", "row_role", "control_name"} & set(
        review_frame
    )
    assert set(key_frame.loc[key_frame["row_role"].eq("compared"), "system_name"]) == {
        "a",
        "b",
    }
    assert set(key_frame["row_role"]) == {"compared", "control"}
    assert manifest["slice_case_counts"] == {"challenge": 1, "natural": 2}
    assert manifest["control_count"] == 1
    assert manifest["review_content_sha256"]

    compared_id = key_frame.loc[key_frame["row_role"].eq("compared"), "review_id"].iloc[0]
    review_frame.loc[
        review_frame["review_id"].eq(compared_id), "draft_reply"
    ] = "A draft that no frozen system produced."
    review_frame.to_csv(review, index=False)
    with pytest.raises(ValueError, match="differs from frozen prediction"):
        validate_reply_review_sources(review, key, gold, predictions)


def test_reply_rating_validator_rejects_partial_row(tmp_path: Path) -> None:
    path = tmp_path / "review.csv"
    row = {
        "review_id": "r1",
        "case_id": "c1",
        "draft_reply": "reply",
        **{dimension: "" for dimension in RATING_DIMENSIONS},
        "critical_error_tags": "",
        "overall_pass": "",
        "reviewer_id": "",
        "review_notes": "",
    }
    row["tone_empathy"] = "2"
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="partially completed"):
        validate_reply_ratings(path)


def test_reply_rating_validator_enforces_frozen_pass_rule(tmp_path: Path) -> None:
    path = tmp_path / "review.csv"
    row = {
        "review_id": "r1",
        "case_id": "c1",
        "draft_reply": "reply",
        **{dimension: "2" for dimension in RATING_DIMENSIONS},
        "critical_error_tags": "unsupported_claim_or_action",
        "overall_pass": "PASS",
        "reviewer_id": "human_a",
        "review_notes": "The draft invents a completed refund.",
    }
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="overall-pass rule"):
        validate_reply_ratings(path)


def test_complete_reply_review_requires_failure_notes_and_one_reviewer(
    tmp_path: Path,
) -> None:
    path = tmp_path / "review.csv"
    failed = {
        "review_id": "r1",
        "case_id": "c1",
        "draft_reply": "reply",
        **{dimension: "2" for dimension in RATING_DIMENSIONS},
        "issue_understanding": "0",
        "critical_error_tags": "irrelevant_or_misunderstood",
        "overall_pass": "FAIL",
        "reviewer_id": "human_a",
        "review_notes": "",
    }
    pd.DataFrame([failed]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="lack explanatory notes"):
        validate_reply_ratings(path, require_complete=True)

    failed["review_notes"] = "The draft does not address the customer's issue."
    passed = {
        **failed,
        "review_id": "r2",
        "case_id": "c2",
        "issue_understanding": "2",
        "critical_error_tags": "",
        "overall_pass": "PASS",
        "reviewer_id": "human_b",
        "review_notes": "",
    }
    pd.DataFrame([failed, passed]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly one reviewer ID"):
        validate_reply_ratings(path, require_complete=True)


def test_reply_quality_freeze_and_pairwise_summary(tmp_path: Path) -> None:
    gold = _gold(tmp_path / "gold.csv")
    registry = tmp_path / "registry.csv"
    pd.DataFrame(
        [
            {"case_id": f"case-{index}", "sample_slice": "natural" if index < 3 else "challenge"}
            for index in range(4)
        ]
    ).to_csv(registry, index=False)
    review = tmp_path / "review.csv"
    key = tmp_path / "key.csv"
    manifest_path = tmp_path / "manifest.json"
    initialize_reply_review(
        gold,
        registry,
        [_prediction(tmp_path / "a.csv", "a"), _prediction(tmp_path / "b.csv", "b")],
        review,
        key,
        manifest_path,
        case_count=3,
        decoy_count=1,
        seed=4,
    )
    review_frame = pd.read_csv(review, dtype="string", keep_default_na=False)
    key_frame = pd.read_csv(key, dtype="string", keep_default_na=False)
    system_by_review = key_frame.set_index("review_id")["system_name"]
    for index, row in review_frame.iterrows():
        system = system_by_review.loc[row["review_id"]]
        is_reference = system == "a"
        is_control = system == "__judge_control__"
        for dimension in RATING_DIMENSIONS:
            review_frame.loc[index, dimension] = (
                "0" if is_control else "2" if is_reference else "1"
            )
        review_frame.loc[index, "critical_error_tags"] = (
            "unsafe_data_request" if is_control else ""
        )
        review_frame.loc[index, "overall_pass"] = (
            "PASS" if is_reference and not is_control else "FAIL"
        )
        review_frame.loc[index, "reviewer_id"] = "human_a"
        review_frame.loc[index, "review_notes"] = (
            "Synthetic control contains an unsafe data request." if is_control else ""
        )
    review_frame.to_csv(review, index=False)

    manifest = freeze_reply_review(review, key, manifest_path)
    report = evaluate_reply_quality(
        review,
        key,
        tmp_path / "quality.json",
        reference_system="a",
        manifest_path=manifest_path,
    )

    assert manifest["label_status"] == "HUMAN_RATED"
    assert report["systems"]["a"]["slices"]["all"]["overall_pass_rate"] == 1
    assert report["systems"]["b"]["slices"]["all"]["overall_pass_rate"] == 0
    assert report["rubric_derived_pairwise"]["b"]["reference_wins"] == 3
    assert report["human_decoy_controls"]["fail_rate"] == 1
    assert report["human_decoy_controls"]["critical_error_detection_rate"] == 1


def test_kappa_statistics_have_known_endpoints() -> None:
    same = pd.Series([0, 1, 2, 1])
    opposite = pd.Series([2, 1, 0, 1])
    assert cohen_kappa(pd.Series(["a", "b"]), pd.Series(["a", "b"])) == 1
    assert weighted_kappa(same, same) == 1
    assert weighted_kappa(same, opposite) < 0


class FakeResponses:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        result = {
            **{dimension: 2 for dimension in RATING_DIMENSIONS},
            "critical_error_tags": [],
            "overall_pass": "PASS",
            "rationale": "Safe, relevant, and appropriate.",
        }
        return SimpleNamespace(
            id="judge-1",
            model="judge-model-2026-09-01",
            output_text=json.dumps(result),
            usage={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
        )


def test_openai_judge_is_blinded_structured_and_cached(tmp_path: Path) -> None:
    responses = FakeResponses()
    judge = OpenAIReplyJudge(
        "judge-model",
        tmp_path / "cache",
        client=SimpleNamespace(responses=responses),
    )
    row = {
        "message": "Thanks Tesco",
        "prior_context": "",
        "gold_intent": "feedback_praise_or_suggestion",
        "gold_handling": "AUTO_HANDLE",
        "gold_reason": "NO_ACTION_NEEDED",
        "must_include": "thanks",
        "must_avoid": "claims",
        "draft_reply": "Thanks for your feedback.",
        "proposed_handling": "AUTO_HANDLE",
        "proposed_reason": "NO_ACTION_NEEDED",
        "evidence_quotes": "[]",
        "system_name": "must-not-be-sent",
    }

    first = judge.rate(row)
    second = judge.rate(row)

    assert first == second
    assert len(responses.calls) == 1
    request = responses.calls[0]
    assert request["store"] is False
    assert "must-not-be-sent" not in request["input"]
    assert request["text"]["format"]["strict"] is True
    provenance = judge.provenance()
    assert provenance["request_count"] == 2
    assert provenance["api_call_count"] == 1
    assert provenance["cache_hit_count"] == 1
    assert provenance["resolved_models"] == ["judge-model-2026-09-01"]
    assert provenance["usage_totals"]["total_tokens"] == 30


def test_judge_rating_rejects_inconsistent_pass() -> None:
    with pytest.raises(ValueError, match="frozen reply-quality rule"):
        JudgeRating(
            **{dimension: 2 for dimension in RATING_DIMENSIONS},
            critical_error_tags=("unsupported_claim_or_action",),
            overall_pass="PASS",
            rationale="The claim is unsupported.",
        )


def test_judge_manifest_contains_request_level_provenance(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    row = {
        "review_id": "review-1",
        "case_id": "case-1",
        "message": "Thanks Tesco",
        "prior_context": "",
        "gold_intent": "feedback_praise_or_suggestion",
        "gold_handling": "AUTO_HANDLE",
        "gold_reason": "NO_ACTION_NEEDED",
        "must_include": "thanks",
        "must_avoid": "invented action",
        "draft_reply": "Thanks for your feedback.",
        "proposed_handling": "AUTO_HANDLE",
        "proposed_reason": "NO_ACTION_NEEDED",
        "evidence_quotes": "[]",
        **{dimension: "2" for dimension in RATING_DIMENSIONS},
        "critical_error_tags": "",
        "overall_pass": "PASS",
        "reviewer_id": "human_a",
        "review_notes": "",
    }
    pd.DataFrame([row]).to_csv(review, index=False)
    output = tmp_path / "judge.csv"
    manifest = judge_review_sheet(
        review,
        output,
        model="judge-model",
        cache_dir=tmp_path / "cache",
        client=SimpleNamespace(responses=FakeResponses()),
    )

    provenance = manifest["judge_provenance"]
    assert provenance["request_count"] == 1
    assert provenance["requests"][0]["review_id"] == "review-1"
    assert provenance["instructions_sha256"]
    validate_judge_output(
        output,
        review_path=review,
        manifest_path=output.with_suffix(".csv.manifest.json"),
    )

    manifest["judge_provenance"]["requests"][0]["response_id"] = ""
    output.with_suffix(".csv.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="lacks response lineage"):
        validate_judge_output(
            output,
            review_path=review,
            manifest_path=output.with_suffix(".csv.manifest.json"),
        )


def test_annotation_and_judge_agreement_reports_are_written(tmp_path: Path) -> None:
    round_one = _gold(tmp_path / "round1.csv")
    round_two_frame = pd.read_csv(round_one).iloc[:3].copy()
    round_two = tmp_path / "round2.csv"
    round_two_frame.to_csv(round_two, index=False)
    annotation_report = annotation_agreement(
        round_one,
        round_two,
        tmp_path / "annotation_agreement.json",
    )

    human_rows = []
    judge_rows = []
    key_rows = []
    for index, score in enumerate((0, 1, 2)):
        critical_tags = "unsafe_data_request" if index == 0 else ""
        human_rows.append(
            {
                "review_id": f"review-{index}",
                "case_id": f"case-{index}",
                "draft_reply": "reply",
                **{dimension: str(score) for dimension in RATING_DIMENSIONS},
                "critical_error_tags": critical_tags,
                "overall_pass": "PASS" if score == 2 else "FAIL",
                "reviewer_id": "human_a",
                "review_notes": (
                    "Synthetic control contains an unsafe data request."
                    if index == 0
                    else ""
                ),
            }
        )
        judge_rows.append(
            {
                "review_id": f"review-{index}",
                **{dimension: score for dimension in RATING_DIMENSIONS},
                "critical_error_tags": critical_tags,
                "overall_pass": "PASS" if score == 2 else "FAIL",
                "rationale": "rated",
                "judge_model": "judge-test",
                "replicate": 1,
            }
        )
        key_rows.append(
            {
                "review_id": f"review-{index}",
                "case_id": f"case-{index}",
                "sample_slice": "natural",
                "row_role": "control" if index == 0 else "compared",
                "system_name": "__judge_control__" if index == 0 else "a",
                "control_name": "unsafe_public_data_and_refund" if index == 0 else "",
            }
        )
    human_path = tmp_path / "human.csv"
    judge_path = tmp_path / "judge.csv"
    key_path = tmp_path / "key.csv"
    pd.DataFrame(human_rows).to_csv(human_path, index=False)
    pd.DataFrame(judge_rows).to_csv(judge_path, index=False)
    pd.DataFrame(key_rows).to_csv(key_path, index=False)
    judge_report = judge_human_agreement(
        human_path,
        [judge_path],
        tmp_path / "judge_agreement.json",
        identity_key_path=key_path,
    )

    assert annotation_report["overlap_count"] == 3
    assert annotation_report["fields"]["intent_label"]["cohen_kappa"] == 1
    assert judge_report["comparisons"]["judge.csv"]["row_count"] == 2
    assert judge_report["comparisons"]["judge.csv"]["overall_pass_cohen_kappa"] == 1
    assert judge_report["judge_decoy_controls"]["judge.csv"]["fail_rate"] == 1
    assert (
        judge_report["judge_decoy_controls"]["judge.csv"][
            "critical_error_detection_rate"
        ]
        == 1
    )
    assert not judge_report["judge_advisory_trust_gate"]["passed"]
    assert "judge_replicate_count" in judge_report["judge_advisory_trust_gate"][
        "failed_checks"
    ]


def test_judge_output_validator_rejects_changed_provenance(tmp_path: Path) -> None:
    review_path = tmp_path / "review.csv"
    judge_path = tmp_path / "judge.csv"
    manifest_path = tmp_path / "judge.csv.manifest.json"
    pd.DataFrame([{"review_id": "r1"}]).to_csv(review_path, index=False)
    pd.DataFrame(
        [
            {
                "review_id": "r1",
                **{dimension: 2 for dimension in RATING_DIMENSIONS},
                "critical_error_tags": "",
                "overall_pass": "PASS",
                "rationale": "Acceptable.",
                "judge_model": "judge-test",
                "replicate": 1,
            }
        ]
    ).to_csv(judge_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "output_sha256": "wrong",
                "review_sha256": "wrong",
                "model": "judge-test",
                "replicate": 1,
                "row_count": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output hash"):
        validate_judge_output(
            judge_path,
            review_path=review_path,
            manifest_path=manifest_path,
        )
