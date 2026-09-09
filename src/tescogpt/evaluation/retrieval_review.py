"""Blinded pooled relevance judgments for retrieval comparison."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tescogpt.retrieval.bm25 import BM25Index
from tescogpt.retrieval.outcome import OutcomeAwareRetriever

RELEVANCE_GRADES = ("0", "1", "2")
REVIEW_COLUMNS = (
    "query_order",
    "candidate_order",
    "retrieval_review_id",
    "query_case_id",
    "query_message",
    "query_prior_context",
    "candidate_message",
    "candidate_reply",
    "relevance_grade",
    "relevance_reason",
    "reviewer_id",
)
KEY_COLUMNS = (
    "retrieval_review_id",
    "query_case_id",
    "candidate_case_id",
    "sample_slice",
    "retrieved_by",
    "bm25_rank",
    "outcome_rank",
    "outcome_tier",
    "lexical_score",
    "rerank_score",
)
IMMUTABLE_REVIEW_COLUMNS = tuple(
    column
    for column in REVIEW_COLUMNS
    if column not in {"relevance_grade", "relevance_reason", "reviewer_id"}
)


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    payload = frame.loc[:, columns].to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _allocate(sizes: pd.Series, total: int) -> pd.Series:
    exact = sizes / int(sizes.sum()) * total
    allocated = exact.astype(int)
    remainder = total - int(allocated.sum())
    for name in (exact - allocated).sort_values(ascending=False).index[:remainder]:
        allocated.loc[name] += 1
    return allocated


def validate_retrieval_review(
    path: str | Path,
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    source = Path(path)
    review = pd.read_csv(source, dtype="string", keep_default_na=False)
    missing = sorted(set(REVIEW_COLUMNS) - set(review.columns))
    if missing:
        raise ValueError(f"Retrieval review is missing columns: {', '.join(missing)}")
    hidden = {"retrieved_by", "bm25_rank", "outcome_rank", "outcome_tier"}
    if hidden & set(review.columns):
        raise ValueError("Blinded retrieval review exposes system or outcome metadata")
    if review["retrieval_review_id"].duplicated().any():
        raise ValueError("Retrieval review IDs must be unique")
    grades = review["relevance_grade"].str.strip()
    invalid = sorted(set(grades.loc[grades.ne("")]) - set(RELEVANCE_GRADES))
    if invalid:
        raise ValueError("Invalid relevance grades: " + ", ".join(invalid))
    reviewers = review["reviewer_id"].str.strip()
    reasons = review["relevance_reason"].str.strip()
    completed = grades.ne("") & reviewers.ne("")
    partial = (grades.ne("") | reasons.ne("") | reviewers.ne("")) & ~completed
    problems = []
    if partial.any():
        problems.append(f"{int(partial.sum())} partially completed rows")
    if require_complete and not completed.all():
        problems.append(f"{int((~completed).sum())} rows are not complete")
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "file": source.name,
        "row_count": len(review),
        "query_count": int(review["query_case_id"].nunique()),
        "completed_count": int(completed.sum()),
        "remaining_count": int((~completed).sum()),
        "is_complete": bool(completed.all()),
        "reviewer_ids": sorted(set(reviewers.loc[reviewers.ne("")])),
    }


def validate_retrieval_review_key(
    review: pd.DataFrame,
    key: pd.DataFrame,
    *,
    top_k: int,
) -> None:
    missing = sorted(set(KEY_COLUMNS) - set(key.columns))
    if missing:
        raise ValueError(f"Retrieval identity key is missing columns: {', '.join(missing)}")
    if key["retrieval_review_id"].duplicated().any():
        raise ValueError("Retrieval identity key IDs must be unique")
    review_ids = set(review["retrieval_review_id"])
    key_ids = set(key["retrieval_review_id"])
    if review_ids != key_ids:
        raise ValueError("Retrieval review and identity key IDs do not match")
    if key.duplicated(["query_case_id", "candidate_case_id"]).any():
        raise ValueError("A candidate may appear only once per retrieval query")
    for system, rank_column in (("bm25", "bm25_rank"), ("outcome", "outcome_rank")):
        rank_values = key[rank_column].astype("string").fillna("")
        present = rank_values.str.strip().ne("")
        ranks = pd.to_numeric(key.loc[present, rank_column], errors="raise")
        if ((ranks % 1).ne(0) | ~ranks.between(1, top_k)).any():
            raise ValueError(f"{rank_column} must contain integer ranks from 1 to {top_k}")
        ranked = key.loc[present, ["query_case_id"]].assign(rank=ranks.to_numpy())
        if ranked.duplicated(["query_case_id", "rank"]).any():
            raise ValueError(f"{system} ranks must be unique within each query")
        named = key["retrieved_by"].str.split(";").map(
            lambda names, target=system: target in names
        )
        if (present.to_numpy() != named.to_numpy()).any():
            raise ValueError(f"{rank_column} and retrieved_by disagree")


def freeze_retrieval_review(
    review_path: str | Path,
    key_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Validate human judgments and update the frozen manifest without hiding edits."""
    review_file = Path(review_path)
    key_file = Path(key_path)
    manifest_file = Path(manifest_path)
    progress = validate_retrieval_review(review_file, require_complete=True)
    review = pd.read_csv(review_file, dtype="string", keep_default_na=False)
    key = pd.read_csv(key_file, dtype="string", keep_default_na=False)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    validate_retrieval_review_key(review, key, top_k=int(manifest["top_k"]))
    if _sha256(key_file) != manifest.get("identity_key_sha256"):
        raise ValueError("Retrieval identity key changed after initialization")
    content_hash = _frame_sha256(review, IMMUTABLE_REVIEW_COLUMNS)
    if content_hash != manifest.get("review_content_sha256"):
        raise ValueError("Retrieval review query or candidate content changed")
    manifest.update(
        {
            "label_status": "HUMAN_LABELED",
            "review_sha256": _sha256(review_file),
            "reviewer_ids": progress["reviewer_ids"],
        }
    )
    manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def initialize_retrieval_review(
    registry_path: str | Path,
    corpus_path: str | Path,
    review_path: str | Path,
    key_path: str | Path,
    manifest_path: str | Path,
    *,
    query_count: int = 25,
    top_k: int = 3,
    candidate_pool: int = 50,
    seed: int = 20260913,
) -> dict[str, Any]:
    """Pool two retrieval systems and hide ranks, systems, and outcome signals."""
    registry_file = Path(registry_path)
    corpus_file = Path(corpus_path)
    registry = pd.read_csv(registry_file, dtype="string", keep_default_na=False)
    corpus = pd.read_csv(corpus_file, dtype="string", keep_default_na=False)
    if "split" not in corpus.columns:
        raise ValueError("Retrieval corpus must contain an explicit split column")
    train = corpus.loc[corpus["split"].eq("train")].copy()
    if not 0 < query_count <= len(registry):
        raise ValueError("query_count must be between 1 and the registry size")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if candidate_pool < top_k:
        raise ValueError("candidate_pool must be at least top_k")
    if train.empty:
        raise ValueError("Retrieval corpus contains no train rows")

    rng = np.random.default_rng(seed)
    allocations = _allocate(registry["sample_slice"].value_counts().sort_index(), query_count)
    selected_parts = []
    for slice_name, count in allocations.items():
        group = registry.loc[registry["sample_slice"].eq(slice_name)]
        indices = rng.choice(group.index.to_numpy(), size=int(count), replace=False)
        selected_parts.append(group.loc[indices].copy())
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)

    bm25 = BM25Index(train)
    outcome = OutcomeAwareRetriever(train, candidate_pool=candidate_pool)
    train_by_id = train.assign(case_id=train["case_id"].astype(str)).set_index("case_id")
    review_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    review_number = 1
    for query_order, case in enumerate(selected.to_dict(orient="records"), start=1):
        query = f"{case.get('prior_context', '')} {case['message']}".strip()
        bm25_results = bm25.search(
            query,
            top_k=top_k,
            exclude_case_id=case["case_id"],
            exclude_conversation_id=case["conversation_id"],
        )
        outcome_results = outcome.search(
            query,
            top_k=top_k,
            exclude_case_id=case["case_id"],
            exclude_conversation_id=case["conversation_id"],
        )
        bm25_map = {result.case_id: (rank, result) for rank, result in enumerate(bm25_results, 1)}
        outcome_map = {
            result.case_id: (rank, result) for rank, result in enumerate(outcome_results, 1)
        }
        candidate_ids = sorted(set(bm25_map) | set(outcome_map))
        if not candidate_ids:
            raise ValueError(f"Neither retrieval system returned a candidate for {case['case_id']}")
        candidate_ids = [candidate_ids[index] for index in rng.permutation(len(candidate_ids))]
        for candidate_order, candidate_id in enumerate(candidate_ids, start=1):
            row = train_by_id.loc[candidate_id]
            review_id = f"retrieval-review-{review_number:03d}"
            review_rows.append(
                {
                    "query_order": query_order,
                    "candidate_order": candidate_order,
                    "retrieval_review_id": review_id,
                    "query_case_id": case["case_id"],
                    "query_message": case["message"],
                    "query_prior_context": case.get("prior_context", ""),
                    "candidate_message": row["message"],
                    "candidate_reply": row["historical_reply"],
                    "relevance_grade": "",
                    "relevance_reason": "",
                    "reviewer_id": "",
                }
            )
            bm25_item = bm25_map.get(candidate_id)
            outcome_item = outcome_map.get(candidate_id)
            outcome_result = outcome_item[1] if outcome_item else None
            key_rows.append(
                {
                    "retrieval_review_id": review_id,
                    "query_case_id": case["case_id"],
                    "candidate_case_id": candidate_id,
                    "sample_slice": case["sample_slice"],
                    "retrieved_by": ";".join(
                        name
                        for name, item in (("bm25", bm25_item), ("outcome", outcome_item))
                        if item is not None
                    ),
                    "bm25_rank": bm25_item[0] if bm25_item else "",
                    "outcome_rank": outcome_item[0] if outcome_item else "",
                    "outcome_tier": outcome_result.outcome_tier if outcome_result else "",
                    "lexical_score": (
                        outcome_result.lexical_score
                        if outcome_result
                        else bm25_item[1].score if bm25_item else ""
                    ),
                    "rerank_score": outcome_result.rerank_score if outcome_result else "",
                }
            )
            review_number += 1

    review = pd.DataFrame(review_rows, columns=REVIEW_COLUMNS)
    key = pd.DataFrame(key_rows)
    review_file = Path(review_path)
    key_file = Path(key_path)
    manifest_file = Path(manifest_path)
    for path in (review_file, key_file, manifest_file):
        path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(review_file, index=False, lineterminator="\n")
    key.to_csv(key_file, index=False, lineterminator="\n")
    validate_retrieval_review(review_file)
    validate_retrieval_review_key(review, key, top_k=top_k)
    manifest = {
        "retrieval_review_schema_version": 1,
        "label_status": "UNLABELED",
        "seed": seed,
        "query_count": query_count,
        "top_k": top_k,
        "candidate_pool": candidate_pool,
        "slice_query_counts": {
            str(name): int(count) for name, count in allocations.items()
        },
        "judgment_count": len(review),
        "review_file": review_file.name,
        "review_sha256": _sha256(review_file),
        "review_content_sha256": _frame_sha256(review, IMMUTABLE_REVIEW_COLUMNS),
        "identity_key_file": key_file.name,
        "identity_key_sha256": _sha256(key_file),
        "registry_sha256": _sha256(registry_file),
        "corpus_sha256": _sha256(corpus_file),
    }
    manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))


def evaluate_retrieval_review(
    review_path: str | Path,
    key_path: str | Path,
    output_path: str | Path,
    *,
    top_k: int = 3,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute pooled Precision@k and nDCG@k for both frozen rankings."""
    validate_retrieval_review(review_path, require_complete=True)
    review = pd.read_csv(review_path, dtype="string", keep_default_na=False)
    key = pd.read_csv(key_path, dtype="string", keep_default_na=False)
    validate_retrieval_review_key(review, key, top_k=top_k)
    if manifest_path is not None:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("label_status") != "HUMAN_LABELED":
            raise ValueError("Retrieval judgments must be frozen before evaluation")
        if int(manifest["top_k"]) != top_k:
            raise ValueError("Evaluation top_k differs from the frozen retrieval review")
        if _sha256(Path(review_path)) != manifest.get("review_sha256"):
            raise ValueError("Retrieval review hash differs from its frozen manifest")
        if _sha256(Path(key_path)) != manifest.get("identity_key_sha256"):
            raise ValueError("Retrieval identity key hash differs from its manifest")
    merged = review.loc[:, ["retrieval_review_id", "relevance_grade"]].merge(
        key,
        on="retrieval_review_id",
        validate="one_to_one",
    )
    merged["relevance_grade"] = pd.to_numeric(merged["relevance_grade"], errors="raise")
    systems: dict[str, Any] = {}
    per_query_scores: dict[str, dict[str, float]] = {"bm25": {}, "outcome": {}}
    query_ids = sorted(set(merged["query_case_id"]))
    for system, rank_column in (("bm25", "bm25_rank"), ("outcome", "outcome_rank")):
        ranked = merged.loc[merged[rank_column].ne("")].copy()
        ranked[rank_column] = pd.to_numeric(ranked[rank_column], errors="raise")
        ranked = ranked.loc[ranked[rank_column].le(top_k)]
        query_metrics = []
        for query_case_id in query_ids:
            group = ranked.loc[ranked["query_case_id"].eq(query_case_id)]
            ordered = group.sort_values(rank_column, kind="stable")
            grades = ordered["relevance_grade"].astype(int).tolist()
            ideal_pool = (
                merged.loc[merged["query_case_id"].eq(query_case_id), "relevance_grade"]
                .astype(int)
                .sort_values(ascending=False)
                .head(top_k)
                .tolist()
            )
            ideal_dcg = _dcg(ideal_pool)
            ndcg = _dcg(grades) / ideal_dcg if ideal_dcg else 0.0
            precision = sum(grade >= 1 for grade in grades) / top_k
            query_metrics.append((precision, ndcg))
            per_query_scores[system][str(query_case_id)] = ndcg
        systems[system] = {
            f"pooled_precision_at_{top_k}": float(
                np.mean([item[0] for item in query_metrics])
            ),
            f"pooled_ndcg_at_{top_k}": float(
                np.mean([item[1] for item in query_metrics])
            ),
            "query_count": len(query_metrics),
        }

    shared_queries = sorted(set(per_query_scores["bm25"]) & set(per_query_scores["outcome"]))
    differences = [
        per_query_scores["outcome"][query] - per_query_scores["bm25"][query]
        for query in shared_queries
    ]
    comparison = {
        "outcome_wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "bm25_wins": sum(value < 0 for value in differences),
        "mean_ndcg_difference_outcome_minus_bm25": float(np.mean(differences)),
    }
    report = {
        "retrieval_evaluation_schema_version": 1,
        "systems": systems,
        "paired_comparison": comparison,
        "important_limitation": (
            "Judgments cover the pooled top-k results, so corpus-wide recall is not estimable."
        ),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report
