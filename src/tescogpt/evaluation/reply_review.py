"""System-blinded human reply-review packet and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tescogpt.evaluation.labels import validate_annotations

RATING_DIMENSIONS = (
    "issue_understanding",
    "helpfulness_actionability",
    "evidence_grounding",
    "tone_empathy",
    "privacy_safety",
    "routing_fit",
)

CRITICAL_ERROR_TAGS = (
    "irrelevant_or_misunderstood",
    "unsupported_claim_or_action",
    "unsafe_data_request",
    "missed_safety_risk",
    "wrong_route",
    "obsolete_policy_or_link",
    "other_critical_error",
)

REVIEW_LABEL_COLUMNS = (
    *RATING_DIMENSIONS,
    "critical_error_tags",
    "overall_pass",
    "reviewer_id",
    "review_notes",
)

REVIEW_CONTEXT_COLUMNS = (
    "review_order",
    "review_id",
    "case_id",
    "message",
    "prior_context",
    "gold_intent",
    "gold_handling",
    "gold_reason",
    "must_include",
    "must_avoid",
    "draft_reply",
    "proposed_handling",
    "proposed_reason",
    "evidence_quotes",
)


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _allocate_stratified_counts(sizes: pd.Series, total: int) -> pd.Series:
    exact = sizes / int(sizes.sum()) * total
    allocated = exact.astype(int)
    for name in (exact - allocated).sort_values(ascending=False).index[
        : total - int(allocated.sum())
    ]:
        allocated.loc[name] += 1
    return allocated


def initialize_reply_review(
    gold_path: str | Path,
    registry_path: str | Path,
    prediction_paths: list[str | Path],
    review_path: str | Path,
    key_path: str | Path,
    manifest_path: str | Path,
    *,
    case_count: int = 30,
    seed: int = 20260912,
) -> dict[str, Any]:
    """Create a stratified case-by-system review with system identity separated."""
    if len(prediction_paths) < 2:
        raise ValueError("Reply review requires at least two compared systems")
    validate_annotations(gold_path, require_complete=True, require_blind=True)
    gold = pd.read_csv(gold_path, dtype="string", keep_default_na=False)
    registry = pd.read_csv(registry_path, dtype="string", keep_default_na=False)
    registry = registry.loc[:, ["case_id", "sample_slice"]]
    labelled = gold.merge(registry, on="case_id", validate="one_to_one")
    if not 0 < case_count <= len(labelled):
        raise ValueError("case_count must be between 1 and the gold-set size")

    rng = np.random.default_rng(seed)
    allocations = _allocate_stratified_counts(
        labelled["sample_slice"].value_counts().sort_index(), case_count
    )
    selected_parts = []
    for slice_name, count in allocations.items():
        group = labelled.loc[labelled["sample_slice"].eq(slice_name)]
        indices = rng.choice(group.index.to_numpy(), size=int(count), replace=False)
        selected_parts.append(group.loc[indices].copy())
    selected = pd.concat(selected_parts, ignore_index=True)

    prediction_frames = []
    for path_value in prediction_paths:
        path = Path(path_value)
        frame = pd.read_csv(path, dtype="string", keep_default_na=False)
        if frame["case_id"].duplicated().any():
            raise ValueError(f"Duplicate prediction case IDs in {path}")
        names = frame["system_name"].unique()
        if len(names) != 1:
            raise ValueError(f"Prediction file must contain one system: {path}")
        if not set(selected["case_id"]).issubset(set(frame["case_id"])):
            raise ValueError(f"Prediction file does not cover every review case: {path}")
        prediction_frames.append(frame.set_index("case_id", drop=False))

    rows: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    review_counter = 1
    for case in selected.to_dict(orient="records"):
        system_order = rng.permutation(len(prediction_frames))
        for frame_index in system_order:
            prediction = prediction_frames[int(frame_index)].loc[case["case_id"]]
            review_id = f"reply-review-{review_counter:03d}"
            rows.append(
                {
                    "review_order": review_counter,
                    "review_id": review_id,
                    "case_id": case["case_id"],
                    "message": case["message"],
                    "prior_context": case.get("prior_context", ""),
                    "gold_intent": case["intent_label"],
                    "gold_handling": case["handling_label"],
                    "gold_reason": case["reason_code"],
                    "must_include": case["must_include"],
                    "must_avoid": case["must_avoid"],
                    "draft_reply": prediction["draft_reply"],
                    "proposed_handling": prediction["handling_decision"],
                    "proposed_reason": prediction["decision_reason"],
                    "evidence_quotes": prediction.get("evidence_quotes", "[]"),
                    **{column: "" for column in REVIEW_LABEL_COLUMNS},
                }
            )
            keys.append(
                {
                    "review_id": review_id,
                    "case_id": case["case_id"],
                    "sample_slice": case["sample_slice"],
                    "system_name": prediction["system_name"],
                }
            )
            review_counter += 1
    permutation = rng.permutation(len(rows))
    review = pd.DataFrame(rows).iloc[permutation].reset_index(drop=True)
    review["review_order"] = np.arange(1, len(review) + 1)
    review = review.loc[:, [*REVIEW_CONTEXT_COLUMNS, *REVIEW_LABEL_COLUMNS]]
    key = pd.DataFrame(keys)
    review_file = Path(review_path)
    key_file = Path(key_path)
    manifest_file = Path(manifest_path)
    for path in (review_file, key_file, manifest_file):
        path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(review_file, index=False, lineterminator="\n")
    key.to_csv(key_file, index=False, lineterminator="\n")
    validate_reply_ratings(review_file)
    manifest = {
        "reply_review_schema_version": 1,
        "label_status": "UNLABELED",
        "seed": seed,
        "case_count": case_count,
        "system_count": len(prediction_frames),
        "review_row_count": len(review),
        "slice_case_counts": {
            str(name): int(count) for name, count in allocations.items()
        },
        "review_file": review_file.name,
        "review_sha256": _sha256(review_file),
        "identity_key_file": key_file.name,
        "identity_key_sha256": _sha256(key_file),
        "gold_file": Path(gold_path).name,
        "gold_sha256": _sha256(Path(gold_path)),
        "prediction_files": [Path(path).name for path in prediction_paths],
        "prediction_sha256": {
            Path(path).name: _sha256(Path(path)) for path in prediction_paths
        },
    }
    manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def validate_reply_ratings(
    path: str | Path,
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    source = Path(path)
    ratings = pd.read_csv(source, dtype="string", keep_default_na=False)
    required = {"review_id", "case_id", "draft_reply", *REVIEW_LABEL_COLUMNS}
    missing = sorted(required - set(ratings.columns))
    if missing:
        raise ValueError(f"Reply review is missing columns: {', '.join(missing)}")
    if "system_name" in ratings.columns or "sample_slice" in ratings.columns:
        raise ValueError("Blinded reply review must not expose system or slice identity")
    if ratings["review_id"].duplicated().any():
        raise ValueError("Reply review contains duplicate review IDs")

    problems: list[str] = []
    dimension_populated = pd.DataFrame(index=ratings.index)
    for dimension in RATING_DIMENSIONS:
        values = ratings[dimension].str.strip()
        invalid = sorted(set(values.loc[values.ne("")]) - {"0", "1", "2"})
        if invalid:
            problems.append(f"invalid {dimension}: {', '.join(invalid)}")
        dimension_populated[dimension] = values.ne("")
    overall = ratings["overall_pass"].str.strip()
    invalid_overall = sorted(set(overall.loc[overall.ne("")]) - {"PASS", "FAIL"})
    if invalid_overall:
        problems.append(f"invalid overall_pass: {', '.join(invalid_overall)}")
    invalid_tags = sorted(
        {
            tag.strip()
            for value in ratings["critical_error_tags"]
            for tag in value.split(";")
            if tag.strip() and tag.strip() not in CRITICAL_ERROR_TAGS
        }
    )
    if invalid_tags:
        problems.append("invalid critical_error_tags: " + ", ".join(invalid_tags))
    reviewers = ratings["reviewer_id"].str.strip()
    completed = dimension_populated.all(axis=1) & overall.ne("") & reviewers.ne("")
    any_labels = (
        ratings.loc[:, list(REVIEW_LABEL_COLUMNS)]
        .apply(lambda col: col.str.strip())
        .ne("")
        .any(axis=1)
    )
    partial = any_labels & ~completed
    if partial.any():
        problems.append(f"{int(partial.sum())} partially completed review rows")
    if require_complete and not completed.all():
        problems.append(f"{int((~completed).sum())} review rows are not complete")
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "file": source.name,
        "row_count": len(ratings),
        "completed_count": int(completed.sum()),
        "remaining_count": int((~completed).sum()),
        "is_complete": bool(completed.all()),
        "reviewer_ids": sorted(set(reviewers.loc[reviewers.ne("")])),
    }
