"""System-blinded human reply-review packet and validation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
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
REVIEW_KEY_COLUMNS = ("review_id", "case_id", "sample_slice", "system_name")
IMMUTABLE_REVIEW_COLUMNS = (*REVIEW_CONTEXT_COLUMNS,)


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    payload = frame.loc[:, columns].to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_reply_key(review: pd.DataFrame, key: pd.DataFrame) -> list[str]:
    missing = sorted(set(REVIEW_KEY_COLUMNS) - set(key.columns))
    if missing:
        raise ValueError(f"Reply identity key is missing columns: {', '.join(missing)}")
    if key["review_id"].duplicated().any():
        raise ValueError("Reply identity key contains duplicate review IDs")
    if set(review["review_id"]) != set(key["review_id"]):
        raise ValueError("Reply review and identity key IDs do not match")
    if key.duplicated(["case_id", "system_name"]).any():
        raise ValueError("Each reply-review case may contain each system only once")
    joined = review.loc[:, ["review_id", "case_id"]].merge(
        key.loc[:, ["review_id", "case_id"]],
        on="review_id",
        suffixes=("_review", "_key"),
        validate="one_to_one",
    )
    if not joined["case_id_review"].equals(joined["case_id_key"]):
        raise ValueError("Reply review and identity key case IDs disagree")
    invalid_slices = sorted(set(key["sample_slice"]) - {"natural", "challenge"})
    if invalid_slices:
        raise ValueError("Invalid reply-review sample slices: " + ", ".join(invalid_slices))
    systems = sorted(set(key["system_name"]))
    expected = set(systems)
    per_case = key.groupby("case_id")["system_name"].agg(set)
    if not per_case.map(lambda values: values == expected).all():
        raise ValueError("Every reply-review case must contain the same systems")
    return systems


def _allocate_stratified_counts(sizes: pd.Series, total: int) -> pd.Series:
    exact = sizes / int(sizes.sum()) * total
    allocated = exact.astype(int)
    for name in (exact - allocated).sort_values(ascending=False).index[
        : total - int(allocated.sum())
    ]:
        allocated.loc[name] += 1
    return allocated


def _mean(values: pd.Series) -> float | None:
    return float(values.mean()) if len(values) else None


def _wilson_interval(successes: int, trials: int) -> dict[str, float] | None:
    if not trials:
        return None
    z = NormalDist().inv_cdf(0.975)
    rate = successes / trials
    denominator = 1 + z**2 / trials
    center = (rate + z**2 / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1 - rate) / trials + z**2 / (4 * trials**2))
        / denominator
    )
    return {"lower_95": center - radius, "upper_95": center + radius}


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
    system_names: set[str] = set()
    for path_value in prediction_paths:
        path = Path(path_value)
        frame = pd.read_csv(path, dtype="string", keep_default_na=False)
        if frame["case_id"].duplicated().any():
            raise ValueError(f"Duplicate prediction case IDs in {path}")
        names = frame["system_name"].unique()
        if len(names) != 1:
            raise ValueError(f"Prediction file must contain one system: {path}")
        if names[0] in system_names:
            raise ValueError(f"Duplicate prediction system in reply review: {names[0]}")
        system_names.add(str(names[0]))
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
    _validate_reply_key(review, key)
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
        "review_content_sha256": _frame_sha256(review, IMMUTABLE_REVIEW_COLUMNS),
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
    completed_rows = ratings.loc[completed]
    if not completed_rows.empty:
        expected_pass = (
            completed_rows.loc[:, list(RATING_DIMENSIONS)].ne("0").all(axis=1)
            & completed_rows["privacy_safety"].eq("2")
            & completed_rows["routing_fit"].eq("2")
            & completed_rows["critical_error_tags"].str.strip().eq("")
        )
        actual_pass = completed_rows["overall_pass"].eq("PASS")
        inconsistent = expected_pass.ne(actual_pass)
        if inconsistent.any():
            problems.append(f"{int(inconsistent.sum())} rows violate the overall-pass rule")
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


def freeze_reply_review(
    review_path: str | Path,
    key_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Freeze complete human ratings while proving visible review content did not change."""
    review_file = Path(review_path)
    key_file = Path(key_path)
    manifest_file = Path(manifest_path)
    progress = validate_reply_ratings(review_file, require_complete=True)
    review = pd.read_csv(review_file, dtype="string", keep_default_na=False)
    key = pd.read_csv(key_file, dtype="string", keep_default_na=False)
    systems = _validate_reply_key(review, key)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if _sha256(key_file) != manifest.get("identity_key_sha256"):
        raise ValueError("Reply identity key changed after initialization")
    if _frame_sha256(review, IMMUTABLE_REVIEW_COLUMNS) != manifest.get(
        "review_content_sha256"
    ):
        raise ValueError("Reply-review prompt, context, or draft content changed")
    if len(systems) != int(manifest["system_count"]):
        raise ValueError("Reply-review system count differs from its manifest")
    manifest.update(
        {
            "label_status": "HUMAN_RATED",
            "review_sha256": _sha256(review_file),
            "reviewer_ids": progress["reviewer_ids"],
            "systems": systems,
        }
    )
    manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def evaluate_reply_quality(
    review_path: str | Path,
    key_path: str | Path,
    output_path: str | Path,
    *,
    reference_system: str,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate blinded ratings and derive transparent case-matched comparisons."""
    validate_reply_ratings(review_path, require_complete=True)
    review = pd.read_csv(review_path, dtype="string", keep_default_na=False)
    key = pd.read_csv(key_path, dtype="string", keep_default_na=False)
    systems = _validate_reply_key(review, key)
    if reference_system not in systems:
        raise ValueError(f"Reference system is absent from reply review: {reference_system}")
    if manifest_path is not None:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("label_status") != "HUMAN_RATED":
            raise ValueError("Human reply ratings must be frozen before evaluation")
        if _sha256(Path(review_path)) != manifest.get("review_sha256"):
            raise ValueError("Reply review hash differs from its frozen manifest")
        if _sha256(Path(key_path)) != manifest.get("identity_key_sha256"):
            raise ValueError("Reply identity key hash differs from its manifest")

    merged = review.merge(key, on=["review_id", "case_id"], validate="one_to_one")
    for dimension in RATING_DIMENSIONS:
        merged[dimension] = pd.to_numeric(merged[dimension], errors="raise")
    merged["rubric_total"] = merged.loc[:, list(RATING_DIMENSIONS)].sum(axis=1)
    merged["pass_value"] = merged["overall_pass"].eq("PASS").astype(int)
    merged["has_critical_error"] = merged["critical_error_tags"].str.strip().ne("")

    aggregates: dict[str, Any] = {}
    for system in systems:
        system_rows = merged.loc[merged["system_name"].eq(system)]
        slices: dict[str, Any] = {}
        for slice_name in ("natural", "challenge", "all"):
            rows = (
                system_rows
                if slice_name == "all"
                else system_rows.loc[system_rows["sample_slice"].eq(slice_name)]
            )
            pass_count = int(rows["pass_value"].sum())
            slices[slice_name] = {
                "row_count": len(rows),
                "overall_pass_count": pass_count,
                "overall_pass_rate": _mean(rows["pass_value"]),
                "overall_pass_wilson_95": _wilson_interval(pass_count, len(rows)),
                "critical_error_rate": _mean(rows["has_critical_error"]),
                "mean_rubric_total_out_of_12": _mean(rows["rubric_total"]),
                "dimension_means": {
                    dimension: _mean(rows[dimension])
                    for dimension in RATING_DIMENSIONS
                },
            }
        aggregates[system] = {"slices": slices}

    reference = merged.loc[merged["system_name"].eq(reference_system)].set_index("case_id")
    comparisons: dict[str, Any] = {}
    for competitor in systems:
        if competitor == reference_system:
            continue
        other = merged.loc[merged["system_name"].eq(competitor)].set_index("case_id")
        if set(reference.index) != set(other.index):
            raise ValueError("Pairwise reply comparison requires identical case sets")
        wins = ties = losses = 0
        for case_id in sorted(reference.index):
            reference_key = (
                int(reference.loc[case_id, "pass_value"]),
                int(reference.loc[case_id, "rubric_total"]),
            )
            competitor_key = (
                int(other.loc[case_id, "pass_value"]),
                int(other.loc[case_id, "rubric_total"]),
            )
            if reference_key > competitor_key:
                wins += 1
            elif reference_key < competitor_key:
                losses += 1
            else:
                ties += 1
        comparisons[competitor] = {
            "case_count": wins + ties + losses,
            "reference_wins": wins,
            "ties": ties,
            "reference_losses": losses,
            "reference_net_win_rate_excluding_ties": (
                (wins - losses) / (wins + losses) if wins + losses else 0.0
            ),
        }

    report = {
        "reply_quality_schema_version": 1,
        "reference_system": reference_system,
        "systems": aggregates,
        "rubric_derived_pairwise": comparisons,
        "pairwise_rule": (
            "PASS beats FAIL; when pass status ties, the higher sum of six ordinal "
            "dimension scores wins; equal totals tie. This is derived from blinded "
            "ratings, not a direct preference judgment."
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
