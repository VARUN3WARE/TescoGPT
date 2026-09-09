"""Schema validation for human annotations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

INTENT_LABELS = (
    "delivery_or_collection",
    "product_quality_or_safety",
    "product_availability",
    "pricing_promotion_or_clubcard",
    "refund_return_or_exchange",
    "online_account_or_checkout",
    "store_or_staff_experience",
    "product_information",
    "feedback_praise_or_suggestion",
    "other_or_unclear",
)

AUTO_REASON_CODES = (
    "SAFE_PUBLIC_GUIDANCE",
    "SAFE_CLARIFICATION",
    "NO_ACTION_NEEDED",
)

ESCALATION_REASON_CODES = (
    "FOOD_SAFETY_OR_INJURY",
    "ACCOUNT_OR_ORDER_LOOKUP",
    "MONEY_OR_COMMITMENT",
    "PERSONAL_DATA_OR_PRIVATE_CHANNEL",
    "IMAGE_OR_MISSING_CONTEXT",
    "CURRENT_POLICY_OR_LIVE_INFO",
    "REPEATED_FAILURE_OR_DISTRESS",
    "OUT_OF_SCOPE_OR_UNCLEAR",
)

RISK_TAGS = (
    "food_safety",
    "injury_or_allergy",
    "money",
    "personal_data",
    "account_or_order",
    "image_required",
    "live_information",
    "repeated_failure",
    "strong_distress",
    "missing_context",
    "unsupported_language",
    "multi_intent",
)

LABEL_COLUMNS = (
    "intent_label",
    "secondary_intent",
    "handling_label",
    "reason_code",
    "risk_tags",
    "must_include",
    "must_avoid",
    "annotator_id",
    "annotation_notes",
)

ANNOTATION_CONTEXT_COLUMNS = (
    "display_order",
    "case_id",
    "conversation_id",
    "tweet_id",
    "created_at",
    "message",
    "prior_context",
)

SAMPLING_METADATA_COLUMNS = ("sample_slice", "challenge_flags")


def _values(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _invalid_values(values: pd.Series, allowed: tuple[str, ...]) -> list[str]:
    populated = values.loc[values.ne("")]
    return sorted(set(populated) - set(allowed))


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def validate_annotations(
    path: str | Path,
    require_complete: bool = False,
    require_blind: bool = False,
) -> dict[str, Any]:
    """Validate an annotation CSV and report completion without changing it."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Annotation sheet does not exist: {source}")
    annotations = pd.read_csv(source, dtype="string", keep_default_na=False)
    required = {"case_id", "conversation_id", "tweet_id", "message", *LABEL_COLUMNS}
    missing = sorted(required - set(annotations.columns))
    if missing:
        raise ValueError(f"Annotation sheet is missing columns: {', '.join(missing)}")
    if annotations["case_id"].duplicated().any():
        raise ValueError("Annotation sheet contains duplicate case IDs")
    if annotations["conversation_id"].duplicated().any():
        raise ValueError("Annotation sheet must contain at most one case per conversation")
    if "historical_reply" in annotations.columns:
        raise ValueError("Blind annotation sheet must not expose historical_reply")
    exposed_metadata = sorted(set(SAMPLING_METADATA_COLUMNS) & set(annotations.columns))
    if require_blind and exposed_metadata:
        raise ValueError(
            "Blind annotation sheet exposes sampling metadata: "
            + ", ".join(exposed_metadata)
        )

    intents = _values(annotations["intent_label"])
    secondary = _values(annotations["secondary_intent"])
    handling = _values(annotations["handling_label"])
    reasons = _values(annotations["reason_code"])
    annotators = _values(annotations["annotator_id"])

    invalid_intents = _invalid_values(intents, INTENT_LABELS)
    invalid_secondary = _invalid_values(secondary, INTENT_LABELS)
    invalid_handling = _invalid_values(handling, ("AUTO_HANDLE", "ESCALATE"))
    invalid_reasons = _invalid_values(
        reasons,
        (*AUTO_REASON_CODES, *ESCALATION_REASON_CODES),
    )
    invalid_risks = sorted(
        {
            tag.strip()
            for value in _values(annotations["risk_tags"])
            for tag in value.split(";")
            if tag.strip() and tag.strip() not in RISK_TAGS
        }
    )
    problems: list[str] = []
    for name, values in (
        ("intent_label", invalid_intents),
        ("secondary_intent", invalid_secondary),
        ("handling_label", invalid_handling),
        ("reason_code", invalid_reasons),
        ("risk_tags", invalid_risks),
    ):
        if values:
            problems.append(f"invalid {name}: {', '.join(values)}")

    same_intent = intents.ne("") & intents.eq(secondary)
    if same_intent.any():
        problems.append("secondary_intent duplicates primary intent")
    auto_wrong_reason = handling.eq("AUTO_HANDLE") & ~reasons.isin(AUTO_REASON_CODES)
    escalate_wrong_reason = handling.eq("ESCALATE") & ~reasons.isin(ESCALATION_REASON_CODES)
    if auto_wrong_reason.any():
        problems.append("AUTO_HANDLE row uses a non-automatic reason code")
    if escalate_wrong_reason.any():
        problems.append("ESCALATE row uses a non-escalation reason code")

    completed = intents.ne("") & handling.ne("") & reasons.ne("") & annotators.ne("")
    partially_completed = (
        annotations.loc[:, list(LABEL_COLUMNS)].apply(_values).ne("").any(axis=1) & ~completed
    )
    if partially_completed.any():
        problems.append(f"{int(partially_completed.sum())} partially completed rows")
    if require_complete and not completed.all():
        problems.append(f"{int((~completed).sum())} rows are not completely labelled")
    if problems:
        raise ValueError("; ".join(problems))

    return {
        "file": source.name,
        "row_count": len(annotations),
        "completed_count": int(completed.sum()),
        "remaining_count": int((~completed).sum()),
        "completion_pct": round(100 * float(completed.mean()), 2) if len(annotations) else 0.0,
        "annotator_ids": sorted(set(annotators.loc[annotators.ne("")])),
        "is_complete": bool(completed.all()),
    }


def initialize_annotation_rounds(
    candidates_path: str | Path,
    round_one_path: str | Path,
    round_two_path: str | Path,
    manifest_path: str | Path,
    second_annotation_count: int = 60,
    seed: int = 20260910,
) -> dict[str, Any]:
    """Create independent round-one and stratified round-two blank sheets."""
    candidates_file = Path(candidates_path)
    round_one_file = Path(round_one_path)
    round_two_file = Path(round_two_path)
    manifest_file = Path(manifest_path)
    progress = validate_annotations(candidates_file)
    if progress["completed_count"]:
        raise ValueError("Candidate sheet already contains labels")

    candidates = pd.read_csv(candidates_file, dtype="string", keep_default_na=False)
    if not 0 < second_annotation_count <= len(candidates):
        raise ValueError("second_annotation_count must be between 1 and the candidate count")
    if "sample_slice" not in candidates.columns:
        raise ValueError("Candidate sheet must include sample_slice for stratified selection")

    rng = np.random.default_rng(seed)
    slice_sizes = candidates["sample_slice"].value_counts().sort_index()
    exact_allocations = slice_sizes / len(candidates) * second_annotation_count
    allocations = exact_allocations.astype(int)
    remainder = second_annotation_count - int(allocations.sum())
    if remainder:
        fractional = (exact_allocations - allocations).sort_values(ascending=False)
        for slice_name in fractional.index[:remainder]:
            allocations.loc[slice_name] += 1

    round_two_parts: list[pd.DataFrame] = []
    for slice_name, count in allocations.items():
        group = candidates.loc[candidates["sample_slice"].eq(slice_name)]
        indices = rng.choice(group.index.to_numpy(), size=int(count), replace=False)
        round_two_parts.append(group.loc[indices].copy())
    selected_round_two = pd.concat(round_two_parts, ignore_index=True)
    selected_round_two = selected_round_two.iloc[
        rng.permutation(len(selected_round_two))
    ].reset_index(drop=True)
    selected_round_two["display_order"] = np.arange(1, len(selected_round_two) + 1)

    annotation_columns = [*ANNOTATION_CONTEXT_COLUMNS, *LABEL_COLUMNS]
    missing_annotation_columns = sorted(set(annotation_columns) - set(candidates.columns))
    if missing_annotation_columns:
        raise ValueError(
            "Candidate sheet is missing annotation columns: "
            + ", ".join(missing_annotation_columns)
        )
    round_one = candidates.loc[:, annotation_columns].copy()
    round_two = selected_round_two.loc[:, annotation_columns].copy()

    for path in (round_one_file, round_two_file, manifest_file):
        path.parent.mkdir(parents=True, exist_ok=True)
    round_one.to_csv(round_one_file, index=False, lineterminator="\n")
    round_two.to_csv(round_two_file, index=False, lineterminator="\n")
    validate_annotations(round_one_file, require_blind=True)
    validate_annotations(round_two_file, require_blind=True)

    manifest: dict[str, Any] = {
        "annotation_round_schema_version": 1,
        "label_status": "UNLABELED",
        "seed": seed,
        "candidate_file": candidates_file.name,
        "candidate_sha256": _sha256(candidates_file),
        "round_one_file": round_one_file.name,
        "round_one_count": len(candidates),
        "round_one_sha256": _sha256(round_one_file),
        "round_two_file": round_two_file.name,
        "round_two_count": len(round_two),
        "round_two_sha256": _sha256(round_two_file),
        "round_two_slice_counts": {
            str(key): int(value)
            for key, value in selected_round_two["sample_slice"].value_counts().items()
        },
        "round_two_case_ids": sorted(round_two["case_id"].astype(str)),
    }
    manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
