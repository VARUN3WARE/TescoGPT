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
    "HUMAN_JUDGMENT_REQUIRED",
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

COMPLETION_COLUMNS = (
    "intent_label",
    "handling_label",
    "reason_code",
    "must_include",
    "must_avoid",
    "annotator_id",
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

    completed = (
        annotations.loc[:, list(COMPLETION_COLUMNS)]
        .apply(_values)
        .ne("")
        .all(axis=1)
    )
    partially_completed = (
        annotations.loc[:, list(LABEL_COLUMNS)].apply(_values).ne("").any(axis=1) & ~completed
    )
    if partially_completed.any():
        problems.append(f"{int(partially_completed.sum())} partially completed rows")
    if require_complete and not completed.all():
        problems.append(f"{int((~completed).sum())} rows are not completely labelled")
    unique_annotators = sorted(set(annotators.loc[annotators.ne("")]))
    if require_complete and len(unique_annotators) != 1:
        problems.append(
            "a completed annotation round must use exactly one annotator ID"
        )
    if problems:
        raise ValueError("; ".join(problems))

    return {
        "file": source.name,
        "row_count": len(annotations),
        "completed_count": int(completed.sum()),
        "remaining_count": int((~completed).sum()),
        "completion_pct": round(100 * float(completed.mean()), 2) if len(annotations) else 0.0,
        "annotator_ids": unique_annotators,
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


def _assert_context_unchanged(
    annotations: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    round_name: str,
) -> None:
    context_columns = (
        "case_id",
        "conversation_id",
        "tweet_id",
        "created_at",
        "message",
        "prior_context",
    )
    missing = sorted(
        set(context_columns) - set(annotations.columns)
        | set(context_columns) - set(candidates.columns)
    )
    if missing:
        raise ValueError(f"Missing frozen context columns: {', '.join(missing)}")
    annotation_context = (
        annotations.loc[:, context_columns]
        .astype(str)
        .sort_values("case_id", kind="stable")
        .reset_index(drop=True)
    )
    candidate_context = (
        candidates.loc[candidates["case_id"].isin(annotations["case_id"]), context_columns]
        .astype(str)
        .sort_values("case_id", kind="stable")
        .reset_index(drop=True)
    )
    if len(annotation_context) != len(candidate_context):
        raise ValueError(f"{round_name} contains a case outside the frozen candidates")
    changed = annotation_context.ne(candidate_context).any(axis=1)
    if changed.any():
        changed_ids = annotation_context.loc[changed, "case_id"].head(5).tolist()
        raise ValueError(
            f"{round_name} changed frozen message/context for: {', '.join(changed_ids)}"
        )


def validate_annotation_artifacts(
    candidates_path: str | Path,
    round_one_path: str | Path,
    round_two_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_gold_count: int,
    expected_overlap_count: int,
    final_path: str | Path | None = None,
) -> dict[str, Any]:
    """Revalidate preregistered sample sizes, identities, context, and independence."""
    if expected_gold_count <= 0 or expected_overlap_count <= 0:
        raise ValueError("Expected annotation sample sizes must be positive")
    if expected_overlap_count > expected_gold_count:
        raise ValueError("Expected overlap cannot exceed the golden-set size")

    candidates_file = Path(candidates_path)
    round_one_file = Path(round_one_path)
    round_two_file = Path(round_two_path)
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    status = str(manifest.get("label_status", ""))
    require_complete = status in {
        "HUMAN_LABELED_UNADJUDICATED",
        "HUMAN_LABELED_ADJUDICATED",
    }
    validate_annotations(round_one_file, require_complete=require_complete, require_blind=True)
    validate_annotations(round_two_file, require_complete=require_complete, require_blind=True)
    candidates = pd.read_csv(candidates_file, dtype="string", keep_default_na=False)
    round_one = pd.read_csv(round_one_file, dtype="string", keep_default_na=False)
    round_two = pd.read_csv(round_two_file, dtype="string", keep_default_na=False)

    if len(candidates) != expected_gold_count or len(round_one) != expected_gold_count:
        raise ValueError("Golden annotation count differs from the preregistered size")
    if len(round_two) != expected_overlap_count:
        raise ValueError("Independent overlap count differs from the preregistered size")
    manifest_counts = {
        "round_one_count": expected_gold_count,
        "round_two_count": expected_overlap_count,
    }
    for field, expected in manifest_counts.items():
        if int(manifest.get(field, -1)) != expected:
            raise ValueError(f"Annotation manifest {field} differs from the preregistration")
    if manifest.get("candidate_sha256") != _sha256(candidates_file):
        raise ValueError("Annotation manifest points to a different candidate registry")
    if manifest.get("round_one_sha256") != _sha256(round_one_file):
        raise ValueError("Round-one annotations differ from their manifest")
    if manifest.get("round_two_sha256") != _sha256(round_two_file):
        raise ValueError("Round-two annotations differ from their manifest")
    if set(round_one["case_id"]) != set(candidates["case_id"]):
        raise ValueError("Round one case IDs must exactly match the frozen candidates")
    expected_round_two_ids = set(manifest.get("round_two_case_ids", []))
    if set(round_two["case_id"]) != expected_round_two_ids:
        raise ValueError("Round two case IDs differ from the frozen overlap")
    _assert_context_unchanged(round_one, candidates, round_name="Round one")
    _assert_context_unchanged(round_two, candidates, round_name="Round two")

    if require_complete:
        first_annotators = set(_values(round_one["annotator_id"])) - {""}
        second_annotators = set(_values(round_two["annotator_id"])) - {""}
        overlap = sorted(first_annotators & second_annotators)
        if overlap:
            raise ValueError("Independent rounds share annotator IDs: " + ", ".join(overlap))
        expected_first = set(manifest.get("round_one_annotator_ids", first_annotators))
        expected_second = set(manifest.get("round_two_annotator_ids", second_annotators))
        if first_annotators != expected_first or second_annotators != expected_second:
            raise ValueError("Annotation manifest annotator IDs differ from the label files")

    if status == "HUMAN_LABELED_ADJUDICATED":
        if final_path is None:
            raise ValueError("Adjudicated annotations require a distinct final artifact")
        final_file = Path(final_path)
        validate_annotations(final_file, require_complete=True, require_blind=True)
        final = pd.read_csv(final_file, dtype="string", keep_default_na=False)
        if len(final) != expected_gold_count or set(final["case_id"]) != set(
            candidates["case_id"]
        ):
            raise ValueError("Final annotations do not exactly cover the frozen candidates")
        if manifest.get("final_sha256") != _sha256(final_file):
            raise ValueError("Final annotations differ from their manifest")
        _assert_context_unchanged(final, candidates, round_name="Final annotations")

    return {
        "label_status": status,
        "gold_count": len(round_one),
        "overlap_count": len(round_two),
        "independent_annotators": require_complete,
        "final_validated": status == "HUMAN_LABELED_ADJUDICATED",
    }


def freeze_human_annotations(
    candidates_path: str | Path,
    round_one_path: str | Path,
    round_two_path: str | Path,
    manifest_path: str | Path,
    codebook_path: str | Path,
) -> dict[str, Any]:
    """Validate completed independent rounds and replace blank-sheet hashes."""
    candidate_file = Path(candidates_path)
    round_one_file = Path(round_one_path)
    round_two_file = Path(round_two_path)
    manifest_file = Path(manifest_path)
    codebook_file = Path(codebook_path)
    validate_annotations(round_one_file, require_complete=True, require_blind=True)
    validate_annotations(round_two_file, require_complete=True, require_blind=True)
    candidates = pd.read_csv(candidate_file, dtype="string", keep_default_na=False)
    round_one = pd.read_csv(round_one_file, dtype="string", keep_default_na=False)
    round_two = pd.read_csv(round_two_file, dtype="string", keep_default_na=False)
    if set(round_one["case_id"]) != set(candidates["case_id"]):
        raise ValueError("Round one case IDs must exactly match the frozen candidates")

    existing = json.loads(manifest_file.read_text(encoding="utf-8"))
    expected_round_two_ids = set(existing.get("round_two_case_ids", []))
    if set(round_two["case_id"]) != expected_round_two_ids:
        raise ValueError("Round two case IDs differ from the frozen overlap")
    if _sha256(candidate_file) != existing.get("candidate_sha256"):
        raise ValueError("Candidate file hash differs from the annotation-round manifest")
    _assert_context_unchanged(round_one, candidates, round_name="Round one")
    _assert_context_unchanged(round_two, candidates, round_name="Round two")

    round_one_annotators = set(_values(round_one["annotator_id"])) - {""}
    round_two_annotators = set(_values(round_two["annotator_id"])) - {""}
    overlap = sorted(round_one_annotators & round_two_annotators)
    if overlap:
        raise ValueError(
            "Independent rounds share annotator IDs: " + ", ".join(overlap)
        )

    existing.update(
        {
            "label_status": "HUMAN_LABELED_UNADJUDICATED",
            "round_one_sha256": _sha256(round_one_file),
            "round_two_sha256": _sha256(round_two_file),
            "round_one_annotator_ids": sorted(round_one_annotators),
            "round_two_annotator_ids": sorted(round_two_annotators),
            "codebook_file": codebook_file.name,
            "codebook_sha256": _sha256(codebook_file),
            "round_one_intent_counts": {
                str(label): int(count)
                for label, count in round_one["intent_label"].value_counts().sort_index().items()
            },
            "round_one_handling_counts": {
                str(label): int(count)
                for label, count in round_one["handling_label"].value_counts().sort_index().items()
            },
        }
    )
    manifest_file.write_text(
        json.dumps(existing, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return existing
