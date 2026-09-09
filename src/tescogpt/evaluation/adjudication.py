"""Human adjudication workflow for the independently labelled overlap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tescogpt.evaluation.labels import (
    AUTO_REASON_CODES,
    ESCALATION_REASON_CODES,
    INTENT_LABELS,
    validate_annotations,
)

ADJUDICATED_FIELDS = ("intent_label", "handling_label", "reason_code")
ADJUDICATION_COLUMNS = (
    "display_order",
    "case_id",
    "conversation_id",
    "tweet_id",
    "message",
    "prior_context",
    "round1_intent_label",
    "round2_intent_label",
    "final_intent_label",
    "round1_handling_label",
    "round2_handling_label",
    "final_handling_label",
    "round1_reason_code",
    "round2_reason_code",
    "final_reason_code",
    "disagreement_fields",
    "adjudicator_id",
    "adjudication_notes",
)
IMMUTABLE_ADJUDICATION_COLUMNS = tuple(
    column
    for column in ADJUDICATION_COLUMNS
    if column
    not in {
        "final_intent_label",
        "final_handling_label",
        "final_reason_code",
        "adjudicator_id",
        "adjudication_notes",
    }
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


def _load_frozen_rounds(
    round_one_path: Path,
    round_two_path: Path,
    round_manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validate_annotations(round_one_path, require_complete=True, require_blind=True)
    validate_annotations(round_two_path, require_complete=True, require_blind=True)
    manifest = json.loads(round_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("label_status") != "HUMAN_LABELED_UNADJUDICATED":
        raise ValueError("Independent annotation rounds must be frozen before adjudication")
    if manifest.get("round_one_sha256") != _sha256(round_one_path):
        raise ValueError("Round-one labels differ from the frozen annotation manifest")
    if manifest.get("round_two_sha256") != _sha256(round_two_path):
        raise ValueError("Round-two labels differ from the frozen annotation manifest")
    first = pd.read_csv(round_one_path, dtype="string", keep_default_na=False)
    second = pd.read_csv(round_two_path, dtype="string", keep_default_na=False)
    if not set(second["case_id"]).issubset(set(first["case_id"])):
        raise ValueError("Round two must remain a subset of round one")
    return first, second, manifest


def validate_adjudication(
    path: str | Path,
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    source = Path(path)
    frame = pd.read_csv(source, dtype="string", keep_default_na=False)
    missing = sorted(set(ADJUDICATION_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Adjudication sheet is missing columns: {', '.join(missing)}")
    if frame["case_id"].duplicated().any():
        raise ValueError("Adjudication sheet contains duplicate case IDs")
    if "historical_reply" in frame.columns:
        raise ValueError("Adjudication must not expose historical replies")

    final_intent = frame["final_intent_label"].str.strip()
    final_handling = frame["final_handling_label"].str.strip()
    final_reason = frame["final_reason_code"].str.strip()
    adjudicator = frame["adjudicator_id"].str.strip()
    invalid_intents = sorted(set(final_intent.loc[final_intent.ne("")]) - set(INTENT_LABELS))
    invalid_handling = sorted(
        set(final_handling.loc[final_handling.ne("")]) - {"AUTO_HANDLE", "ESCALATE"}
    )
    allowed_reasons = set(AUTO_REASON_CODES) | set(ESCALATION_REASON_CODES)
    invalid_reasons = sorted(set(final_reason.loc[final_reason.ne("")]) - allowed_reasons)
    problems = []
    if invalid_intents:
        problems.append("invalid final_intent_label: " + ", ".join(invalid_intents))
    if invalid_handling:
        problems.append("invalid final_handling_label: " + ", ".join(invalid_handling))
    if invalid_reasons:
        problems.append("invalid final_reason_code: " + ", ".join(invalid_reasons))

    disagreements = frame["disagreement_fields"].str.strip().ne("")
    complete_fields = final_intent.ne("") & final_handling.ne("") & final_reason.ne("")
    resolved = complete_fields & (~disagreements | adjudicator.ne(""))
    partial = adjudicator.ne("") & ~complete_fields
    if partial.any():
        problems.append(f"{int(partial.sum())} partially adjudicated rows")

    for field in ADJUDICATED_FIELDS:
        first = frame[f"round1_{field}"]
        second = frame[f"round2_{field}"]
        final = frame[f"final_{field}"]
        agreed = first.eq(second)
        changed_agreement = agreed & final.ne(first)
        if changed_agreement.any():
            problems.append(f"{int(changed_agreement.sum())} agreed {field} values changed")
    expected_disagreements = frame.apply(
        lambda row: ";".join(
            field
            for field in ADJUDICATED_FIELDS
            if row[f"round1_{field}"] != row[f"round2_{field}"]
        ),
        axis=1,
    )
    disagreement_mismatch = (
        expected_disagreements.astype("string")
        .ne(frame["disagreement_fields"].astype("string"))
        .any()
    )
    if disagreement_mismatch:
        problems.append("disagreement_fields does not match the source labels")
    auto_wrong = (
        final_handling.eq("AUTO_HANDLE")
        & final_reason.ne("")
        & ~final_reason.isin(AUTO_REASON_CODES)
    )
    escalate_wrong = (
        final_handling.eq("ESCALATE")
        & final_reason.ne("")
        & ~final_reason.isin(ESCALATION_REASON_CODES)
    )
    if auto_wrong.any() or escalate_wrong.any():
        problems.append("Final reason codes must be compatible with final handling")
    if require_complete and not resolved.all():
        problems.append(f"{int((~resolved).sum())} adjudication rows are unresolved")
    missing_notes = disagreements & frame["adjudication_notes"].str.strip().eq("")
    if require_complete and missing_notes.any():
        problems.append(
            f"{int(missing_notes.sum())} resolved disagreements lack adjudication notes"
        )
    adjudicator_ids = sorted(set(adjudicator.loc[adjudicator.ne("")]))
    if require_complete and disagreements.any() and len(adjudicator_ids) != 1:
        problems.append("completed disagreements must use exactly one adjudicator ID")
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "file": source.name,
        "row_count": len(frame),
        "disagreement_count": int(disagreements.sum()),
        "resolved_count": int(resolved.sum()),
        "remaining_count": int((~resolved).sum()),
        "is_complete": bool(resolved.all()),
        "adjudicator_ids": adjudicator_ids,
    }


def initialize_adjudication(
    round_one_path: str | Path,
    round_two_path: str | Path,
    round_manifest_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Create an overlap sheet with only disagreements left for human resolution."""
    round_one_file = Path(round_one_path)
    round_two_file = Path(round_two_path)
    round_manifest_file = Path(round_manifest_path)
    first, second, _ = _load_frozen_rounds(
        round_one_file, round_two_file, round_manifest_file
    )
    context_columns = (
        "case_id",
        "conversation_id",
        "tweet_id",
        "message",
        "prior_context",
    )
    merged = second.loc[:, [*context_columns, *ADJUDICATED_FIELDS]].merge(
        first.loc[:, ["case_id", *ADJUDICATED_FIELDS]],
        on="case_id",
        suffixes=("_two", "_one"),
        validate="one_to_one",
    )
    rows = []
    for display_order, row in enumerate(merged.to_dict(orient="records"), start=1):
        disagreements = [
            field
            for field in ADJUDICATED_FIELDS
            if row[f"{field}_one"] != row[f"{field}_two"]
        ]
        rows.append(
            {
                "display_order": display_order,
                "case_id": row["case_id"],
                "conversation_id": row["conversation_id"],
                "tweet_id": row["tweet_id"],
                "message": row["message"],
                "prior_context": row["prior_context"],
                **{
                    f"round1_{field}": row[f"{field}_one"]
                    for field in ADJUDICATED_FIELDS
                },
                **{
                    f"round2_{field}": row[f"{field}_two"]
                    for field in ADJUDICATED_FIELDS
                },
                **{
                    f"final_{field}": (
                        row[f"{field}_one"]
                        if row[f"{field}_one"] == row[f"{field}_two"]
                        else ""
                    )
                    for field in ADJUDICATED_FIELDS
                },
                "disagreement_fields": ";".join(disagreements),
                "adjudicator_id": "",
                "adjudication_notes": "",
            }
        )
    frame = pd.DataFrame(rows, columns=ADJUDICATION_COLUMNS)
    destination = Path(output_path)
    manifest_file = Path(manifest_path)
    for path in (destination, manifest_file):
        path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, lineterminator="\n")
    progress = validate_adjudication(destination)
    manifest = {
        "adjudication_schema_version": 1,
        "label_status": "UNADJUDICATED",
        "round_one_file": round_one_file.name,
        "round_one_sha256": _sha256(round_one_file),
        "round_two_file": round_two_file.name,
        "round_two_sha256": _sha256(round_two_file),
        "adjudication_file": destination.name,
        "adjudication_sha256": _sha256(destination),
        "adjudication_content_sha256": _frame_sha256(
            frame, IMMUTABLE_ADJUDICATION_COLUMNS
        ),
        "overlap_count": len(frame),
        "disagreement_count": progress["disagreement_count"],
    }
    manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def finalize_adjudication(
    round_one_path: str | Path,
    round_two_path: str | Path,
    round_manifest_path: str | Path,
    adjudication_path: str | Path,
    adjudication_manifest_path: str | Path,
    final_output_path: str | Path,
) -> dict[str, Any]:
    """Merge resolved overlap labels into a distinct, provenance-rich final gold file."""
    round_one_file = Path(round_one_path)
    round_two_file = Path(round_two_path)
    round_manifest_file = Path(round_manifest_path)
    first, second, round_manifest = _load_frozen_rounds(
        round_one_file, round_two_file, round_manifest_file
    )
    adjudication_file = Path(adjudication_path)
    progress = validate_adjudication(adjudication_file, require_complete=True)
    adjudication = pd.read_csv(
        adjudication_file, dtype="string", keep_default_na=False
    )
    adjudication_manifest_file = Path(adjudication_manifest_path)
    adjudication_manifest = json.loads(
        adjudication_manifest_file.read_text(encoding="utf-8")
    )
    if adjudication_manifest.get("round_one_sha256") != _sha256(round_one_file):
        raise ValueError("Adjudication manifest points to different round-one labels")
    if adjudication_manifest.get("round_two_sha256") != _sha256(round_two_file):
        raise ValueError("Adjudication manifest points to different round-two labels")
    if _frame_sha256(adjudication, IMMUTABLE_ADJUDICATION_COLUMNS) != (
        adjudication_manifest.get("adjudication_content_sha256")
    ):
        raise ValueError("Adjudication source labels or context changed")
    if set(adjudication["case_id"]) != set(second["case_id"]):
        raise ValueError("Adjudication IDs differ from the frozen overlap")

    final = first.copy()
    final["label_provenance"] = "round_one"
    final["adjudicator_id"] = ""
    final["adjudication_notes"] = ""
    final_by_id = final.set_index("case_id", drop=False)
    for row in adjudication.to_dict(orient="records"):
        case_id = row["case_id"]
        for field in ADJUDICATED_FIELDS:
            final_by_id.loc[case_id, field] = row[f"final_{field}"]
        if row["disagreement_fields"]:
            final_by_id.loc[case_id, "label_provenance"] = "adjudicated"
            final_by_id.loc[case_id, "adjudicator_id"] = row["adjudicator_id"]
            final_by_id.loc[case_id, "adjudication_notes"] = row[
                "adjudication_notes"
            ]
        else:
            final_by_id.loc[case_id, "label_provenance"] = "independent_agreement"
    final = final_by_id.reset_index(drop=True)
    final["display_order"] = np.arange(1, len(final) + 1)
    destination = Path(final_output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(destination, index=False, lineterminator="\n")
    validate_annotations(destination, require_complete=True, require_blind=True)

    adjudication_manifest.update(
        {
            "label_status": "HUMAN_LABELED_ADJUDICATED",
            "adjudication_sha256": _sha256(adjudication_file),
            "adjudicator_ids": progress["adjudicator_ids"],
            "final_file": destination.name,
            "final_sha256": _sha256(destination),
            "final_count": len(final),
        }
    )
    adjudication_manifest_file.write_text(
        json.dumps(adjudication_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    round_manifest.update(
        {
            "label_status": "HUMAN_LABELED_ADJUDICATED",
            "adjudication_file": adjudication_file.name,
            "adjudication_sha256": _sha256(adjudication_file),
            "adjudication_manifest_file": adjudication_manifest_file.name,
            "final_file": destination.name,
            "final_sha256": _sha256(destination),
        }
    )
    round_manifest_file.write_text(
        json.dumps(round_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return adjudication_manifest
