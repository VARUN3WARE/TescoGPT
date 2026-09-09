"""Label-free static safety audit for committed prediction files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from tescogpt.policy.safety import inspect_reply


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def audit_prediction_safety(
    prediction_paths: list[str | Path],
    output_path: str | Path,
    details_directory: str | Path,
) -> dict[str, Any]:
    """Inspect drafts without treating regex warnings as human quality labels."""
    details_dir = Path(details_directory)
    details_dir.mkdir(parents=True, exist_ok=True)
    systems: dict[str, Any] = {}
    for prediction_path in prediction_paths:
        path = Path(prediction_path)
        predictions = pd.read_csv(path, dtype="string", keep_default_na=False)
        required = {"case_id", "system_name", "draft_reply", "handling_decision"}
        missing = sorted(required - set(predictions.columns))
        if missing:
            raise ValueError(f"Predictions are missing columns: {', '.join(missing)}")
        names = predictions["system_name"].unique()
        if len(names) != 1:
            raise ValueError(f"Prediction file must contain one system: {path}")
        system_name = str(names[0])
        flag_values = predictions["draft_reply"].map(inspect_reply)
        detail = predictions.loc[
            :, ["case_id", "system_name", "handling_decision", "draft_reply"]
        ].copy()
        detail["static_warning_flags"] = flag_values.map(json.dumps)
        detail["static_warning_count"] = flag_values.map(len)
        detail_path = details_dir / f"{path.stem}_static_safety.csv"
        detail.to_csv(detail_path, index=False, lineterminator="\n")
        counts: dict[str, int] = {}
        for flags in flag_values:
            for flag in flags:
                counts[flag] = counts.get(flag, 0) + 1
        flagged = flag_values.map(bool)
        auto = predictions["handling_decision"].eq("AUTO_HANDLE")
        systems[system_name] = {
            "prediction_file": path.name,
            "prediction_sha256": _sha256(path),
            "row_count": len(predictions),
            "flagged_draft_count": int(flagged.sum()),
            "flagged_draft_rate": float(flagged.mean()),
            "flag_counts": dict(sorted(counts.items())),
            "auto_handled_count": int(auto.sum()),
            "flagged_auto_handled_count": int((flagged & auto).sum()),
            "details_file": detail_path.name,
            "details_sha256": _sha256(detail_path),
        }

    report = {
        "static_safety_schema_version": 1,
        "warning": "Regex warnings are deterministic checks, not human reply-quality labels.",
        "systems": systems,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report
