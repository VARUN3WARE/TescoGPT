import json
from pathlib import Path

import pandas as pd
import pytest

from tescogpt.evaluation.reproduce import (
    _validate_generation_provenance,
    _write_reproduction_report,
)


def _manifest() -> dict:
    request_hash = "a" * 64
    return {
        "model": "test-model",
        "generation_provenance": {
            "provider": "openai",
            "requested_model": "test-model",
            "resolved_models": ["test-model-2026-09-01"],
            "instructions_sha256": "b" * 64,
            "schema_sha256": "c" * 64,
            "request_count": 1,
            "unique_request_count": 1,
            "api_call_count": 1,
            "cache_hit_count": 0,
            "requests": [
                {
                    "case_id": "case-1",
                    "request_sha256": request_hash,
                    "cache_hit": False,
                    "response_id": "response-1",
                    "response_model": "test-model-2026-09-01",
                }
            ],
        },
    }


def test_api_prediction_provenance_covers_every_output_row() -> None:
    predictions = pd.DataFrame([{"case_id": "case-1"}])

    _validate_generation_provenance(_manifest(), predictions, Path("predictions.csv"))


def test_api_prediction_provenance_rejects_changed_case_identity() -> None:
    predictions = pd.DataFrame([{"case_id": "different-case"}])

    with pytest.raises(ValueError, match="request IDs differ"):
        _validate_generation_provenance(
            _manifest(),
            predictions,
            Path("predictions.csv"),
        )


def test_api_prediction_provenance_reconciles_cache_and_call_counts() -> None:
    predictions = pd.DataFrame([{"case_id": "case-1"}])
    manifest = _manifest()
    manifest["generation_provenance"]["cache_hit_count"] = 1

    with pytest.raises(ValueError, match="cache-hit count"):
        _validate_generation_provenance(
            manifest,
            predictions,
            Path("predictions.csv"),
        )


def test_reproduction_runtime_gate_persists_failure_without_dynamic_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "reproduction.json"
    monkeypatch.setattr(
        "tescogpt.evaluation.reproduce.time.perf_counter",
        lambda: 901.0,
    )

    with pytest.raises(RuntimeError, match="15-minute"):
        _write_reproduction_report(
            {"status": "COMPLETE"},
            output,
            started=0.0,
        )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["runtime_limit_seconds"] == 900
    assert persisted["under_15_minutes"] is False
    assert "elapsed_seconds" not in persisted
