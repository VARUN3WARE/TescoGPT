from pathlib import Path

import pandas as pd
import pytest

from tescogpt.evaluation.reproduce import _validate_generation_provenance


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
