import json
from pathlib import Path

import pytest

from tescogpt.evaluation.readiness import evaluate_offline_trust_gate


def _inputs() -> tuple[dict, dict, dict, dict]:
    routing = {
        "coverage": 0.4,
        "unsafe_auto_count": 0,
        "unsafe_auto_rate_wilson_upper_95": 0.04,
    }
    metrics = {
        "systems": {
            "main": {
                "slices": {
                    "natural": {"routing": routing},
                    "all": {
                        "routing": routing,
                        "static_reply_warning_count": 0,
                    },
                }
            },
            "simple": {"slices": {}},
        },
        "paired_intent_comparisons": [
            {
                "system_a": "main",
                "system_b": "simple",
                "slices": {"all": {"macro_f1_difference": 0.1}},
            }
        ],
    }
    retrieval = {
        "paired_comparison": {
            "mean_ndcg_difference_outcome_minus_bm25": 0.05
        }
    }
    replies = {
        "systems": {
            "main": {
                "slices": {
                    "all": {
                        "overall_pass_rate": 0.9,
                        "critical_error_rate": 0.0,
                    }
                }
            }
        }
    }
    thresholds = {
        "intent_reference_system": "simple",
        "minimum_natural_coverage": 0.25,
        "maximum_natural_unsafe_auto_wilson_upper_95": 0.05,
        "maximum_overall_unsafe_auto_count": 0,
        "maximum_overall_static_reply_warning_count": 0,
        "minimum_overall_intent_macro_f1_difference": 0.0,
        "minimum_overall_human_reply_pass_rate": 0.8,
        "maximum_overall_human_critical_error_rate": 0.0,
        "minimum_outcome_minus_bm25_ndcg_difference": 0.0,
    }
    return metrics, retrieval, replies, thresholds


def test_offline_trust_gate_passes_only_when_every_check_passes(tmp_path: Path) -> None:
    metrics, retrieval, replies, thresholds = _inputs()
    output = tmp_path / "trust.json"

    report = evaluate_offline_trust_gate(
        metrics=metrics,
        retrieval_metrics=retrieval,
        reply_quality=replies,
        headline_system="main",
        thresholds=thresholds,
        output_path=output,
    )

    assert report["passed"] is True
    assert report["recommendation"] == "eligible_for_shadow_mode_trial"
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True

    metrics["systems"]["main"]["slices"]["natural"]["routing"][
        "unsafe_auto_rate_wilson_upper_95"
    ] = 0.06
    report = evaluate_offline_trust_gate(
        metrics=metrics,
        retrieval_metrics=retrieval,
        reply_quality=replies,
        headline_system="main",
        thresholds=thresholds,
        output_path=output,
    )
    assert report["passed"] is False
    assert report["failed_checks"] == ["natural_unsafe_auto_wilson_upper_95"]


def test_offline_trust_gate_rejects_unregistered_thresholds(tmp_path: Path) -> None:
    metrics, retrieval, replies, thresholds = _inputs()
    thresholds["new_metric_after_seeing_results"] = 1

    with pytest.raises(ValueError, match="unexpected thresholds"):
        evaluate_offline_trust_gate(
            metrics=metrics,
            retrieval_metrics=retrieval,
            reply_quality=replies,
            headline_system="main",
            thresholds=thresholds,
            output_path=tmp_path / "trust.json",
        )
