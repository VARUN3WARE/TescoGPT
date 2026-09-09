"""Pre-registered offline trust gate for the headline support agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_THRESHOLDS = (
    "intent_reference_system",
    "minimum_natural_coverage",
    "maximum_natural_unsafe_auto_wilson_upper_95",
    "maximum_overall_unsafe_auto_count",
    "maximum_overall_static_reply_warning_count",
    "minimum_overall_intent_macro_f1_difference",
    "minimum_overall_human_reply_pass_rate",
    "maximum_overall_human_critical_error_rate",
    "minimum_outcome_minus_bm25_ndcg_difference",
)


def _unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Offline trust threshold {name} must be numeric")
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError(f"Offline trust threshold {name} must be between zero and one")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Offline trust threshold {name} must be a non-negative integer")
    return value


def _intent_difference(
    metrics: dict[str, Any],
    headline_system: str,
    reference_system: str,
) -> float:
    for comparison in metrics["paired_intent_comparisons"]:
        systems = {comparison["system_a"], comparison["system_b"]}
        if systems != {headline_system, reference_system}:
            continue
        difference = float(
            comparison["slices"]["all"]["macro_f1_difference"]
        )
        return difference if comparison["system_a"] == headline_system else -difference
    raise ValueError("Offline trust gate cannot find the headline/reference intent comparison")


def evaluate_offline_trust_gate(
    *,
    metrics: dict[str, Any],
    retrieval_metrics: dict[str, Any],
    reply_quality: dict[str, Any],
    headline_system: str,
    thresholds: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Evaluate frozen go/no-go criteria without blocking result reproduction."""
    missing = sorted(set(REQUIRED_THRESHOLDS) - set(thresholds))
    unexpected = sorted(set(thresholds) - set(REQUIRED_THRESHOLDS))
    if missing:
        raise ValueError("Offline trust gate is missing thresholds: " + ", ".join(missing))
    if unexpected:
        raise ValueError(
            "Offline trust gate has unexpected thresholds: " + ", ".join(unexpected)
        )
    if headline_system not in metrics["systems"]:
        raise ValueError("Offline trust headline system is absent from core metrics")
    if headline_system not in reply_quality["systems"]:
        raise ValueError("Offline trust headline system is absent from human reply ratings")

    reference_system = str(thresholds["intent_reference_system"]).strip()
    if not reference_system or reference_system == headline_system:
        raise ValueError("Offline trust intent reference must be a different named system")
    if reference_system not in metrics["systems"]:
        raise ValueError("Offline trust intent reference system is absent from core metrics")

    minimum_coverage = _unit_interval(
        thresholds["minimum_natural_coverage"], "minimum_natural_coverage"
    )
    maximum_risk_bound = _unit_interval(
        thresholds["maximum_natural_unsafe_auto_wilson_upper_95"],
        "maximum_natural_unsafe_auto_wilson_upper_95",
    )
    maximum_unsafe_count = _nonnegative_integer(
        thresholds["maximum_overall_unsafe_auto_count"],
        "maximum_overall_unsafe_auto_count",
    )
    maximum_warning_count = _nonnegative_integer(
        thresholds["maximum_overall_static_reply_warning_count"],
        "maximum_overall_static_reply_warning_count",
    )
    minimum_intent_difference = float(
        thresholds["minimum_overall_intent_macro_f1_difference"]
    )
    if not -1 <= minimum_intent_difference <= 1:
        raise ValueError(
            "Offline trust threshold minimum_overall_intent_macro_f1_difference "
            "must be between minus one and one"
        )
    minimum_reply_pass = _unit_interval(
        thresholds["minimum_overall_human_reply_pass_rate"],
        "minimum_overall_human_reply_pass_rate",
    )
    maximum_critical_error = _unit_interval(
        thresholds["maximum_overall_human_critical_error_rate"],
        "maximum_overall_human_critical_error_rate",
    )
    minimum_ndcg_difference = float(
        thresholds["minimum_outcome_minus_bm25_ndcg_difference"]
    )
    if not -1 <= minimum_ndcg_difference <= 1:
        raise ValueError(
            "Offline trust threshold minimum_outcome_minus_bm25_ndcg_difference "
            "must be between minus one and one"
        )

    headline = metrics["systems"][headline_system]["slices"]
    natural_routing = headline["natural"]["routing"]
    overall = headline["all"]
    human_overall = reply_quality["systems"][headline_system]["slices"]["all"]
    risk_bound = natural_routing["unsafe_auto_rate_wilson_upper_95"]
    intent_difference = _intent_difference(metrics, headline_system, reference_system)
    ndcg_difference = float(
        retrieval_metrics["paired_comparison"][
            "mean_ndcg_difference_outcome_minus_bm25"
        ]
    )

    checks = [
        {
            "name": "natural_automation_coverage",
            "actual": float(natural_routing["coverage"]),
            "criterion": f">= {minimum_coverage}",
            "passed": float(natural_routing["coverage"]) >= minimum_coverage,
        },
        {
            "name": "natural_unsafe_auto_wilson_upper_95",
            "actual": None if risk_bound is None else float(risk_bound),
            "criterion": f"<= {maximum_risk_bound}",
            "passed": risk_bound is not None and float(risk_bound) <= maximum_risk_bound,
        },
        {
            "name": "overall_observed_unsafe_auto_count",
            "actual": int(overall["routing"]["unsafe_auto_count"]),
            "criterion": f"<= {maximum_unsafe_count}",
            "passed": int(overall["routing"]["unsafe_auto_count"])
            <= maximum_unsafe_count,
        },
        {
            "name": "overall_static_reply_warning_count",
            "actual": int(overall["static_reply_warning_count"]),
            "criterion": f"<= {maximum_warning_count}",
            "passed": int(overall["static_reply_warning_count"])
            <= maximum_warning_count,
        },
        {
            "name": "overall_intent_macro_f1_difference_vs_reference",
            "actual": intent_difference,
            "criterion": f">= {minimum_intent_difference}",
            "passed": intent_difference >= minimum_intent_difference,
        },
        {
            "name": "overall_human_reply_pass_rate",
            "actual": float(human_overall["overall_pass_rate"]),
            "criterion": f">= {minimum_reply_pass}",
            "passed": float(human_overall["overall_pass_rate"])
            >= minimum_reply_pass,
        },
        {
            "name": "overall_human_critical_error_rate",
            "actual": float(human_overall["critical_error_rate"]),
            "criterion": f"<= {maximum_critical_error}",
            "passed": float(human_overall["critical_error_rate"])
            <= maximum_critical_error,
        },
        {
            "name": "outcome_minus_bm25_mean_ndcg_difference",
            "actual": ndcg_difference,
            "criterion": f">= {minimum_ndcg_difference}",
            "passed": ndcg_difference >= minimum_ndcg_difference,
        },
    ]
    failed = [check["name"] for check in checks if not check["passed"]]
    report = {
        "offline_trust_gate_schema_version": 1,
        "headline_system": headline_system,
        "intent_reference_system": reference_system,
        "thresholds": thresholds,
        "passed": not failed,
        "recommendation": (
            "eligible_for_shadow_mode_trial" if not failed else "not_ready_for_shadow_mode"
        ),
        "checks": checks,
        "failed_checks": failed,
        "scope_limit": (
            "Passing supports a monitored shadow-mode trial with human review; it does "
            "not authorize unattended production sending."
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
