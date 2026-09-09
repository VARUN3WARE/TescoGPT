"""Frozen intent, routing, and selective-automation metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from tescogpt.evaluation.labels import (
    AUTO_REASON_CODES,
    ESCALATION_REASON_CODES,
    INTENT_LABELS,
    validate_annotations,
)
from tescogpt.policy.safety import inspect_reply


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def wilson_upper(successes: int, trials: int, confidence: float = 0.95) -> float | None:
    """One-sided Wilson upper confidence bound for a binomial rate."""
    if trials == 0:
        return None
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    z = NormalDist().inv_cdf(confidence)
    rate = successes / trials
    denominator = 1 + z**2 / trials
    center = rate + z**2 / (2 * trials)
    radius = z * math.sqrt(rate * (1 - rate) / trials + z**2 / (4 * trials**2))
    return (center + radius) / denominator


def intent_metrics(gold: pd.Series, predicted: pd.Series) -> dict[str, Any]:
    """Return accuracy, macro-F1, and transparent per-intent counts."""
    per_intent: dict[str, Any] = {}
    f1_values: list[float] = []
    for label in INTENT_LABELS:
        true_positive = int(((gold == label) & (predicted == label)).sum())
        false_positive = int(((gold != label) & (predicted == label)).sum())
        false_negative = int(((gold == label) & (predicted != label)).sum())
        support = int((gold == label).sum())
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        if precision is None or recall is None or precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        if support:
            f1_values.append(f1)
        per_intent[label] = {
            "support": support,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "accuracy": float((gold == predicted).mean()),
        "macro_f1_observed_intents": float(np.mean(f1_values)) if f1_values else 0.0,
        "macro_f1_all_taxonomy": float(
            np.mean([metrics["f1"] for metrics in per_intent.values()])
        ),
        "observed_intent_count": len(f1_values),
        "per_intent": per_intent,
    }


def bootstrap_macro_f1(
    gold: pd.Series,
    predicted: pd.Series,
    *,
    iterations: int = 2000,
    seed: int = 20260911,
) -> dict[str, float]:
    """Percentile interval over examples; no tuning decisions use this result."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if len(gold) != len(predicted) or len(gold) == 0:
        raise ValueError("Bootstrap inputs must be equally sized and non-empty")
    gold_values = gold.reset_index(drop=True)
    predicted_values = predicted.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sample = rng.integers(0, len(gold_values), size=len(gold_values))
        estimates[index] = intent_metrics(
            gold_values.iloc[sample].reset_index(drop=True),
            predicted_values.iloc[sample].reset_index(drop=True),
        )["macro_f1_observed_intents"]
    return {
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "iterations": iterations,
        "seed": seed,
    }


def routing_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    predicted_auto = frame["handling_decision"].eq("AUTO_HANDLE")
    gold_auto = frame["handling_label"].eq("AUTO_HANDLE")
    unsafe_auto = predicted_auto & ~gold_auto
    needless_escalation = ~predicted_auto & gold_auto
    auto_count = int(predicted_auto.sum())
    gold_auto_count = int(gold_auto.sum())
    unsafe_count = int(unsafe_auto.sum())
    needless_count = int(needless_escalation.sum())
    return {
        "coverage": float(predicted_auto.mean()),
        "auto_handled_count": auto_count,
        "unsafe_auto_count": unsafe_count,
        "unsafe_auto_rate_given_auto": _safe_divide(unsafe_count, auto_count),
        "unsafe_auto_rate_wilson_upper_95": wilson_upper(unsafe_count, auto_count),
        "gold_auto_handle_count": gold_auto_count,
        "needless_escalation_count": needless_count,
        "needless_escalation_rate_given_gold_auto": _safe_divide(
            needless_count, gold_auto_count
        ),
        "handling_accuracy": float((predicted_auto == gold_auto).mean()),
    }


def decision_reason_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Measure the stated operational reason without hiding route errors."""
    correct = frame["reason_code"].eq(frame["decision_reason"])
    handling_correct = frame["handling_label"].eq(frame["handling_decision"])
    confusions = (
        frame.loc[~correct]
        .groupby(["reason_code", "decision_reason"], sort=True)
        .size()
        .reset_index(name="count")
        .sort_values(
            ["count", "reason_code", "decision_reason"],
            ascending=[False, True, True],
            kind="stable",
        )
    )
    return {
        "exact_accuracy": float(correct.mean()),
        "correct_count": int(correct.sum()),
        "accuracy_given_correct_handling": _safe_divide(
            int((correct & handling_correct).sum()),
            int(handling_correct.sum()),
        ),
        "correct_handling_count": int(handling_correct.sum()),
        "confusions": confusions.to_dict(orient="records"),
    }


def risk_coverage_curve(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rank by automation score and show risk if the top share were automated."""
    ordered = frame.sort_values(
        ["automation_score", "case_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    rows: list[dict[str, Any]] = [
        {
            "target_coverage": 0.0,
            "selected_count": 0,
            "realized_coverage": 0.0,
            "minimum_score": None,
            "unsafe_count": 0,
            "unsafe_rate": None,
            "unsafe_rate_wilson_upper_95": None,
        }
    ]
    for target in np.linspace(0.05, 1.0, 20):
        count = min(len(ordered), max(1, math.ceil(float(target) * len(ordered))))
        selected = ordered.iloc[:count]
        unsafe_count = int(selected["handling_label"].ne("AUTO_HANDLE").sum())
        rows.append(
            {
                "target_coverage": round(float(target), 2),
                "selected_count": count,
                "realized_coverage": count / len(ordered),
                "minimum_score": float(selected["automation_score"].iloc[-1]),
                "unsafe_count": unsafe_count,
                "unsafe_rate": unsafe_count / count,
                "unsafe_rate_wilson_upper_95": wilson_upper(unsafe_count, count),
            }
        )
    return rows


def _evaluate_slice(
    frame: pd.DataFrame,
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    intents = intent_metrics(frame["intent_label"], frame["predicted_intent"])
    intents["bootstrap_95"] = bootstrap_macro_f1(
        frame["intent_label"],
        frame["predicted_intent"],
        iterations=bootstrap_iterations,
        seed=seed,
    )
    static_flags = frame["draft_reply"].map(inspect_reply)
    return {
        "example_count": len(frame),
        "intent": intents,
        "routing": routing_metrics(frame),
        "decision_reason": decision_reason_metrics(frame),
        "static_reply_warning_count": int(static_flags.map(bool).sum()),
        "risk_coverage_curve": risk_coverage_curve(frame),
    }


def evaluate_predictions(
    gold_path: str | Path,
    registry_path: str | Path,
    prediction_paths: list[str | Path],
    output_path: str | Path,
    *,
    bootstrap_iterations: int = 2000,
    seed: int = 20260911,
) -> dict[str, Any]:
    """Evaluate all systems against one complete human-labelled sheet."""
    if not prediction_paths:
        raise ValueError("At least one prediction file is required")
    validate_annotations(gold_path, require_complete=True, require_blind=True)
    gold = pd.read_csv(gold_path, dtype="string", keep_default_na=False)
    registry = pd.read_csv(registry_path, dtype="string", keep_default_na=False)
    slice_map = registry.set_index("case_id")["sample_slice"]
    gold["sample_slice"] = gold["case_id"].map(slice_map)
    if gold["sample_slice"].isna().any():
        raise ValueError("Every gold case must exist in the sampling registry")

    results: dict[str, Any] = {}
    for prediction_path in prediction_paths:
        path = Path(prediction_path)
        predictions = pd.read_csv(path, dtype="string", keep_default_na=False)
        required = {
            "case_id",
            "system_name",
            "predicted_intent",
            "draft_reply",
            "handling_decision",
            "decision_reason",
            "automation_score",
        }
        missing = sorted(required - set(predictions.columns))
        if missing:
            raise ValueError(f"Predictions are missing columns: {', '.join(missing)}")
        if predictions["case_id"].duplicated().any():
            raise ValueError(f"Duplicate prediction case IDs in {path}")
        if set(predictions["case_id"]) != set(gold["case_id"]):
            raise ValueError(f"Prediction case IDs do not exactly match gold: {path}")
        predictions["automation_score"] = pd.to_numeric(
            predictions["automation_score"], errors="raise"
        )
        invalid_intents = sorted(
            set(predictions["predicted_intent"]) - set(INTENT_LABELS)
        )
        if invalid_intents:
            raise ValueError("Unknown predicted intents: " + ", ".join(invalid_intents))
        invalid_handling = sorted(
            set(predictions["handling_decision"]) - {"AUTO_HANDLE", "ESCALATE"}
        )
        if invalid_handling:
            raise ValueError("Unknown handling decisions: " + ", ".join(invalid_handling))
        allowed_reasons = set(AUTO_REASON_CODES) | set(ESCALATION_REASON_CODES)
        invalid_reasons = sorted(set(predictions["decision_reason"]) - allowed_reasons)
        if invalid_reasons:
            raise ValueError("Unknown decision reasons: " + ", ".join(invalid_reasons))
        incompatible_reasons = (
            predictions["handling_decision"].eq("AUTO_HANDLE")
            & ~predictions["decision_reason"].isin(AUTO_REASON_CODES)
        ) | (
            predictions["handling_decision"].eq("ESCALATE")
            & ~predictions["decision_reason"].isin(ESCALATION_REASON_CODES)
        )
        if incompatible_reasons.any():
            raise ValueError("Decision reasons must be compatible with handling decisions")
        if not predictions["automation_score"].between(0, 1).all():
            raise ValueError("Automation scores must be between zero and one")
        if predictions["draft_reply"].str.strip().eq("").any():
            raise ValueError("Draft replies must not be blank")
        merged = gold.merge(predictions, on="case_id", validate="one_to_one")
        system_names = predictions["system_name"].unique()
        if len(system_names) != 1:
            raise ValueError(f"Prediction file must contain one system name: {path}")
        system_name = str(system_names[0])
        slices = {"all": merged}
        slices.update(
            {
                str(name): group.copy()
                for name, group in merged.groupby("sample_slice", sort=True)
            }
        )
        results[system_name] = {
            "prediction_file": path.name,
            "slices": {
                name: _evaluate_slice(
                    frame,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed + offset,
                )
                for offset, (name, frame) in enumerate(slices.items())
            },
        }

    report = {
        "evaluation_schema_version": 1,
        "gold_file": Path(gold_path).name,
        "registry_file": Path(registry_path).name,
        "system_count": len(results),
        "systems": results,
        "important_definition": (
            "unsafe_auto means predicted AUTO_HANDLE where a human labelled ESCALATE"
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
