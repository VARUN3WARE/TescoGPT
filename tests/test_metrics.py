from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tescogpt.evaluation.labels import LABEL_COLUMNS
from tescogpt.evaluation.metrics import (
    evaluate_predictions,
    intent_metrics,
    risk_coverage_curve,
    routing_metrics,
    wilson_upper,
)
from tescogpt.evaluation.safety import audit_prediction_safety


def test_intent_metrics_include_errors_and_observed_macro_f1() -> None:
    gold = pd.Series(["delivery_or_collection", "delivery_or_collection", "other_or_unclear"])
    predicted = pd.Series(
        ["delivery_or_collection", "other_or_unclear", "other_or_unclear"]
    )
    result = intent_metrics(gold, predicted)

    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["observed_intent_count"] == 2
    assert result["macro_f1_observed_intents"] == pytest.approx(2 / 3)
    assert result["macro_f1_all_taxonomy"] == pytest.approx((2 / 3 + 2 / 3) / 10)
    assert result["per_intent"]["delivery_or_collection"]["false_negative"] == 1


def test_routing_reports_coverage_and_both_error_directions() -> None:
    frame = pd.DataFrame(
        {
            "handling_label": ["ESCALATE", "AUTO_HANDLE", "AUTO_HANDLE", "ESCALATE"],
            "handling_decision": ["AUTO_HANDLE", "ESCALATE", "AUTO_HANDLE", "ESCALATE"],
        }
    )
    result = routing_metrics(frame)

    assert result["coverage"] == 0.5
    assert result["unsafe_auto_count"] == 1
    assert result["needless_escalation_count"] == 1
    assert result["handling_accuracy"] == 0.5
    assert wilson_upper(0, 10) > 0
    assert wilson_upper(0, 0) is None


def test_risk_coverage_uses_score_ranking() -> None:
    frame = pd.DataFrame(
        {
            "case_id": ["a", "b", "c", "d"],
            "automation_score": [0.9, 0.8, 0.2, 0.1],
            "handling_label": ["AUTO_HANDLE", "AUTO_HANDLE", "ESCALATE", "ESCALATE"],
        }
    )
    curve = risk_coverage_curve(frame)

    assert curve[0]["selected_count"] == 0
    assert curve[1]["selected_count"] == 1
    assert curve[1]["unsafe_rate"] == 0
    assert curve[-1]["unsafe_rate"] == 0.5


def test_evaluation_refuses_incomplete_human_labels(tmp_path: Path) -> None:
    row = {
        "case_id": "case-1",
        "conversation_id": "1",
        "tweet_id": "1",
        "message": "hello",
        **{column: "" for column in LABEL_COLUMNS},
    }
    gold_path = tmp_path / "gold.csv"
    pd.DataFrame([row]).to_csv(gold_path, index=False)
    registry_path = tmp_path / "registry.csv"
    pd.DataFrame([{"case_id": "case-1", "sample_slice": "natural"}]).to_csv(
        registry_path, index=False
    )

    with pytest.raises(ValueError, match="not completely labelled"):
        evaluate_predictions(
            gold_path,
            registry_path,
            [tmp_path / "unused.csv"],
            tmp_path / "metrics.json",
            bootstrap_iterations=5,
        )


def test_complete_evaluation_and_static_audit_write_artifacts(tmp_path: Path) -> None:
    rows = []
    for index, (intent, handling, reason) in enumerate(
        [
            ("delivery_or_collection", "ESCALATE", "ACCOUNT_OR_ORDER_LOOKUP"),
            ("feedback_praise_or_suggestion", "AUTO_HANDLE", "NO_ACTION_NEEDED"),
        ]
    ):
        rows.append(
            {
                "case_id": f"case-{index}",
                "conversation_id": str(index),
                "tweet_id": str(index),
                "message": "message",
                **{column: "" for column in LABEL_COLUMNS},
                "intent_label": intent,
                "handling_label": handling,
                "reason_code": reason,
                "annotator_id": "human_a",
            }
        )
    gold_path = tmp_path / "gold.csv"
    pd.DataFrame(rows).to_csv(gold_path, index=False)
    registry_path = tmp_path / "registry.csv"
    pd.DataFrame(
        [
            {"case_id": "case-0", "sample_slice": "natural"},
            {"case_id": "case-1", "sample_slice": "challenge"},
        ]
    ).to_csv(registry_path, index=False)
    prediction_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {
                "case_id": "case-0",
                "system_name": "test-system",
                "predicted_intent": "delivery_or_collection",
                "draft_reply": "DM your order number.",
                "handling_decision": "ESCALATE",
                "automation_score": 0.1,
            },
            {
                "case_id": "case-1",
                "system_name": "test-system",
                "predicted_intent": "feedback_praise_or_suggestion",
                "draft_reply": "Thanks for your feedback.",
                "handling_decision": "AUTO_HANDLE",
                "automation_score": 0.9,
            },
        ]
    ).to_csv(prediction_path, index=False)

    metrics = evaluate_predictions(
        gold_path,
        registry_path,
        [prediction_path],
        tmp_path / "metrics.json",
        bootstrap_iterations=5,
    )
    safety = audit_prediction_safety(
        [prediction_path],
        tmp_path / "safety.json",
        tmp_path / "details",
    )

    assert metrics["systems"]["test-system"]["slices"]["all"]["intent"]["accuracy"] == 1
    assert safety["systems"]["test-system"]["flagged_draft_count"] == 1
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "details" / "predictions_static_safety.csv").is_file()
