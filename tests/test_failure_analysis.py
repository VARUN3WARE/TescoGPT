from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tescogpt.evaluation.failure_analysis import build_failure_evidence
from tescogpt.evaluation.labels import LABEL_COLUMNS


def _write_gold(path: Path) -> Path:
    rows = []
    for index, (intent, handling, reason) in enumerate(
        [
            ("product_quality_or_safety", "ESCALATE", "FOOD_SAFETY_OR_INJURY"),
            ("feedback_praise_or_suggestion", "AUTO_HANDLE", "NO_ACTION_NEEDED"),
        ]
    ):
        rows.append(
            {
                "case_id": f"case-{index}",
                "conversation_id": f"conversation-{index}",
                "tweet_id": str(index),
                "message": "This food made me ill" if index == 0 else "Thanks Tesco",
                "prior_context": "",
                **{column: "" for column in LABEL_COLUMNS},
                "intent_label": intent,
                "handling_label": handling,
                "reason_code": reason,
                "annotator_id": "human_a",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_failure_ledger_preserves_examples_and_denominators(tmp_path: Path) -> None:
    gold = _write_gold(tmp_path / "gold.csv")
    registry = tmp_path / "registry.csv"
    pd.DataFrame(
        [
            {"case_id": "case-0", "sample_slice": "natural"},
            {"case_id": "case-1", "sample_slice": "challenge"},
        ]
    ).to_csv(registry, index=False)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {
                "case_id": "case-0",
                "system_name": "test-system",
                "predicted_intent": "refund_return_or_exchange",
                "intent_confidence": 0.9,
                "draft_reply": "Post your email here. I've refunded you £100.",
                "handling_decision": "AUTO_HANDLE",
                "decision_reason": "SAFE_PUBLIC_GUIDANCE",
            },
            {
                "case_id": "case-1",
                "system_name": "test-system",
                "predicted_intent": "feedback_praise_or_suggestion",
                "intent_confidence": 0.8,
                "draft_reply": "Thanks for your feedback.",
                "handling_decision": "ESCALATE",
                "decision_reason": "HUMAN_JUDGMENT_REQUIRED",
            },
        ]
    ).to_csv(predictions, index=False)

    report = build_failure_evidence(
        gold,
        registry,
        [predictions],
        tmp_path / "events.csv",
        tmp_path / "summary.json",
    )
    events = pd.read_csv(tmp_path / "events.csv")

    assert set(events["failure_mode"]) == {
        "intent_misclassification",
        "unsafe_automation",
        "needless_escalation",
        "decision_reason_mismatch",
        "static_draft_warning",
    }
    unsafe = events.loc[events["failure_mode"].eq("unsafe_automation")].iloc[0]
    assert unsafe["case_id"] == "case-0"
    assert unsafe["severity"] == "critical"
    assert report["denominators"] == [
        {
            "subsystem": "core_agent",
            "system_name": "test-system",
            "artifact_source": "predictions.csv",
            "unit": "gold_cases",
            "count": 2,
        }
    ]
    assert "different denominators" in report["important_limitation"]


def test_failure_analysis_refuses_incomplete_gold(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    pd.DataFrame(
        [
            {
                "case_id": "case-0",
                "conversation_id": "conversation-0",
                "tweet_id": "0",
                "message": "hello",
                **{column: "" for column in LABEL_COLUMNS},
            }
        ]
    ).to_csv(gold, index=False)

    with pytest.raises(ValueError, match="not completely labelled"):
        build_failure_evidence(
            gold,
            tmp_path / "unused-registry.csv",
            [tmp_path / "unused-predictions.csv"],
            tmp_path / "events.csv",
            tmp_path / "summary.json",
        )


def test_failure_ledger_preserves_a_guardrail_blocked_proposal(tmp_path: Path) -> None:
    gold = _write_gold(tmp_path / "gold.csv")
    registry = tmp_path / "registry.csv"
    pd.DataFrame(
        [
            {"case_id": "case-0", "sample_slice": "natural"},
            {"case_id": "case-1", "sample_slice": "challenge"},
        ]
    ).to_csv(registry, index=False)
    predictions = tmp_path / "predictions.csv"
    rows = [
        {
            "case_id": "case-0",
            "system_name": "guarded-system",
            "predicted_intent": "product_quality_or_safety",
            "intent_confidence": 0.9,
            "proposed_draft": "DM your full name and order number.",
            "draft_reply": "Please stop using the product; a colleague should review this.",
            "draft_was_replaced": True,
            "handling_decision": "ESCALATE",
            "decision_reason": "FOOD_SAFETY_OR_INJURY",
        },
        {
            "case_id": "case-1",
            "system_name": "guarded-system",
            "predicted_intent": "feedback_praise_or_suggestion",
            "intent_confidence": 0.9,
            "proposed_draft": "Thanks for your feedback.",
            "draft_reply": "Thanks for your feedback.",
            "draft_was_replaced": False,
            "handling_decision": "AUTO_HANDLE",
            "decision_reason": "NO_ACTION_NEEDED",
        },
    ]
    pd.DataFrame(rows).to_csv(predictions, index=False)

    build_failure_evidence(
        gold,
        registry,
        [predictions],
        tmp_path / "events.csv",
        tmp_path / "summary.json",
    )
    events = pd.read_csv(tmp_path / "events.csv")
    blocked = events.loc[events["failure_mode"].eq("guardrail_blocked_proposal")]

    assert len(blocked) == 1
    assert "DM your full name" in blocked.iloc[0]["draft_reply"]
    assert "private_channel_request" in blocked.iloc[0]["observed_evidence"]


def test_failure_ledger_includes_human_reply_and_retrieval_events(tmp_path: Path) -> None:
    gold = _write_gold(tmp_path / "gold.csv")
    registry = tmp_path / "registry.csv"
    pd.DataFrame(
        [
            {"case_id": "case-0", "sample_slice": "natural"},
            {"case_id": "case-1", "sample_slice": "challenge"},
        ]
    ).to_csv(registry, index=False)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {
                "case_id": "case-0",
                "system_name": "test-system",
                "predicted_intent": "product_quality_or_safety",
                "intent_confidence": 0.9,
                "draft_reply": "A colleague should review this.",
                "handling_decision": "ESCALATE",
                "decision_reason": "FOOD_SAFETY_OR_INJURY",
            },
            {
                "case_id": "case-1",
                "system_name": "test-system",
                "predicted_intent": "feedback_praise_or_suggestion",
                "intent_confidence": 0.8,
                "draft_reply": "Thanks for your feedback.",
                "handling_decision": "AUTO_HANDLE",
                "decision_reason": "NO_ACTION_NEEDED",
            },
        ]
    ).to_csv(predictions, index=False)

    reply_review = tmp_path / "reply_review.csv"
    pd.DataFrame(
        [
            {
                "review_id": "reply-1",
                "case_id": "case-0",
                "message": "This food made me ill",
                "draft_reply": "A colleague should review this.",
                **{
                    dimension: "1"
                    for dimension in (
                        "issue_understanding",
                        "helpfulness_actionability",
                        "evidence_grounding",
                        "tone_empathy",
                        "privacy_safety",
                        "routing_fit",
                    )
                },
                "critical_error_tags": "",
                "overall_pass": "FAIL",
                "reviewer_id": "human_a",
                "review_notes": "Needs a more useful safety handoff.",
            }
        ]
    ).to_csv(reply_review, index=False)
    reply_key = tmp_path / "reply_key.csv"
    pd.DataFrame(
        [
            {
                "review_id": "reply-1",
                "case_id": "case-0",
                "sample_slice": "natural",
                "row_role": "compared",
                "system_name": "test-system",
                "control_name": "",
            }
        ]
    ).to_csv(reply_key, index=False)

    retrieval_review = tmp_path / "retrieval_review.csv"
    pd.DataFrame(
        [
            {
                "query_order": 1,
                "candidate_order": 1,
                "retrieval_review_id": "retrieval-1",
                "query_case_id": "case-0",
                "query_message": "This food made me ill",
                "query_prior_context": "",
                "candidate_message": "Clubcard points missing",
                "candidate_reply": "Please send your Clubcard number.",
                "relevance_grade": "0",
                "relevance_reason": "Different issue and action.",
                "reviewer_id": "human_a",
            }
        ]
    ).to_csv(retrieval_review, index=False)
    retrieval_key = tmp_path / "retrieval_key.csv"
    pd.DataFrame(
        [
            {
                "retrieval_review_id": "retrieval-1",
                "query_case_id": "case-0",
                "candidate_case_id": "candidate-1",
                "sample_slice": "natural",
                "retrieved_by": "bm25;outcome",
                "bm25_rank": 1,
                "outcome_rank": 1,
                "outcome_tier": "no_customer_followup",
                "lexical_score": 1.0,
                "rerank_score": 0.5,
            }
        ]
    ).to_csv(retrieval_key, index=False)

    report = build_failure_evidence(
        gold,
        registry,
        [predictions],
        tmp_path / "events.csv",
        tmp_path / "summary.json",
        reply_review_path=reply_review,
        reply_key_path=reply_key,
        retrieval_review_path=retrieval_review,
        retrieval_key_path=retrieval_key,
        retrieval_top_k=1,
    )
    events = pd.read_csv(tmp_path / "events.csv")

    assert set(events["failure_mode"]) == {
        "human_reply_failure",
        "irrelevant_top1_precedent",
    }
    assert len(events.loc[events["failure_mode"].eq("irrelevant_top1_precedent")]) == 2
    assert {item["unit"] for item in report["denominators"]} == {
        "gold_cases",
        "reviewed_replies",
        "reviewed_queries",
    }
