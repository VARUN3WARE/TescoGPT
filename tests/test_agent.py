from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from tescogpt.agent.drafting import (
    DraftCandidate,
    OpenAIDrafter,
    SafeTemplateDrafter,
    safe_template,
)
from tescogpt.agent.main import EvidencePolicyAgent
from tescogpt.evaluation.labels import INTENT_LABELS
from tescogpt.policy.routing import decide_handling
from tescogpt.policy.safety import inspect_reply
from tescogpt.retrieval.outcome import Precedent


def _corpus() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "train-1",
                "conversation_id": "10",
                "message": "thank you to the kind cashier",
                "historical_reply": "Thanks for sharing this with us.",
                "customer_followup": "",
                "private_handoff_proxy": False,
                "positive_outcome_proxy": False,
                "unresolved_outcome_proxy": False,
            }
        ]
    )


def _precedent() -> Precedent:
    return Precedent(
        case_id="train-1",
        conversation_id="10",
        lexical_score=1.0,
        rerank_score=0.8,
        outcome_tier="positive_followup_proxy",
        safety_penalty_flags=(),
        message="thank you to the cashier",
        historical_reply="Thanks for sharing this with us.",
        customer_followup="Thanks",
    )


class UnsafeDrafter:
    name = "unsafe_test"

    def draft(self, case: dict, precedents: list[Precedent]) -> DraftCandidate:
        return DraftCandidate(
            predicted_intent="feedback_praise_or_suggestion",
            intent_confidence=0.99,
            draft_reply="DM your full name and address and I've credited your account.",
            used_evidence_case_ids=("train-1",),
        )


def test_final_gate_replaces_unsafe_draft_and_escalates() -> None:
    result = EvidencePolicyAgent(_corpus(), UnsafeDrafter()).predict(
        {
            "case_id": "test-1",
            "conversation_id": "99",
            "message": "Thanks to your kind cashier",
            "prior_context": "",
        }
    )

    assert result.handling_decision == "ESCALATE"
    assert result.decision_reason == "PERSONAL_DATA_OR_PRIVATE_CHANNEL"
    assert "DM" not in result.draft_reply
    assert "draft_guardrail:personal_data_request" in result.safety_flags


def test_template_agent_auto_handles_plain_feedback() -> None:
    result = EvidencePolicyAgent(_corpus(), SafeTemplateDrafter()).predict(
        {
            "case_id": "test-1",
            "conversation_id": "99",
            "message": "Thank you to your brilliant cashier",
            "prior_context": "",
        }
    )

    assert result.handling_decision == "AUTO_HANDLE"
    assert result.decision_reason == "NO_ACTION_NEEDED"
    assert result.evidence_case_ids == ("train-1",)


def test_safety_risk_overrides_positive_wording() -> None:
    decision = decide_handling(
        intent="product_quality_or_safety",
        intent_confidence=0.99,
        message="Thanks, but the glass in this jar cut me",
        prior_context="",
        draft_reply=safe_template("product_quality_or_safety"),
    )

    assert decision.handling_decision == "ESCALATE"
    assert decision.reason == "FOOD_SAFETY_OR_INJURY"
    assert decision.automation_score == 0


def test_every_fallback_template_fits_public_safety_contract() -> None:
    for intent in INTENT_LABELS:
        reply = safe_template(intent)
        assert len(reply) <= 280
        assert inspect_reply(reply) == ()


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        output = {
            "predicted_intent": "feedback_praise_or_suggestion",
            "intent_confidence": 0.91,
            "draft_reply": "Thanks for sharing your feedback with Tesco.",
            "used_evidence_case_ids": ["train-1"],
        }
        return SimpleNamespace(
            id="response-test",
            output_text=json.dumps(output),
            usage={"input_tokens": 10, "output_tokens": 8},
        )


def test_openai_drafter_uses_strict_schema_store_false_and_cache(tmp_path: Path) -> None:
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    drafter = OpenAIDrafter("test-model", tmp_path / "cache", client=client)
    case = {"message": "Thanks Tesco", "prior_context": ""}

    first = drafter.draft(case, [_precedent()])
    second = drafter.draft(case, [_precedent()])

    assert first == second
    assert len(responses.calls) == 1
    request = responses.calls[0]
    assert request["store"] is False
    assert request["text"]["format"]["type"] == "json_schema"
    assert list((tmp_path / "cache").glob("*.json"))
