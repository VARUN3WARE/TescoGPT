from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

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


class LinkDrafter:
    name = "link_test"

    def draft(self, case: dict, precedents: list[Precedent]) -> DraftCandidate:
        return DraftCandidate(
            predicted_intent="feedback_praise_or_suggestion",
            intent_confidence=0.99,
            draft_reply="Thanks! See the latest details at https://example.com/help",
            used_evidence_case_ids=("train-1",),
        )


class NoEvidenceDrafter:
    name = "no_evidence_test"

    def draft(self, case: dict, precedents: list[Precedent]) -> DraftCandidate:
        return DraftCandidate(
            predicted_intent="feedback_praise_or_suggestion",
            intent_confidence=0.99,
            draft_reply="Thanks for sharing your feedback with Tesco.",
            used_evidence_case_ids=(),
        )


class UnknownEvidenceDrafter:
    name = "unknown_evidence_test"

    def draft(self, case: dict, precedents: list[Precedent]) -> DraftCandidate:
        return DraftCandidate(
            predicted_intent="feedback_praise_or_suggestion",
            intent_confidence=0.99,
            draft_reply="Thanks for sharing your feedback with Tesco.",
            used_evidence_case_ids=("not-retrieved",),
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
    assert result.decision_reason == "MONEY_OR_COMMITMENT"
    assert result.proposed_draft == (
        "DM your full name and address and I've credited your account."
    )
    assert result.draft_was_replaced
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
    assert result.proposed_draft == result.draft_reply
    assert not result.draft_was_replaced
    assert result.evidence_case_ids == ("train-1",)


def test_prediction_records_only_evidence_the_drafter_declares_used() -> None:
    result = EvidencePolicyAgent(_corpus(), NoEvidenceDrafter()).predict(
        {
            "case_id": "test-1",
            "conversation_id": "99",
            "message": "Thank you to your brilliant cashier",
            "prior_context": "",
        }
    )

    assert result.evidence_case_ids == ()
    assert result.evidence_quotes == ()
    assert result.evidence_scores == ()


def test_agent_rejects_drafter_evidence_outside_retrieved_pack() -> None:
    with pytest.raises(ValueError, match="not retrieved"):
        EvidencePolicyAgent(_corpus(), UnknownEvidenceDrafter()).predict(
            {
                "case_id": "test-1",
                "conversation_id": "99",
                "message": "Thank you to your brilliant cashier",
                "prior_context": "",
            }
        )


def test_draft_candidate_rejects_duplicate_or_blank_evidence_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        DraftCandidate(
            predicted_intent="feedback_praise_or_suggestion",
            intent_confidence=0.99,
            draft_reply="Thanks.",
            used_evidence_case_ids=("train-1", "train-1"),
        )
    with pytest.raises(ValueError, match="non-empty"):
        DraftCandidate(
            predicted_intent="feedback_praise_or_suggestion",
            intent_confidence=0.99,
            draft_reply="Thanks.",
            used_evidence_case_ids=("",),
        )


def test_url_only_guardrail_produces_valid_escalation_and_replacement() -> None:
    result = EvidencePolicyAgent(_corpus(), LinkDrafter()).predict(
        {
            "case_id": "test-1",
            "conversation_id": "99",
            "message": "Thanks to your kind cashier",
            "prior_context": "",
        }
    )

    assert result.handling_decision == "ESCALATE"
    assert result.decision_reason == "CURRENT_POLICY_OR_LIVE_INFO"
    assert result.automation_score == 0
    assert result.draft_was_replaced
    assert "https://" not in result.draft_reply
    assert "draft_guardrail:external_or_legacy_url" in result.safety_flags


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


def test_frozen_reason_priority_survives_multiple_generated_hazards() -> None:
    decision = decide_handling(
        intent="product_quality_or_safety",
        intent_confidence=0.99,
        message="There was glass in this jar and it cut me",
        prior_context="",
        draft_reply=(
            "DM your full name and address. We've refunded you; see "
            "https://example.com/help"
        ),
    )

    assert set(decision.reply_flags) >= {
        "private_channel_request",
        "personal_data_request",
        "unsupported_backend_action",
        "external_or_legacy_url",
    }
    assert decision.handling_decision == "ESCALATE"
    assert decision.reason == "FOOD_SAFETY_OR_INJURY"


@pytest.mark.parametrize(
    ("draft_reply", "expected_reason"),
    [
        ("Please send your order number", "PERSONAL_DATA_OR_PRIVATE_CHANNEL"),
        ("We've processed the refund", "MONEY_OR_COMMITMENT"),
        ("See https://example.com/help", "CURRENT_POLICY_OR_LIVE_INFO"),
        ("x" * 281, "HUMAN_JUDGMENT_REQUIRED"),
    ],
)
def test_every_guardrail_family_maps_to_an_escalation_reason(
    draft_reply: str,
    expected_reason: str,
) -> None:
    decision = decide_handling(
        intent="feedback_praise_or_suggestion",
        intent_confidence=0.99,
        message="Thanks Tesco",
        prior_context="",
        draft_reply=draft_reply,
    )

    assert decision.reply_flags
    assert decision.handling_decision == "ESCALATE"
    assert decision.reason == expected_reason
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
            model="test-model-2026-09-01",
            output_text=json.dumps(output),
            usage={"input_tokens": 10, "output_tokens": 8, "total_tokens": 18},
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
    provenance = drafter.provenance()
    assert provenance["request_count"] == 2
    assert provenance["unique_request_count"] == 1
    assert provenance["api_call_count"] == 1
    assert provenance["cache_hit_count"] == 1
    assert provenance["resolved_models"] == ["test-model-2026-09-01"]
    assert provenance["usage_totals"]["total_tokens"] == 18


def test_openai_drafter_rejects_cache_with_changed_provenance(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    drafter = OpenAIDrafter(
        "test-model",
        cache,
        client=SimpleNamespace(responses=FakeResponses()),
    )
    drafter.draft({"case_id": "c1", "message": "Thanks", "prior_context": ""}, [_precedent()])
    cache_path = next(cache.glob("*.json"))
    record = json.loads(cache_path.read_text(encoding="utf-8"))
    record["model"] = "different-model"
    cache_path.write_text(json.dumps(record), encoding="utf-8")
    fresh = OpenAIDrafter(
        "test-model",
        cache,
        client=SimpleNamespace(responses=FakeResponses()),
    )

    with pytest.raises(ValueError, match="Cached draft provenance"):
        fresh.draft(
            {"case_id": "c1", "message": "Thanks", "prior_context": ""},
            [_precedent()],
        )
