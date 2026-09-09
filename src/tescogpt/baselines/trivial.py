"""Constant baseline: fixed intent, generic reply, and full abstention."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tescogpt.agent.schema import AgentOutput


class TrivialBaseline:
    name = "trivial_constant_v1"

    def predict(self, case: Mapping[str, Any]) -> AgentOutput:
        return AgentOutput(
            case_id=str(case["case_id"]),
            system_name=self.name,
            predicted_intent="other_or_unclear",
            intent_confidence=0.0,
            draft_reply=(
                "Thanks for getting in touch. A colleague will review your message and help."
            ),
            handling_decision="ESCALATE",
            decision_reason="OUT_OF_SCOPE_OR_UNCLEAR",
        )
