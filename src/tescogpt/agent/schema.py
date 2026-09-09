"""Validated output contract shared by every compared support system."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tescogpt.evaluation.labels import (
    AUTO_REASON_CODES,
    ESCALATION_REASON_CODES,
    INTENT_LABELS,
)


@dataclass(frozen=True)
class AgentOutput:
    """One auditable prediction for one incoming customer case."""

    case_id: str
    system_name: str
    predicted_intent: str
    intent_confidence: float
    draft_reply: str
    handling_decision: str
    decision_reason: str
    evidence_case_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence_quotes: tuple[str, ...] = field(default_factory=tuple)
    evidence_scores: tuple[float, ...] = field(default_factory=tuple)
    safety_flags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be blank")
        if not self.system_name.strip():
            raise ValueError("system_name must not be blank")
        if self.predicted_intent not in INTENT_LABELS:
            raise ValueError(f"Unknown predicted intent: {self.predicted_intent}")
        if not 0 <= self.intent_confidence <= 1:
            raise ValueError("intent_confidence must be between 0 and 1")
        if not self.draft_reply.strip():
            raise ValueError("draft_reply must not be blank")
        if self.handling_decision not in {"AUTO_HANDLE", "ESCALATE"}:
            raise ValueError(f"Unknown handling decision: {self.handling_decision}")
        allowed_reasons = (
            AUTO_REASON_CODES
            if self.handling_decision == "AUTO_HANDLE"
            else ESCALATION_REASON_CODES
        )
        if self.decision_reason not in allowed_reasons:
            raise ValueError(
                f"{self.handling_decision} is incompatible with {self.decision_reason}"
            )
        if len(self.evidence_case_ids) != len(self.evidence_quotes):
            raise ValueError("Every evidence case ID must have one evidence quote")
        if len(self.evidence_case_ids) != len(self.evidence_scores):
            raise ValueError("Every evidence case ID must have one retrieval score")

    def to_record(self) -> dict[str, Any]:
        """Return a stable, CSV-friendly representation."""
        return {
            "case_id": self.case_id,
            "system_name": self.system_name,
            "predicted_intent": self.predicted_intent,
            "intent_confidence": round(float(self.intent_confidence), 6),
            "draft_reply": self.draft_reply,
            "handling_decision": self.handling_decision,
            "decision_reason": self.decision_reason,
            "evidence_case_ids": json.dumps(self.evidence_case_ids),
            "evidence_quotes": json.dumps(self.evidence_quotes, ensure_ascii=False),
            "evidence_scores": json.dumps(
                tuple(round(float(score), 6) for score in self.evidence_scores)
            ),
            "safety_flags": json.dumps(self.safety_flags),
        }


PREDICTION_COLUMNS = tuple(AgentOutput.__dataclass_fields__)
