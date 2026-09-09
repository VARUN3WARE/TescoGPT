"""Validated output contract shared by every compared support system."""

from __future__ import annotations

import json
import math
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
    automation_score: float = 0.0
    evidence_case_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence_quotes: tuple[str, ...] = field(default_factory=tuple)
    evidence_scores: tuple[float, ...] = field(default_factory=tuple)
    safety_flags: tuple[str, ...] = field(default_factory=tuple)
    proposed_draft: str | None = None
    draft_was_replaced: bool = False

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
        if self.proposed_draft is not None and not self.proposed_draft.strip():
            raise ValueError("proposed_draft must not be blank when supplied")
        if not isinstance(self.draft_was_replaced, bool):
            raise ValueError("draft_was_replaced must be boolean")
        proposed = self.proposed_draft or self.draft_reply
        if self.draft_was_replaced and proposed == self.draft_reply:
            raise ValueError("A replaced draft must differ from the public draft")
        if not self.draft_was_replaced and proposed != self.draft_reply:
            raise ValueError("Different proposed and public drafts require replacement provenance")
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
        if not 0 <= self.automation_score <= 1:
            raise ValueError("automation_score must be between 0 and 1")
        if len(self.evidence_case_ids) != len(self.evidence_quotes):
            raise ValueError("Every evidence case ID must have one evidence quote")
        if len(self.evidence_case_ids) != len(self.evidence_scores):
            raise ValueError("Every evidence case ID must have one retrieval score")
        if any(
            not isinstance(case_id, str) or not case_id.strip()
            for case_id in self.evidence_case_ids
        ):
            raise ValueError("Evidence case IDs must be non-empty strings")
        if len(self.evidence_case_ids) != len(set(self.evidence_case_ids)):
            raise ValueError("Evidence case IDs must be unique")
        if any(not isinstance(quote, str) or not quote.strip() for quote in self.evidence_quotes):
            raise ValueError("Evidence quotes must be non-empty strings")
        if any(not math.isfinite(float(score)) for score in self.evidence_scores):
            raise ValueError("Evidence scores must be finite")

    def to_record(self) -> dict[str, Any]:
        """Return a stable, CSV-friendly representation."""
        return {
            "case_id": self.case_id,
            "system_name": self.system_name,
            "predicted_intent": self.predicted_intent,
            "intent_confidence": round(float(self.intent_confidence), 6),
            "proposed_draft": self.proposed_draft or self.draft_reply,
            "draft_reply": self.draft_reply,
            "draft_was_replaced": self.draft_was_replaced,
            "handling_decision": self.handling_decision,
            "decision_reason": self.decision_reason,
            "automation_score": round(float(self.automation_score), 6),
            "evidence_case_ids": json.dumps(self.evidence_case_ids),
            "evidence_quotes": json.dumps(self.evidence_quotes, ensure_ascii=False),
            "evidence_scores": json.dumps(
                tuple(round(float(score), 6) for score in self.evidence_scores)
            ),
            "safety_flags": json.dumps(self.safety_flags),
        }


PREDICTION_COLUMNS = tuple(AgentOutput.__dataclass_fields__)
