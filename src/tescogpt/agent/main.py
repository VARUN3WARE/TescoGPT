"""Evidence-aware support agent with a deterministic final policy gate."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tescogpt.agent.drafting import Drafter, safe_template
from tescogpt.agent.schema import AgentOutput
from tescogpt.policy.routing import decide_handling
from tescogpt.retrieval.outcome import OutcomeAwareRetriever


class EvidencePolicyAgent:
    """Retrieve evidence, draft, then enforce non-bypassable public safeguards."""

    def __init__(self, corpus: pd.DataFrame, drafter: Drafter) -> None:
        self._retriever = OutcomeAwareRetriever(corpus)
        self._drafter = drafter
        self.name = f"tescogpt_{drafter.name}_policy_v1"

    def predict(self, case: Mapping[str, Any]) -> AgentOutput:
        message = str(case.get("message", ""))
        prior_context = str(case.get("prior_context", ""))
        query = f"{prior_context} {message}".strip()
        precedents = self._retriever.search(
            query,
            top_k=3,
            exclude_case_id=str(case["case_id"]),
            exclude_conversation_id=case.get("conversation_id"),
        )
        draft = self._drafter.draft(case, precedents)
        policy = decide_handling(
            intent=draft.predicted_intent,
            intent_confidence=draft.intent_confidence,
            message=message,
            prior_context=prior_context,
            draft_reply=draft.draft_reply,
        )
        reply = draft.draft_reply
        guardrail_flags = tuple(f"draft_guardrail:{flag}" for flag in policy.reply_flags)
        draft_was_replaced = bool(guardrail_flags)
        if draft_was_replaced:
            reply = safe_template(draft.predicted_intent)

        return AgentOutput(
            case_id=str(case["case_id"]),
            system_name=self.name,
            predicted_intent=draft.predicted_intent,
            intent_confidence=draft.intent_confidence,
            proposed_draft=draft.draft_reply,
            draft_reply=reply,
            draft_was_replaced=draft_was_replaced,
            handling_decision=policy.handling_decision,
            decision_reason=policy.reason,
            automation_score=policy.automation_score,
            evidence_case_ids=tuple(item.case_id for item in precedents),
            evidence_quotes=tuple(item.historical_reply for item in precedents),
            evidence_scores=tuple(item.rerank_score for item in precedents),
            safety_flags=tuple(sorted({*policy.input_flags, *guardrail_flags})),
        )
