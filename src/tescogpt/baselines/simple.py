"""Keyword routing plus nearest historical reply baseline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tescogpt.agent.intent import classify_intent
from tescogpt.agent.schema import AgentOutput
from tescogpt.data.cases import challenge_flags
from tescogpt.retrieval.bm25 import BM25Index


def route_case(intent: str, text: str) -> tuple[str, str, tuple[str, ...]]:
    """Apply intentionally simple, inspectable safety routing rules."""
    flags = tuple(challenge_flags(text))
    if {"food_safety", "injury_or_allergy"} & set(flags):
        return "ESCALATE", "FOOD_SAFETY_OR_INJURY", flags
    if intent in {"delivery_or_collection", "online_account_or_checkout"}:
        return "ESCALATE", "ACCOUNT_OR_ORDER_LOOKUP", flags
    if intent in {"pricing_promotion_or_clubcard", "refund_return_or_exchange"}:
        return "ESCALATE", "MONEY_OR_COMMITMENT", flags
    if "image_or_link_dependent" in flags:
        return "ESCALATE", "IMAGE_OR_MISSING_CONTEXT", flags
    if intent in {"product_availability", "product_information"}:
        return "ESCALATE", "CURRENT_POLICY_OR_LIVE_INFO", flags
    if {"repeated_failure", "strong_distress"} & set(flags):
        return "ESCALATE", "REPEATED_FAILURE_OR_DISTRESS", flags
    if intent == "other_or_unclear":
        return "ESCALATE", "OUT_OF_SCOPE_OR_UNCLEAR", flags
    if intent == "feedback_praise_or_suggestion":
        return "AUTO_HANDLE", "NO_ACTION_NEEDED", flags
    if intent == "store_or_staff_experience":
        return "ESCALATE", "HUMAN_JUDGMENT_REQUIRED", flags
    return "AUTO_HANDLE", "SAFE_CLARIFICATION", flags


class SimpleBaseline:
    name = "simple_rules_bm25_v1"

    def __init__(self, corpus: pd.DataFrame) -> None:
        self._index = BM25Index(corpus)

    def predict(self, case: Mapping[str, Any]) -> AgentOutput:
        message = str(case.get("message", ""))
        prior_context = str(case.get("prior_context", ""))
        query = f"{prior_context} {message}".strip()
        intent, confidence = classify_intent(query)
        handling, reason, flags = route_case(intent, query)
        automation_score = {
            "NO_ACTION_NEEDED": 0.80,
            "SAFE_PUBLIC_GUIDANCE": 0.70,
            "SAFE_CLARIFICATION": 0.65,
            "CURRENT_POLICY_OR_LIVE_INFO": 0.20,
            "OUT_OF_SCOPE_OR_UNCLEAR": 0.15,
        }.get(reason, 0.05)
        results = self._index.search(
            query,
            top_k=1,
            exclude_case_id=str(case["case_id"]),
            exclude_conversation_id=case.get("conversation_id"),
        )
        if results:
            evidence_ids = (results[0].case_id,)
            evidence_quotes = (results[0].historical_reply,)
            evidence_scores = (results[0].score,)
            draft = results[0].historical_reply
        else:
            evidence_ids = ()
            evidence_quotes = ()
            evidence_scores = ()
            draft = "Thanks for getting in touch. Could you tell us a little more?"
        return AgentOutput(
            case_id=str(case["case_id"]),
            system_name=self.name,
            predicted_intent=intent,
            intent_confidence=confidence,
            draft_reply=draft,
            handling_decision=handling,
            decision_reason=reason,
            automation_score=automation_score,
            evidence_case_ids=evidence_ids,
            evidence_quotes=evidence_quotes,
            evidence_scores=evidence_scores,
            safety_flags=flags,
        )
