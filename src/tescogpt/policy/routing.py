"""Deterministic handling gate for support-agent drafts."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tescogpt.data.cases import challenge_flags
from tescogpt.policy.safety import inspect_reply

_POSITIVE_FEEDBACK = re.compile(
    r"\b(?:thank|thanks|brilliant|excellent|lovely|helpful|great service|well done|love)\w*\b",
    re.I,
)


@dataclass(frozen=True)
class PolicyDecision:
    handling_decision: str
    reason: str
    automation_score: float
    input_flags: tuple[str, ...]
    reply_flags: tuple[str, ...]


def decide_handling(
    intent: str,
    intent_confidence: float,
    message: str,
    prior_context: str,
    draft_reply: str,
) -> PolicyDecision:
    """Choose automation only for narrow, reversible public responses."""
    input_flags = tuple(challenge_flags(message, prior_context))
    reply_flags = inspect_reply(draft_reply)
    flag_set = set(input_flags)

    if {"private_channel_request", "personal_data_request", "literal_email", "literal_phone"} & set(
        reply_flags
    ):
        reason = "PERSONAL_DATA_OR_PRIVATE_CHANNEL"
        base_score = 0.0
    elif {"unsupported_backend_action", "unsupported_financial_promise"} & set(reply_flags):
        reason = "MONEY_OR_COMMITMENT"
        base_score = 0.0
    elif "food_safety" in flag_set or "injury_or_allergy" in flag_set:
        reason = "FOOD_SAFETY_OR_INJURY"
        base_score = 0.0
    elif intent in {"delivery_or_collection", "online_account_or_checkout"}:
        reason = "ACCOUNT_OR_ORDER_LOOKUP"
        base_score = 0.05
    elif intent in {"pricing_promotion_or_clubcard", "refund_return_or_exchange"}:
        reason = "MONEY_OR_COMMITMENT"
        base_score = 0.05
    elif "<email_redacted>" in message or "<number_redacted>" in message:
        reason = "PERSONAL_DATA_OR_PRIVATE_CHANNEL"
        base_score = 0.0
    elif "image_or_link_dependent" in flag_set or "context_dependent" in flag_set:
        reason = "IMAGE_OR_MISSING_CONTEXT"
        base_score = 0.05
    elif intent in {"product_availability", "product_information"}:
        reason = "CURRENT_POLICY_OR_LIVE_INFO"
        base_score = 0.15
    elif "repeated_failure" in flag_set or "strong_distress" in flag_set:
        reason = "REPEATED_FAILURE_OR_DISTRESS"
        base_score = 0.05
    elif (
        intent == "store_or_staff_experience"
        and _POSITIVE_FEEDBACK.search(f"{prior_context} {message}")
    ):
        reason = "NO_ACTION_NEEDED"
        base_score = 0.90
    elif intent == "store_or_staff_experience":
        reason = "HUMAN_JUDGMENT_REQUIRED"
        base_score = 0.15
    elif intent == "feedback_praise_or_suggestion":
        reason = "NO_ACTION_NEEDED"
        base_score = 0.95
    else:
        reason = "OUT_OF_SCOPE_OR_UNCLEAR"
        base_score = 0.10

    automation_score = round(base_score * max(0.0, min(intent_confidence, 1.0)), 6)
    unsafe_reply = bool(reply_flags)
    may_auto_handle = reason == "NO_ACTION_NEEDED" and not unsafe_reply
    return PolicyDecision(
        handling_decision="AUTO_HANDLE" if may_auto_handle else "ESCALATE",
        reason=reason,
        automation_score=automation_score,
        input_flags=input_flags,
        reply_flags=reply_flags,
    )
