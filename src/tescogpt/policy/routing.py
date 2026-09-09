"""Deterministic handling gate for support-agent drafts."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tescogpt.data.cases import challenge_flags
from tescogpt.evaluation.labels import ESCALATION_REASON_CODES
from tescogpt.policy.safety import inspect_reply

_POSITIVE_FEEDBACK = re.compile(
    r"\b(?:thank|thanks|brilliant|excellent|lovely|helpful|great service|well done|love)\w*\b",
    re.I,
)

_ESCALATION_BASE_SCORES = {
    "FOOD_SAFETY_OR_INJURY": 0.0,
    "ACCOUNT_OR_ORDER_LOOKUP": 0.05,
    "MONEY_OR_COMMITMENT": 0.05,
    "PERSONAL_DATA_OR_PRIVATE_CHANNEL": 0.0,
    "IMAGE_OR_MISSING_CONTEXT": 0.05,
    "CURRENT_POLICY_OR_LIVE_INFO": 0.15,
    "REPEATED_FAILURE_OR_DISTRESS": 0.05,
    "HUMAN_JUDGMENT_REQUIRED": 0.15,
    "OUT_OF_SCOPE_OR_UNCLEAR": 0.10,
}

if set(_ESCALATION_BASE_SCORES) != set(ESCALATION_REASON_CODES):
    raise RuntimeError("Routing scores must cover every frozen escalation reason")


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

    reply_flag_set = set(reply_flags)
    escalation_reasons: set[str] = set()
    mapped_reply_flags: set[str] = set()
    personal_reply_flags = {
        "private_channel_request",
        "personal_data_request",
        "literal_email",
        "literal_phone",
    }
    money_reply_flags = {
        "unsupported_backend_action",
        "unsupported_financial_promise",
    }
    if personal_reply_flags & reply_flag_set:
        escalation_reasons.add("PERSONAL_DATA_OR_PRIVATE_CHANNEL")
        mapped_reply_flags.update(personal_reply_flags & reply_flag_set)
    if money_reply_flags & reply_flag_set:
        escalation_reasons.add("MONEY_OR_COMMITMENT")
        mapped_reply_flags.update(money_reply_flags & reply_flag_set)
    if "external_or_legacy_url" in reply_flag_set:
        escalation_reasons.add("CURRENT_POLICY_OR_LIVE_INFO")
        mapped_reply_flags.add("external_or_legacy_url")
    if "over_public_length_limit" in reply_flag_set:
        escalation_reasons.add("HUMAN_JUDGMENT_REQUIRED")
        mapped_reply_flags.add("over_public_length_limit")
    if reply_flag_set - mapped_reply_flags:
        escalation_reasons.add("HUMAN_JUDGMENT_REQUIRED")

    if "food_safety" in flag_set or "injury_or_allergy" in flag_set:
        escalation_reasons.add("FOOD_SAFETY_OR_INJURY")
    if intent in {"delivery_or_collection", "online_account_or_checkout"}:
        escalation_reasons.add("ACCOUNT_OR_ORDER_LOOKUP")
    if intent in {"pricing_promotion_or_clubcard", "refund_return_or_exchange"}:
        escalation_reasons.add("MONEY_OR_COMMITMENT")
    if "<email_redacted>" in message or "<number_redacted>" in message:
        escalation_reasons.add("PERSONAL_DATA_OR_PRIVATE_CHANNEL")
    if "image_or_link_dependent" in flag_set or "context_dependent" in flag_set:
        escalation_reasons.add("IMAGE_OR_MISSING_CONTEXT")
    if intent in {"product_availability", "product_information"}:
        escalation_reasons.add("CURRENT_POLICY_OR_LIVE_INFO")
    if "repeated_failure" in flag_set or "strong_distress" in flag_set:
        escalation_reasons.add("REPEATED_FAILURE_OR_DISTRESS")

    positive_store_feedback = bool(
        intent == "store_or_staff_experience"
        and _POSITIVE_FEEDBACK.search(f"{prior_context} {message}")
    )
    if intent == "store_or_staff_experience" and not positive_store_feedback:
        escalation_reasons.add("HUMAN_JUDGMENT_REQUIRED")

    if escalation_reasons:
        reason = next(
            code for code in ESCALATION_REASON_CODES if code in escalation_reasons
        )
        base_score = 0.0 if reply_flag_set else _ESCALATION_BASE_SCORES[reason]
        handling_decision = "ESCALATE"
    elif positive_store_feedback:
        reason = "NO_ACTION_NEEDED"
        base_score = 0.90
        handling_decision = "AUTO_HANDLE"
    elif intent == "feedback_praise_or_suggestion":
        reason = "NO_ACTION_NEEDED"
        base_score = 0.95
        handling_decision = "AUTO_HANDLE"
    else:
        reason = "OUT_OF_SCOPE_OR_UNCLEAR"
        base_score = _ESCALATION_BASE_SCORES[reason]
        handling_decision = "ESCALATE"

    automation_score = round(base_score * max(0.0, min(intent_confidence, 1.0)), 6)
    return PolicyDecision(
        handling_decision=handling_decision,
        reason=reason,
        automation_score=automation_score,
        input_flags=input_flags,
        reply_flags=reply_flags,
    )
