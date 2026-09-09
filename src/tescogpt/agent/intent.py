"""Shared deterministic intent heuristic used as baseline and safe fallback."""

from __future__ import annotations

import re

_INTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "product_quality_or_safety",
        re.compile(
            r"\b(?:allerg|anaphyla|mould|mold|rotten|raw|poison|contaminat|glass|"
            r"foreign object|expired|out of date|injur|hurt|sick|ill|quality|damaged)\w*\b",
            re.I,
        ),
    ),
    (
        "delivery_or_collection",
        re.compile(
            r"\b(?:deliver|driver|order|slot|substitut|missing (?:bag|item)|"
            r"click\s*(?:&|and)\s*collect)\w*\b",
            re.I,
        ),
    ),
    (
        "online_account_or_checkout",
        re.compile(
            r"\b(?:login|log in|password|account|website|app|basket|checkout|"
            r"payment (?:fail|error))\w*\b",
            re.I,
        ),
    ),
    (
        "store_or_staff_experience",
        re.compile(
            r"\b(?:staff|manager|cashier|queue|store|shop|toilet|trolley|parking|"
            r"opening hours?|customer service)\w*\b",
            re.I,
        ),
    ),
    (
        "pricing_promotion_or_clubcard",
        re.compile(
            r"\b(?:clubcard|price|offer|discount|coupon|voucher|points?|promotion|"
            r"overcharg|gift card)\w*\b|[£$€]",
            re.I,
        ),
    ),
    (
        "refund_return_or_exchange",
        re.compile(r"\b(?:refund|return|exchange|replacement|money back)\w*\b", re.I),
    ),
    (
        "product_availability",
        re.compile(
            r"\b(?:stock|sell|sold out|available|availability|discontinued|"
            r"bring back|find this)\w*\b",
            re.I,
        ),
    ),
    (
        "product_information",
        re.compile(
            r"\b(?:ingredient|nutrition|calorie|vegan|vegetarian|gluten|allergen|"
            r"contain|cook|suitable|packag|source)\w*\b",
            re.I,
        ),
    ),
    (
        "feedback_praise_or_suggestion",
        re.compile(
            r"\b(?:thank|thanks|brilliant|great service|well done|love|suggestion|"
            r"feedback)\w*\b",
            re.I,
        ),
    ),
)


def classify_intent(text: str) -> tuple[str, float]:
    """Return the first matching taxonomy intent using codebook priority."""
    for intent, pattern in _INTENT_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            confidence = min(0.55 + 0.08 * len(matches), 0.87)
            return intent, confidence
    return "other_or_unclear", 0.2
