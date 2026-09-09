"""Conservative static checks for unsafe public support drafts."""

from __future__ import annotations

import re

_PRIVATE_CHANNEL = re.compile(r"\b(?:dm|direct message|private message)\b", re.I)
_PERSONAL_DATA_REQUEST = re.compile(
    r"\b(?:dm|send|provide|confirm|share|tell)\b.{0,80}"
    r"\b(?:full name|address|postcode|post code|e-?mail|phone|mobile|"
    r"order number|clubcard number|account number|contact details)\b",
    re.I,
)
_BACKEND_ACTION = re.compile(
    r"\b(?:i|we)(?:'ve| have| just)?\s+"
    r"(?:refunded|credited|changed|updated|cancelled|canceled|arranged|processed)\b",
    re.I,
)
_FINANCIAL_PROMISE = re.compile(
    r"\b(?:you(?:'ll| will) receive|refund (?:will|should)|credit (?:will|should)|"
    r"money (?:will|should) be)\b",
    re.I,
)
_URL = re.compile(r"https?://|www\.", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")


def inspect_reply(text: str, max_length: int = 280) -> tuple[str, ...]:
    """Return stable flags; absence of flags is not a quality guarantee."""
    value = str(text or "")
    flags: list[str] = []
    checks = (
        ("private_channel_request", _PRIVATE_CHANNEL),
        ("personal_data_request", _PERSONAL_DATA_REQUEST),
        ("unsupported_backend_action", _BACKEND_ACTION),
        ("unsupported_financial_promise", _FINANCIAL_PROMISE),
        ("external_or_legacy_url", _URL),
        ("literal_email", _EMAIL),
        ("literal_phone", _PHONE),
    )
    for flag, pattern in checks:
        if pattern.search(value):
            flags.append(flag)
    if len(value) > max_length:
        flags.append("over_public_length_limit")
    return tuple(flags)
