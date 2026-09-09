"""Build point-in-time customer cases from reconstructed conversation graphs."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from tescogpt.data.threads import ConversationDataError

CASE_COLUMNS = [
    "case_id",
    "conversation_id",
    "tweet_id",
    "created_at",
    "message",
    "prior_context",
    "historical_reply",
    "historical_reply_ids",
    "historical_reply_fragment_count",
    "customer_followup",
    "private_handoff_proxy",
    "positive_outcome_proxy",
    "unresolved_outcome_proxy",
    "challenge_flags",
]

_NUMERIC_HANDLE = re.compile(r"@\d+")
_TCO_URL = re.compile(r"https?://t\.co/\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")
_GREETING_NAME = re.compile(
    r"(?P<prefix>(?:@customer\s+)?(?:[Hh]i|[Hh]ello|[Hh]ey|[Hh]iya|[Mm]orning|"
    r"[Aa]fternoon|[Ee]vening)\s*[,!-]?\s+)"
    r"(?!There\b|All\b|Team\b|Can\b|Could\b|So\b|Our\b|We\b|Thanks\b|"
    r"Thank\b|Sorry\b|Hope\b|Please\b|Just\b|This\b|Everyone\b|Tesco\b)"
    r"(?P<name>[A-Z][A-Za-z'-]{1,30})(?P<punctuation>[,!.:]|\s)",
)
_AGENT_SIGNOFF = re.compile(r"(?:\^|\s[-–—]\s)[A-Z][A-Za-z .'-]{1,30}$")
_WHITESPACE = re.compile(r"\s+")

_PRIVATE_HANDOFF = re.compile(
    r"\b(?:dm|dms|direct message|private message|private channel|privately|follow us)\b",
    re.IGNORECASE,
)
_POSITIVE = re.compile(
    r"\b(?:thank|thanks|thx|worked|working now|fixed|resolved|sorted|all good|great)\b",
    re.IGNORECASE,
)
_UNRESOLVED = re.compile(
    r"\b(?:still|not working|didn't work|doesn't work|nope|again|same issue|"
    r"waiting|nothing|worse|unresolved)\b",
    re.IGNORECASE,
)

_CHALLENGE_PATTERNS: dict[str, re.Pattern[str]] = {
    "food_safety": re.compile(
        r"\b(?:mould|mold|rotten|raw|poison|contaminat|glass|metal|bone|slug|worm|"
        r"maggot|bug|insect|foreign object|expired|out of date|tamper)\w*\b",
        re.IGNORECASE,
    ),
    "injury_or_allergy": re.compile(
        r"\b(?:allerg|anaphyla|injur|hurt|hospital|doctor|chok|cut|burn|sick|ill|"
        r"broke?\s+(?:a\s+)?tooth)\w*\b",
        re.IGNORECASE,
    ),
    "money": re.compile(
        r"\b(?:refund|charged|charge|money|voucher|compensat|credit|price|overcharg|"
        r"payment|cash)\w*\b|[£$€]",
        re.IGNORECASE,
    ),
    "account_or_order": re.compile(
        r"\b(?:account|login|log in|password|clubcard|order number|delivery|checkout|"
        r"click and collect|click & collect)\b",
        re.IGNORECASE,
    ),
    "repeated_failure": re.compile(
        r"\b(?:again|still|every time|third time|several times|no reply|not replied|"
        r"keeps? (?:happening|failing)|already (?:tried|contacted))\b",
        re.IGNORECASE,
    ),
    "strong_distress": re.compile(
        r"\b(?:disgusting|unacceptable|furious|outrage|worst|shocking|dangerous|"
        r"fuck|fucking|shit|damn|never again)\b|!{2,}",
        re.IGNORECASE,
    ),
}


def sanitize_public_text(text: str) -> str:
    """Remove link tokens and contact-like strings while preserving meaning."""
    value = str(text or "")
    value = _NUMERIC_HANDLE.sub("@customer", value)
    value = _TCO_URL.sub("<media_or_link>", value)
    value = _EMAIL.sub("<email_redacted>", value)
    value = _PHONE.sub("<number_redacted>", value)
    value = _GREETING_NAME.sub(
        r"\g<prefix><name_redacted>\g<punctuation>",
        value,
    )
    value = _AGENT_SIGNOFF.sub("", value)
    return _WHITESPACE.sub(" ", value).strip()


def challenge_flags(message: str, prior_context: str = "") -> list[str]:
    combined = f"{prior_context} {message}".strip()
    flags = [name for name, pattern in _CHALLENGE_PATTERNS.items() if pattern.search(combined)]
    if "<media_or_link>" in message:
        flags.append("image_or_link_dependent")
    words = re.findall(r"\b\w+\b", re.sub(r"@\w+", "", message))
    if len(words) <= 5:
        flags.append("short_message")
    if re.match(r"^(?:@\w+\s*)?(?:yes|no|it|that|this|they|same|done|sent)\b", message, re.I):
        flags.append("context_dependent")
    if message.count("?") + message.count("!") >= 3:
        flags.append("punctuation_heavy")
    return sorted(set(flags))


def _collect_brand_response_ids(
    target_id: int,
    children: dict[int, list[int]],
    rows: dict[int, dict[str, Any]],
) -> list[int]:
    response_ids: list[int] = []
    frontier = [
        child_id
        for child_id in children.get(target_id, [])
        if rows[child_id]["author_role"] == "brand"
    ]
    seen: set[int] = set()
    while frontier:
        current = frontier.pop(0)
        if current in seen:
            raise ConversationDataError("Cycle found while collecting brand reply fragments")
        seen.add(current)
        response_ids.append(current)
        frontier.extend(
            child_id
            for child_id in children.get(current, [])
            if rows[child_id]["author_role"] == "brand" and child_id not in seen
        )
    return sorted(
        response_ids,
        key=lambda item: (
            rows[item]["thread_depth"],
            rows[item]["_created_at_sort"],
            item,
        ),
    )


def _collect_prior_context(
    target_id: int,
    rows: dict[int, dict[str, Any]],
    max_messages: int,
) -> str:
    context: list[str] = []
    current = int(rows[target_id]["parent_tweet_id"])
    seen: set[int] = set()
    while current >= 0 and current in rows and len(context) < max_messages:
        if current in seen:
            raise ConversationDataError("Cycle found while collecting prior context")
        seen.add(current)
        role = "Tesco" if rows[current]["author_role"] == "brand" else "Customer"
        context.append(f"[{role}] {sanitize_public_text(rows[current]['text'])}")
        current = int(rows[current]["parent_tweet_id"])
    return " || ".join(reversed(context))


def build_cases(
    messages: pd.DataFrame,
    brand: str = "Tesco",
    max_context_messages: int = 4,
) -> pd.DataFrame:
    """Create one case for each customer message that received a brand reply."""
    required = {
        "conversation_id",
        "thread_depth",
        "tweet_id",
        "parent_tweet_id",
        "author_id",
        "author_role",
        "created_at",
        "text",
    }
    missing = sorted(required - set(messages.columns))
    if missing:
        raise ValueError(f"Prepared messages are missing columns: {', '.join(missing)}")
    if messages["tweet_id"].duplicated().any():
        raise ValueError("Prepared messages contain duplicate tweet IDs")
    if max_context_messages < 0:
        raise ValueError("max_context_messages must be non-negative")

    records = messages.copy()
    records["author_role"] = records["author_role"].astype(str)
    records["text"] = records["text"].fillna("").astype(str)
    records["parent_tweet_id"] = (
        pd.to_numeric(records["parent_tweet_id"], errors="coerce").fillna(-1).astype("int64")
    )
    records["_created_at_sort"] = pd.to_datetime(
        records["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        utc=True,
        errors="coerce",
    )
    if records["_created_at_sort"].isna().any():
        raise ValueError("Prepared messages contain an unparseable created_at timestamp")
    rows = records.set_index("tweet_id").to_dict(orient="index")
    children: dict[int, list[int]] = defaultdict(list)
    for row in records.itertuples(index=False):
        parent_id = int(row.parent_tweet_id)
        if parent_id >= 0 and parent_id in rows:
            children[parent_id].append(int(row.tweet_id))

    cases: list[dict[str, Any]] = []
    customer_rows = records.loc[records["author_role"] != "brand"]
    for target in customer_rows.itertuples(index=False):
        target_id = int(target.tweet_id)
        response_ids = _collect_brand_response_ids(target_id, children, rows)
        if not response_ids:
            continue

        prior_context = _collect_prior_context(target_id, rows, max_context_messages)
        message = sanitize_public_text(target.text)
        reply_parts = [sanitize_public_text(rows[item]["text"]) for item in response_ids]
        reply = " ".join(part for part in reply_parts if part)
        followup_ids = sorted(
            {
                child_id
                for response_id in response_ids
                for child_id in children.get(response_id, [])
                if rows[child_id]["author_role"] != "brand"
            },
            key=lambda item: (rows[item]["_created_at_sort"], item),
        )
        followup = " || ".join(
            sanitize_public_text(rows[item]["text"]) for item in followup_ids
        )
        flags = challenge_flags(message, prior_context)
        cases.append(
            {
                "case_id": f"tesco-{target_id}",
                "conversation_id": int(target.conversation_id),
                "tweet_id": target_id,
                "created_at": str(target.created_at),
                "message": message,
                "prior_context": prior_context,
                "historical_reply": reply,
                "historical_reply_ids": ";".join(str(item) for item in response_ids),
                "historical_reply_fragment_count": len(response_ids),
                "customer_followup": followup,
                "private_handoff_proxy": bool(_PRIVATE_HANDOFF.search(reply)),
                "positive_outcome_proxy": bool(_POSITIVE.search(followup)),
                "unresolved_outcome_proxy": bool(_UNRESOLVED.search(followup)),
                "challenge_flags": ";".join(flags),
            }
        )

    if not cases:
        raise ValueError(f"No customer messages with replies from {brand!r} were found")
    output = pd.DataFrame(cases, columns=CASE_COLUMNS)
    output["_created_at_sort"] = pd.to_datetime(
        output["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        utc=True,
        errors="raise",
    )
    return output.sort_values(
        ["_created_at_sort", "conversation_id", "tweet_id"], kind="stable"
    ).drop(columns="_created_at_sort")


def build_cases_from_csv(
    input_path: str | Path,
    output_path: str | Path,
    brand: str = "Tesco",
    max_context_messages: int = 4,
) -> pd.DataFrame:
    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Prepared conversation CSV does not exist: {source}")
    messages = pd.read_csv(source)
    cases = build_cases(messages, brand=brand, max_context_messages=max_context_messages)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(destination, index=False, lineterminator="\n")
    return cases
