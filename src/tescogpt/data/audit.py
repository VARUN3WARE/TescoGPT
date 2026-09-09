"""Aggregate, reproducible audits for reconstructed brand conversations."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

_PRIVATE_HANDOFF = re.compile(
    r"\b(?:dm|dms|direct message|private message|private channel|privately|follow us)\b",
    re.IGNORECASE,
)
_ACTION_LANGUAGE = re.compile(
    r"\b(?:try|check|open|select|tap|go to|settings|restart|reset|update|uninstall|"
    r"reinstall|track|contact|visit|click|follow|turn off|turn on|clear|sign out|sign in)\b",
    re.IGNORECASE,
)
_EMPATHY = re.compile(
    r"\b(?:sorry|apologi[sz]e|regret|understand|frustrat|disappoint|concern)\w*\b",
    re.IGNORECASE,
)
_POSITIVE_FOLLOWUP = re.compile(
    r"\b(?:thank|thanks|thx|worked|working now|fixed|resolved|sorted|all good|"
    r"that did it|problem solved|great)\b",
    re.IGNORECASE,
)
_UNRESOLVED_FOLLOWUP = re.compile(
    r"\b(?:still|not working|didn't work|doesn't work|nope|again|same issue|"
    r"waiting|nothing|worse|unresolved)\b",
    re.IGNORECASE,
)
_NUMBERED_FRAGMENT = re.compile(r"\b\d+/\d+\b")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _percentage(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 2) if denominator else 0.0


def _normalize_reply(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"https?://\S+", "<url>", normalized)
    normalized = re.sub(r"@\w+", "<user>", normalized)
    normalized = re.sub(r"\b\d+\b", "<number>", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _read_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Prepare manifest does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_table(messages: pd.DataFrame, brand: str) -> None:
    required = {
        "conversation_id",
        "thread_depth",
        "tweet_id",
        "parent_tweet_id",
        "author_id",
        "author_role",
        "inbound",
        "created_at",
        "text",
    }
    missing = sorted(required - set(messages.columns))
    if missing:
        raise ValueError(f"Conversation table is missing columns: {', '.join(missing)}")
    if messages.empty:
        raise ValueError("Conversation table is empty")
    if not messages["author_id"].eq(brand).any():
        raise ValueError(f"Conversation table contains no messages from brand {brand!r}")
    if messages["tweet_id"].duplicated().any():
        raise ValueError("Conversation table contains duplicate tweet_id values")


def _render_markdown(audit: dict[str, Any]) -> str:
    structure = audit["structure"]
    behavior = audit["brand_reply_behavior"]
    outcomes = audit["customer_followup_proxies"]
    provenance = audit["provenance"]
    month_lines = "\n".join(
        f"| {month} | {count:,} |" for month, count in audit["date_distribution"].items()
    )
    behavior_lines = "\n".join(
        [
            _metric_row(
                "Direct replies to customer/other messages",
                behavior["direct_reply_to_customer_count"],
                behavior["direct_reply_to_customer_pct"],
            ),
            _metric_row(
                "Private-channel handoff language",
                behavior["private_handoff_count"],
                behavior["private_handoff_pct"],
            ),
            _metric_row("Contains a URL", behavior["url_count"], behavior["url_pct"]),
            _metric_row(
                "Numbered fragment such as 1/3",
                behavior["numbered_fragment_count"],
                behavior["numbered_fragment_pct"],
            ),
            _metric_row(
                "Contains action language",
                behavior["action_language_count"],
                behavior["action_language_pct"],
            ),
            _metric_row(
                "Contains apology/empathy language",
                behavior["empathy_count"],
                behavior["empathy_pct"],
            ),
            _metric_row(
                "Contains a question mark",
                behavior["question_count"],
                behavior["question_pct"],
            ),
        ]
    )
    outcome_lines = "\n".join(
        [
            "| Direct follow-ups | " f"{outcomes['followup_count']:,} | 100.00% |",
            _metric_row(
                "Contains a positive cue",
                outcomes["positive_cue_count"],
                outcomes["positive_cue_pct"],
            ),
            _metric_row(
                "Contains an unresolved cue",
                outcomes["unresolved_cue_count"],
                outcomes["unresolved_cue_pct"],
            ),
            _metric_row(
                "Contains both",
                outcomes["both_cues_count"],
                outcomes["both_cues_pct"],
            ),
        ]
    )
    return f"""# Tesco conversation data audit

This report is generated from code. It profiles the reconstructed public reply
graphs before taxonomy design or model training.

## Provenance

- Input: `{provenance['input_file']}`
- Input SHA-256: `{provenance['input_sha256']}`
- Source rows: {provenance.get('source_rows', 'not supplied')}
- Source SHA-256: `{provenance.get('source_sha256', 'not supplied')}`

## Structure

| Measure | Value |
|---|---:|
| Messages | {structure['message_count']:,} |
| Conversations | {structure['conversation_count']:,} |
| Tesco messages | {structure['brand_message_count']:,} |
| Customer/other messages | {structure['customer_or_other_message_count']:,} |
| Median messages per conversation | {structure['messages_per_conversation_p50']:.1f} |
| 90th percentile messages per conversation | {structure['messages_per_conversation_p90']:.1f} |
| Maximum messages in one conversation | {structure['messages_per_conversation_max']:,} |
| Maximum graph depth | {structure['max_thread_depth']:,} |
| Conversations with at least 3 messages | {structure['multi_turn_conversation_pct']:.2f}% |
| Rows whose parent is unavailable | {structure['missing_parent_count']:,} |

## Brand reply behavior

These are regex-based descriptive indicators, not quality labels.

| Indicator | Count | Share of Tesco messages |
|---|---:|---:|
{behavior_lines}

Normalized unique reply share is {behavior['normalized_unique_reply_pct']:.2f}%.
The ten most common normalized templates account for
{behavior['top_10_template_share_pct']:.2f}% of Tesco messages.

## Customer follow-up proxies

Only customer/other messages directly replying to a Tesco-authored message are
included here.

| Indicator | Count | Share of direct customer follow-ups |
|---|---:|---:|
{outcome_lines}

Positive language is **not** a resolution label: “thanks, but it still does not
work” contains both signals, and “thanks, I sent a DM” confirms only a handoff.
These proxies are used to draw annotation candidates, never as golden truth.

## Date distribution

| Month | Messages |
|---|---:|
{month_lines}

## Consequences for the system

1. Reconstruct and split complete conversations before creating examples.
2. Join numbered brand fragments before treating them as evidence.
3. Separate public triage, private handoff, unknown outcome, explicit failure,
   and human-verified success.
4. Treat URLs and operational policy as historical artifacts, not current truth.
5. Require human review for food safety, allergens, injuries, refunds, private
   account access, and image-dependent complaints.
"""


def _metric_row(label: str, count: int, percentage: float) -> str:
    return f"| {label} | {count:,} | {percentage:.2f}% |"


def audit_conversations(
    input_path: str | Path,
    output_path: str | Path,
    markdown_path: str | Path,
    brand: str = "Tesco",
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Profile a prepared conversation table and write JSON and Markdown outputs."""
    source = Path(input_path)
    output = Path(output_path)
    markdown = Path(markdown_path)
    manifest_file = Path(manifest_path) if manifest_path is not None else None
    if not source.is_file():
        raise FileNotFoundError(f"Conversation table does not exist: {source}")

    messages = pd.read_csv(
        source,
        dtype={
            "conversation_id": "int64",
            "thread_depth": "int32",
            "tweet_id": "int64",
            "parent_tweet_id": "int64",
            "author_id": "string",
            "author_role": "string",
            "text": "string",
            "created_at": "string",
        },
    )
    _validate_table(messages, brand)
    manifest = _read_manifest(manifest_file)

    role_by_id = messages.set_index("tweet_id")["author_role"]
    parent_roles = messages["parent_tweet_id"].map(role_by_id)
    brand_mask = messages["author_id"].eq(brand)
    customer_mask = ~brand_mask
    brand_messages = messages.loc[brand_mask]
    brand_text = brand_messages["text"].fillna("").astype(str)
    followup_mask = customer_mask & parent_roles.eq("brand")
    followup_text = messages.loc[followup_mask, "text"].fillna("").astype(str)

    private_handoff = brand_text.str.contains(_PRIVATE_HANDOFF)
    urls = brand_text.str.contains(_URL)
    numbered = brand_text.str.contains(_NUMBERED_FRAGMENT)
    actions = brand_text.str.contains(_ACTION_LANGUAGE)
    empathy = brand_text.str.contains(_EMPATHY)
    questions = brand_text.str.contains(r"\?", regex=True)
    positive = followup_text.str.contains(_POSITIVE_FOLLOWUP)
    unresolved = followup_text.str.contains(_UNRESOLVED_FOLLOWUP)

    conversation_sizes = messages.groupby("conversation_id").size()
    normalized_counts = brand_text.map(_normalize_reply).value_counts()
    month = pd.to_datetime(
        messages["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        utc=True,
        errors="coerce",
    ).dt.strftime("%Y-%m")
    month_counts = month.value_counts().sort_index()

    brand_count = int(brand_mask.sum())
    followup_count = int(followup_mask.sum())
    direct_reply = int((brand_mask & parent_roles.eq("customer_or_other")).sum())
    missing_parent = int(
        (
            messages["parent_tweet_id"].ge(0)
            & ~messages["parent_tweet_id"].isin(messages["tweet_id"])
        ).sum()
    )
    positive_count = int(positive.sum())
    unresolved_count = int(unresolved.sum())
    both_count = int((positive & unresolved).sum())

    audit: dict[str, Any] = {
        "audit_schema_version": 1,
        "brand": brand,
        "provenance": {
            "input_file": source.name,
            "input_sha256": _sha256(source),
            "source_file": manifest.get("source_file") if manifest else None,
            "source_rows": manifest.get("source_rows") if manifest else None,
            "source_sha256": manifest.get("source_sha256") if manifest else None,
        },
        "structure": {
            "message_count": len(messages),
            "conversation_count": int(messages["conversation_id"].nunique()),
            "brand_message_count": brand_count,
            "customer_or_other_message_count": int(customer_mask.sum()),
            "messages_per_conversation_p50": round(float(conversation_sizes.quantile(0.5)), 2),
            "messages_per_conversation_p90": round(float(conversation_sizes.quantile(0.9)), 2),
            "messages_per_conversation_max": int(conversation_sizes.max()),
            "max_thread_depth": int(messages["thread_depth"].max()),
            "multi_turn_conversation_count": int((conversation_sizes >= 3).sum()),
            "multi_turn_conversation_pct": _percentage(
                int((conversation_sizes >= 3).sum()), len(conversation_sizes)
            ),
            "missing_parent_count": missing_parent,
        },
        "brand_reply_behavior": {
            "direct_reply_to_customer_count": direct_reply,
            "direct_reply_to_customer_pct": _percentage(direct_reply, brand_count),
            "private_handoff_count": int(private_handoff.sum()),
            "private_handoff_pct": _percentage(int(private_handoff.sum()), brand_count),
            "url_count": int(urls.sum()),
            "url_pct": _percentage(int(urls.sum()), brand_count),
            "numbered_fragment_count": int(numbered.sum()),
            "numbered_fragment_pct": _percentage(int(numbered.sum()), brand_count),
            "action_language_count": int(actions.sum()),
            "action_language_pct": _percentage(int(actions.sum()), brand_count),
            "empathy_count": int(empathy.sum()),
            "empathy_pct": _percentage(int(empathy.sum()), brand_count),
            "question_count": int(questions.sum()),
            "question_pct": _percentage(int(questions.sum()), brand_count),
            "normalized_unique_reply_count": int(len(normalized_counts)),
            "normalized_unique_reply_pct": _percentage(len(normalized_counts), brand_count),
            "top_10_template_count": int(normalized_counts.head(10).sum()),
            "top_10_template_share_pct": _percentage(
                int(normalized_counts.head(10).sum()), brand_count
            ),
        },
        "customer_followup_proxies": {
            "followup_count": followup_count,
            "positive_cue_count": positive_count,
            "positive_cue_pct": _percentage(positive_count, followup_count),
            "unresolved_cue_count": unresolved_count,
            "unresolved_cue_pct": _percentage(unresolved_count, followup_count),
            "both_cues_count": both_count,
            "both_cues_pct": _percentage(both_count, followup_count),
        },
        "date_distribution": {str(key): int(value) for key, value in month_counts.items()},
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown.write_text(_render_markdown(audit), encoding="utf-8", newline="\n")
    return audit
