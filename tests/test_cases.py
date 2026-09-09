from __future__ import annotations

from pathlib import Path

import pandas as pd

from tescogpt.data.cases import build_cases, challenge_flags, sanitize_public_text
from tescogpt.data.threads import extract_brand_conversations

FIXTURE = Path(__file__).parent / "fixtures" / "twcs_mini.csv"


def test_builds_point_in_time_cases_without_future_reply_in_context(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.csv"
    extract_brand_conversations(FIXTURE, prepared, chunk_size=3)
    messages = pd.read_csv(prepared)

    cases = build_cases(messages)
    by_tweet = cases.set_index("tweet_id")

    assert set(cases["tweet_id"]) == {1, 4}
    assert by_tweet.loc[1, "prior_context"] == ""
    assert "Please tell us" in by_tweet.loc[1, "historical_reply"]
    assert "Please tell us" not in by_tweet.loc[1, "prior_context"]
    assert "[Tesco]" in by_tweet.loc[4, "prior_context"]
    assert "private channel" in by_tweet.loc[4, "historical_reply"]
    assert by_tweet.loc[4, "private_handoff_proxy"]
    assert by_tweet.loc[4, "positive_outcome_proxy"]


def test_sanitizes_public_identifiers_and_flags_hard_cases() -> None:
    sanitized = sanitize_public_text(
        "@12345 email me at person@example.com or +44 7700 900123 https://t.co/abc"
    )

    assert "@customer" in sanitized
    assert "person@example.com" not in sanitized
    assert "7700" not in sanitized
    assert "https://" not in sanitized
    assert "<media_or_link>" in sanitized
    flags = challenge_flags("@Tesco glass in my food!! https://t.co/abc")
    assert "food_safety" in flags
    assert "punctuation_heavy" not in flags
    assert "image_or_link_dependent" not in flags  # flags expect already-sanitized text
    assert "image_or_link_dependent" in challenge_flags(sanitized)
