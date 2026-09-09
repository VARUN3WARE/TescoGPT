from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tescogpt.data.threads import ConversationDataError, extract_brand_conversations

FIXTURE = Path(__file__).parent / "fixtures" / "twcs_mini.csv"


def test_extracts_complete_trees_containing_brand(tmp_path: Path) -> None:
    output = tmp_path / "tesco.csv"

    manifest = extract_brand_conversations(FIXTURE, output, brand="Tesco", chunk_size=3)
    messages = pd.read_csv(output)

    assert set(messages["tweet_id"]) == {1, 2, 3, 4, 5, 6, 20, 21}
    assert set(messages["conversation_id"]) == {1, 20}
    assert messages.set_index("tweet_id").loc[6, "thread_depth"] == 4
    assert messages.set_index("tweet_id").loc[20, "thread_depth"] == 0
    assert manifest["source_rows"] == 11
    assert manifest["output_rows"] == 8
    assert manifest["conversation_count"] == 2
    assert manifest["brand_message_count"] == 3
    assert manifest["missing_parent_count"] == 0


def test_manifest_and_output_are_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    first_manifest = extract_brand_conversations(FIXTURE, first, chunk_size=4)
    second_manifest = extract_brand_conversations(FIXTURE, second, chunk_size=7)

    assert first.read_bytes() == second.read_bytes()
    assert first_manifest["source_sha256"] == second_manifest["source_sha256"]
    assert first_manifest["output_sha256"] == second_manifest["output_sha256"]
    saved = json.loads(first.with_suffix(".csv.manifest.json").read_text(encoding="utf-8"))
    assert saved == first_manifest


def test_rejects_missing_brand(tmp_path: Path) -> None:
    with pytest.raises(ConversationDataError, match="No messages authored"):
        extract_brand_conversations(FIXTURE, tmp_path / "missing.csv", brand="NotARealBrand")


def test_rejects_nonpositive_chunk_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        extract_brand_conversations(FIXTURE, tmp_path / "tesco.csv", chunk_size=0)


def test_rejects_parent_cycles(tmp_path: Path) -> None:
    cyclic = tmp_path / "cyclic.csv"
    pd.DataFrame(
        [
            {
                "tweet_id": 1,
                "author_id": "Tesco",
                "inbound": False,
                "created_at": "Wed Nov 01 09:00:00 +0000 2017",
                "text": "first",
                "response_tweet_id": 2,
                "in_response_to_tweet_id": 2,
            },
            {
                "tweet_id": 2,
                "author_id": "100001",
                "inbound": True,
                "created_at": "Wed Nov 01 09:01:00 +0000 2017",
                "text": "second",
                "response_tweet_id": 1,
                "in_response_to_tweet_id": 1,
            },
        ]
    ).to_csv(cyclic, index=False)

    with pytest.raises(ConversationDataError, match="parent cycle"):
        extract_brand_conversations(cyclic, tmp_path / "tesco.csv", chunk_size=1)
