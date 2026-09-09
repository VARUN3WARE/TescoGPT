from __future__ import annotations

from pathlib import Path

import pandas as pd

from tescogpt.data.threads import extract_brand_conversations
from tescogpt.evaluation.sampling import (
    assign_chronological_splits,
    sample_golden_candidates,
)

FIXTURE = Path(__file__).parent / "fixtures" / "twcs_mini.csv"
CODEBOOK = Path(__file__).parents[1] / "docs" / "ANNOTATION_GUIDE.md"


def test_chronological_split_keeps_conversations_together() -> None:
    messages = pd.DataFrame(
        {
            "conversation_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "created_at": [
                "Wed Nov 01 09:00:00 +0000 2017",
                "Wed Nov 01 09:01:00 +0000 2017",
                "Thu Nov 02 09:00:00 +0000 2017",
                "Thu Nov 02 09:01:00 +0000 2017",
                "Fri Nov 03 09:00:00 +0000 2017",
                "Fri Nov 03 09:01:00 +0000 2017",
                "Sat Nov 04 09:00:00 +0000 2017",
                "Sat Nov 04 09:01:00 +0000 2017",
            ],
        }
    )
    split, manifest = assign_chronological_splits(
        messages, train_fraction=0.5, dev_fraction=0.25
    )

    assert list(split) == ["train", "train", "train", "train", "dev", "dev", "test", "test"]
    assert manifest["splits"]["test"]["conversation_count"] == 1


def test_sampling_is_blind_unique_and_deterministic(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.csv"
    extract_brand_conversations(FIXTURE, prepared, chunk_size=3)

    def run(suffix: str) -> tuple[dict, Path]:
        candidates = tmp_path / f"candidates-{suffix}.csv"
        manifest = sample_golden_candidates(
            prepared_messages_path=prepared,
            cases_output_path=tmp_path / f"cases-{suffix}.csv",
            candidates_output_path=candidates,
            manifest_output_path=tmp_path / f"manifest-{suffix}.json",
            codebook_path=CODEBOOK,
            natural_count=1,
            challenge_count=0,
            seed=17,
            train_fraction=0,
            dev_fraction=0,
        )
        return manifest, candidates

    first_manifest, first_path = run("first")
    second_manifest, second_path = run("second")
    first = pd.read_csv(first_path)

    assert len(first) == 1
    assert first["conversation_id"].is_unique
    assert "historical_reply" not in first.columns
    assert first["intent_label"].isna().all()
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_manifest["candidate_sha256"] == second_manifest["candidate_sha256"]
