from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tescogpt.policy.safety import inspect_reply
from tescogpt.retrieval.outcome import OutcomeAwareRetriever, write_retrieval_artifact


def _row(
    case_id: str,
    *,
    positive: bool = False,
    unresolved: bool = False,
    private: bool = False,
    reply: str = "Thanks for the report.",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "conversation_id": case_id,
        "message": "damaged yoghurt pot",
        "historical_reply": reply,
        "customer_followup": "thanks sorted" if positive else "still broken" if unresolved else "",
        "private_handoff_proxy": private,
        "positive_outcome_proxy": positive,
        "unresolved_outcome_proxy": unresolved,
        "split": "train",
    }


def test_positive_followup_reranks_identical_lexical_match() -> None:
    corpus = pd.DataFrame(
        [
            _row("a-unresolved", unresolved=True),
            _row("z-positive", positive=True),
        ]
    )
    results = OutcomeAwareRetriever(corpus).search("damaged yoghurt pot", top_k=2)

    assert [result.case_id for result in results] == ["z-positive", "a-unresolved"]
    assert results[0].outcome_tier == "positive_followup_proxy"


def test_reply_inspection_surfaces_unsupported_historical_behavior() -> None:
    flags = inspect_reply(
        "DM your full name, address and order number. I've refunded you at https://old.test"
    )

    assert "private_channel_request" in flags
    assert "personal_data_request" in flags
    assert "unsupported_backend_action" in flags
    assert "external_or_legacy_url" in flags


def test_artifact_writer_requires_train_partition(tmp_path: Path) -> None:
    cases = pd.DataFrame(
        [{"case_id": "query", "conversation_id": "q", "message": "damaged pot"}]
    )
    corpus = pd.DataFrame([_row("train")]).drop(columns="split")
    cases_path = tmp_path / "cases.csv"
    corpus_path = tmp_path / "corpus.csv"
    cases.to_csv(cases_path, index=False)
    corpus.to_csv(corpus_path, index=False)

    with pytest.raises(ValueError, match="explicit split"):
        write_retrieval_artifact(cases_path, corpus_path, tmp_path / "retrieval.csv")
