from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tescogpt.evaluation.labels import (
    LABEL_COLUMNS,
    freeze_human_annotations,
    initialize_annotation_rounds,
    validate_annotations,
)

CODEBOOK = Path(__file__).parents[1] / "docs" / "ANNOTATION_GUIDE.md"


def _sheet(path: Path, **labels: str) -> Path:
    row = {
        "case_id": "tesco-1",
        "conversation_id": "1",
        "tweet_id": "1",
        "message": "Where is my order?",
        **{column: "" for column in LABEL_COLUMNS},
        **labels,
    }
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def test_reports_unlabelled_and_complete_progress(tmp_path: Path) -> None:
    path = _sheet(tmp_path / "annotations.csv")
    progress = validate_annotations(path)
    assert progress["remaining_count"] == 1
    assert not progress["is_complete"]

    path = _sheet(
        path,
        intent_label="delivery_or_collection",
        handling_label="ESCALATE",
        reason_code="ACCOUNT_OR_ORDER_LOOKUP",
        risk_tags="account_or_order",
        annotator_id="annotator_a",
    )
    progress = validate_annotations(path, require_complete=True)
    assert progress["completed_count"] == 1
    assert progress["is_complete"]


def test_rejects_incompatible_reason_and_future_reply(tmp_path: Path) -> None:
    path = _sheet(
        tmp_path / "annotations.csv",
        intent_label="delivery_or_collection",
        handling_label="AUTO_HANDLE",
        reason_code="MONEY_OR_COMMITMENT",
        annotator_id="annotator_a",
    )
    with pytest.raises(ValueError, match="non-automatic reason"):
        validate_annotations(path)

    frame = pd.read_csv(path)
    frame["historical_reply"] = "Future information"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="must not expose historical_reply"):
        validate_annotations(path)


def test_initializes_independent_stratified_annotation_rounds(tmp_path: Path) -> None:
    rows = []
    for index in range(4):
        rows.append(
            {
                "display_order": index + 1,
                "case_id": f"tesco-{index}",
                "sample_slice": "natural" if index < 3 else "challenge",
                "conversation_id": index,
                "tweet_id": index,
                "created_at": "Wed Nov 01 09:00:00 +0000 2017",
                "message": f"message {index}",
                "prior_context": "",
                **{column: "" for column in LABEL_COLUMNS},
            }
        )
    candidates = tmp_path / "candidates.csv"
    pd.DataFrame(rows).to_csv(candidates, index=False)
    round_one = tmp_path / "round1.csv"
    round_two = tmp_path / "round2.csv"

    manifest = initialize_annotation_rounds(
        candidates,
        round_one,
        round_two,
        tmp_path / "manifest.json",
        second_annotation_count=3,
        seed=9,
    )

    round_one_frame = pd.read_csv(round_one)
    round_two_frame = pd.read_csv(round_two)
    assert len(round_one_frame) == 4
    assert len(round_two_frame) == 3
    assert "sample_slice" not in round_one_frame
    assert "sample_slice" not in round_two_frame
    assert "challenge_flags" not in round_one_frame
    assert manifest["round_one_sha256"] != manifest["candidate_sha256"]
    assert manifest["round_two_slice_counts"] == {"natural": 2, "challenge": 1}
    assert manifest["label_status"] == "UNLABELED"


def test_require_blind_rejects_sampling_metadata(tmp_path: Path) -> None:
    path = _sheet(tmp_path / "annotations.csv")
    frame = pd.read_csv(path)
    frame["challenge_flags"] = "money"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="exposes sampling metadata"):
        validate_annotations(path, require_blind=True)


def test_freezes_independent_labels_and_rejects_context_edits(tmp_path: Path) -> None:
    rows = []
    for index in range(4):
        rows.append(
            {
                "display_order": index + 1,
                "case_id": f"tesco-{index}",
                "sample_slice": "natural" if index < 3 else "challenge",
                "conversation_id": index,
                "tweet_id": index,
                "created_at": "Wed Nov 01 09:00:00 +0000 2017",
                "message": f"message {index}",
                "prior_context": "",
                "challenge_flags": "",
                **{column: "" for column in LABEL_COLUMNS},
            }
        )
    candidates = tmp_path / "candidates.csv"
    pd.DataFrame(rows).to_csv(candidates, index=False)
    round_one = tmp_path / "round1.csv"
    round_two = tmp_path / "round2.csv"
    manifest = tmp_path / "manifest.json"
    initialize_annotation_rounds(
        candidates,
        round_one,
        round_two,
        manifest,
        second_annotation_count=3,
        seed=9,
    )

    for path, annotator in ((round_one, "human_a"), (round_two, "human_b")):
        frame = pd.read_csv(path, dtype="string", keep_default_na=False)
        frame["intent_label"] = "feedback_praise_or_suggestion"
        frame["handling_label"] = "AUTO_HANDLE"
        frame["reason_code"] = "NO_ACTION_NEEDED"
        frame["annotator_id"] = annotator
        frame.to_csv(path, index=False, lineterminator="\n")

    frozen = freeze_human_annotations(
        candidates,
        round_one,
        round_two,
        manifest,
        CODEBOOK,
    )
    assert frozen["label_status"] == "HUMAN_LABELED_UNADJUDICATED"
    assert frozen["round_one_annotator_ids"] == ["human_a"]
    assert frozen["round_two_annotator_ids"] == ["human_b"]

    changed = pd.read_csv(round_two)
    changed.loc[0, "message"] = "edited after sampling"
    changed.to_csv(round_two, index=False)
    with pytest.raises(ValueError, match="changed frozen message/context"):
        freeze_human_annotations(
            candidates,
            round_one,
            round_two,
            manifest,
            CODEBOOK,
        )
