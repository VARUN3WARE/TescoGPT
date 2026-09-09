from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from tescogpt.evaluation.adjudication import (
    finalize_adjudication,
    initialize_adjudication,
    validate_adjudication,
)
from tescogpt.evaluation.labels import LABEL_COLUMNS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(
    index: int,
    *,
    intent: str,
    handling: str,
    reason: str,
    annotator: str,
) -> dict[str, str | int]:
    return {
        "display_order": index + 1,
        "case_id": f"case-{index}",
        "conversation_id": f"conversation-{index}",
        "tweet_id": str(index),
        "created_at": "2017-10-01 00:00:00",
        "message": f"message {index}",
        "prior_context": "",
        **{column: "" for column in LABEL_COLUMNS},
        "intent_label": intent,
        "handling_label": handling,
        "reason_code": reason,
        "must_include": "Give an appropriate next step.",
        "must_avoid": "Do not claim an action was completed.",
        "annotator_id": annotator,
    }


def _rounds(tmp_path: Path) -> tuple[Path, Path, Path]:
    first = tmp_path / "round1.csv"
    second = tmp_path / "round2.csv"
    first_rows = [
        _row(
            0,
            intent="feedback_praise_or_suggestion",
            handling="AUTO_HANDLE",
            reason="NO_ACTION_NEEDED",
            annotator="human_a",
        ),
        _row(
            1,
            intent="product_availability",
            handling="AUTO_HANDLE",
            reason="SAFE_CLARIFICATION",
            annotator="human_a",
        ),
        _row(
            2,
            intent="delivery_or_collection",
            handling="ESCALATE",
            reason="ACCOUNT_OR_ORDER_LOOKUP",
            annotator="human_a",
        ),
    ]
    second_rows = [
        _row(
            0,
            intent="feedback_praise_or_suggestion",
            handling="AUTO_HANDLE",
            reason="NO_ACTION_NEEDED",
            annotator="human_b",
        ),
        _row(
            1,
            intent="product_information",
            handling="ESCALATE",
            reason="HUMAN_JUDGMENT_REQUIRED",
            annotator="human_b",
        ),
    ]
    pd.DataFrame(first_rows).to_csv(first, index=False)
    pd.DataFrame(second_rows).to_csv(second, index=False)
    manifest = tmp_path / "annotation_rounds.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "label_status": "HUMAN_LABELED_UNADJUDICATED",
                "round_one_sha256": _sha256(first),
                "round_two_sha256": _sha256(second),
            }
        ),
        encoding="utf-8",
    )
    return first, second, manifest


def test_adjudication_prefills_agreements_and_builds_distinct_final_gold(
    tmp_path: Path,
) -> None:
    first, second, round_manifest = _rounds(tmp_path)
    adjudication = tmp_path / "adjudication.csv"
    adjudication_manifest = tmp_path / "adjudication.manifest.json"

    manifest = initialize_adjudication(
        first,
        second,
        round_manifest,
        adjudication,
        adjudication_manifest,
    )
    frame = pd.read_csv(adjudication, dtype="string", keep_default_na=False)

    assert manifest["overlap_count"] == 2
    assert manifest["disagreement_count"] == 1
    assert frame.loc[frame["case_id"].eq("case-0"), "final_intent_label"].iloc[0]
    assert frame.loc[frame["case_id"].eq("case-1"), "final_intent_label"].iloc[0] == ""
    assert validate_adjudication(adjudication)["remaining_count"] == 1

    disagreement = frame["case_id"].eq("case-1")
    frame.loc[disagreement, "final_intent_label"] = "product_information"
    frame.loc[disagreement, "final_handling_label"] = "ESCALATE"
    frame.loc[disagreement, "final_reason_code"] = "HUMAN_JUDGMENT_REQUIRED"
    frame.loc[disagreement, "adjudicator_id"] = "human_c"
    frame.loc[disagreement, "adjudication_notes"] = "The customer asks for product facts."
    frame.to_csv(adjudication, index=False)

    final_path = tmp_path / "final_annotations.csv"
    final_manifest = finalize_adjudication(
        first,
        second,
        round_manifest,
        adjudication,
        adjudication_manifest,
        final_path,
    )
    final = pd.read_csv(final_path, dtype="string", keep_default_na=False).set_index(
        "case_id"
    )

    assert final_manifest["label_status"] == "HUMAN_LABELED_ADJUDICATED"
    assert final.loc["case-1", "intent_label"] == "product_information"
    assert final.loc["case-1", "label_provenance"] == "adjudicated"
    assert final.loc["case-0", "label_provenance"] == "independent_agreement"
    assert final.loc["case-2", "label_provenance"] == "round_one"
    updated_round_manifest = json.loads(round_manifest.read_text(encoding="utf-8"))
    assert updated_round_manifest["final_sha256"] == _sha256(final_path)


def test_unresolved_disagreement_requires_adjudicator(tmp_path: Path) -> None:
    first, second, round_manifest = _rounds(tmp_path)
    adjudication = tmp_path / "adjudication.csv"
    initialize_adjudication(
        first,
        second,
        round_manifest,
        adjudication,
        tmp_path / "adjudication.manifest.json",
    )
    frame = pd.read_csv(adjudication, dtype="string", keep_default_na=False)
    disagreement = frame["case_id"].eq("case-1")
    frame.loc[disagreement, "final_intent_label"] = "product_information"
    frame.loc[disagreement, "final_handling_label"] = "ESCALATE"
    frame.loc[disagreement, "final_reason_code"] = "HUMAN_JUDGMENT_REQUIRED"
    frame.to_csv(adjudication, index=False)

    with pytest.raises(ValueError, match="unresolved"):
        validate_adjudication(adjudication, require_complete=True)


def test_complete_adjudication_requires_notes_and_one_adjudicator(
    tmp_path: Path,
) -> None:
    first, second, round_manifest = _rounds(tmp_path)
    adjudication = tmp_path / "adjudication.csv"
    initialize_adjudication(
        first,
        second,
        round_manifest,
        adjudication,
        tmp_path / "adjudication.manifest.json",
    )
    frame = pd.read_csv(adjudication, dtype="string", keep_default_na=False)
    disagreement = frame["case_id"].eq("case-1")
    frame.loc[disagreement, "final_intent_label"] = "product_information"
    frame.loc[disagreement, "final_handling_label"] = "ESCALATE"
    frame.loc[disagreement, "final_reason_code"] = "HUMAN_JUDGMENT_REQUIRED"
    frame.loc[disagreement, "adjudicator_id"] = "human_c"
    frame.to_csv(adjudication, index=False)

    with pytest.raises(ValueError, match="lack adjudication notes"):
        validate_adjudication(adjudication, require_complete=True)

    frame.loc[disagreement, "adjudication_notes"] = "The customer asks for facts."
    second_disagreement = frame.loc[disagreement].copy()
    second_disagreement["case_id"] = "case-extra"
    second_disagreement["conversation_id"] = "conversation-extra"
    second_disagreement["tweet_id"] = "tweet-extra"
    second_disagreement["adjudicator_id"] = "human_d"
    pd.concat([frame, second_disagreement], ignore_index=True).to_csv(
        adjudication, index=False
    )
    with pytest.raises(ValueError, match="exactly one adjudicator ID"):
        validate_adjudication(adjudication, require_complete=True)


def test_adjudication_cannot_change_an_independent_agreement(tmp_path: Path) -> None:
    first, second, round_manifest = _rounds(tmp_path)
    adjudication = tmp_path / "adjudication.csv"
    initialize_adjudication(
        first,
        second,
        round_manifest,
        adjudication,
        tmp_path / "adjudication.manifest.json",
    )
    frame = pd.read_csv(adjudication, dtype="string", keep_default_na=False)
    agreement = frame["case_id"].eq("case-0")
    frame.loc[agreement, "final_intent_label"] = "other_or_unclear"
    frame.to_csv(adjudication, index=False)

    with pytest.raises(ValueError, match="agreed intent_label values changed"):
        validate_adjudication(adjudication)


def test_finalization_rejects_changed_adjudication_context(tmp_path: Path) -> None:
    first, second, round_manifest = _rounds(tmp_path)
    adjudication = tmp_path / "adjudication.csv"
    adjudication_manifest = tmp_path / "adjudication.manifest.json"
    initialize_adjudication(
        first,
        second,
        round_manifest,
        adjudication,
        adjudication_manifest,
    )
    frame = pd.read_csv(adjudication, dtype="string", keep_default_na=False)
    disagreement = frame["case_id"].eq("case-1")
    frame.loc[disagreement, "message"] = "changed message"
    frame.loc[disagreement, "final_intent_label"] = "product_information"
    frame.loc[disagreement, "final_handling_label"] = "ESCALATE"
    frame.loc[disagreement, "final_reason_code"] = "HUMAN_JUDGMENT_REQUIRED"
    frame.loc[disagreement, "adjudicator_id"] = "human_c"
    frame.loc[disagreement, "adjudication_notes"] = "The customer asks for facts."
    frame.to_csv(adjudication, index=False)

    with pytest.raises(ValueError, match="source labels or context changed"):
        finalize_adjudication(
            first,
            second,
            round_manifest,
            adjudication,
            adjudication_manifest,
            tmp_path / "final_annotations.csv",
        )
