from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tescogpt.evaluation.labels import LABEL_COLUMNS
from tescogpt.evaluation.status import _judge_replicates_ready, project_status


def _blank_annotations(path: Path, count: int) -> None:
    rows = []
    for index in range(count):
        rows.append(
            {
                "display_order": index + 1,
                "case_id": f"case-{index}",
                "conversation_id": f"conversation-{index}",
                "tweet_id": str(index),
                "message": f"message {index}",
                **{column: "" for column in LABEL_COLUMNS},
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _config(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    config_dir = project / "config"
    data_dir = project / "data" / "golden"
    config_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    _blank_annotations(data_dir / "round1.csv", 2)
    _blank_annotations(data_dir / "round2.csv", 1)
    (data_dir / "manifest.json").write_text(
        json.dumps({"label_status": "UNLABELED"}),
        encoding="utf-8",
    )
    config_path = config_dir / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "status": "AWAITING_HUMAN_LABELS",
                "gold_file": "data/golden/round1.csv",
                "round_one_annotation_file": "data/golden/round1.csv",
                "second_annotation_file": "data/golden/round2.csv",
                "annotation_manifest": "data/golden/manifest.json",
                "prediction_files": [],
                "status_output": "outputs/status.json",
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_status_reports_first_concrete_checkpoint_and_writes_artifact(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    report = project_status(config)

    assert not report["ready_for_final_reproduction"]
    assert report["next_action"]["gate"] == "primary_annotations_complete"
    assert report["human_progress"]["round_one"]["remaining_count"] == 2
    assert (config.parent.parent / "outputs" / "status.json").is_file()


def test_status_can_fail_as_a_final_preflight(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with pytest.raises(RuntimeError, match="primary_annotations_complete"):
        project_status(config, require_final_ready=True)


def test_frozen_annotation_status_requires_matching_hashes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifest_path = config.parent.parent / "data" / "golden" / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "label_status": "HUMAN_LABELED_UNADJUDICATED",
                "round_one_sha256": "wrong",
                "round_two_sha256": "wrong",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Round-one labels"):
        project_status(config)


def test_judge_gate_requires_distinct_replicates_of_one_model() -> None:
    assert not _judge_replicates_ready(
        [
            {"model": "judge-a", "replicate": 1},
            {"model": "judge-a", "replicate": 1},
        ]
    )
    assert not _judge_replicates_ready(
        [
            {"model": "judge-a", "replicate": 1},
            {"model": "judge-b", "replicate": 2},
        ]
    )
    assert _judge_replicates_ready(
        [
            {"model": "judge-a", "replicate": 1},
            {"model": "judge-a", "replicate": 2},
        ]
    )
