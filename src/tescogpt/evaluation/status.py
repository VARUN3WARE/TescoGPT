"""Deterministic progress report and final-experiment preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from tescogpt.evaluation.adjudication import validate_adjudication
from tescogpt.evaluation.judge import validate_judge_output
from tescogpt.evaluation.labels import validate_annotations
from tescogpt.evaluation.reply_review import validate_reply_ratings
from tescogpt.evaluation.report import validate_submission_report
from tescogpt.evaluation.reproduce import (
    _validate_generation_provenance,
    verify_artifact_integrity,
)
from tescogpt.evaluation.retrieval_review import validate_retrieval_review
from tescogpt.evaluation.submission import validate_submission_package


def _resolve(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _missing_progress(file_name: str) -> dict[str, Any]:
    return {
        "file": file_name,
        "row_count": 0,
        "completed_count": 0,
        "remaining_count": None,
        "is_complete": False,
        "status": "NOT_STARTED",
    }


def _require_frozen_hash(
    path: Path,
    manifest: dict[str, Any],
    field: str,
    label: str,
) -> None:
    if manifest.get(field) != _sha256(path):
        raise ValueError(f"{label} differs from its frozen manifest")


def _prediction_inventory(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    inventory = []
    for value in config.get("prediction_files", []):
        path = _resolve(root, value)
        assert path is not None
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        if not path.is_file() or not manifest_path.is_file():
            inventory.append(
                {
                    "file": path.name,
                    "status": "MISSING",
                    "system": None,
                    "row_count": 0,
                    "api_backed": False,
                }
            )
            continue
        manifest = _read_manifest(manifest_path)
        frame = pd.read_csv(path, dtype="string", keep_default_na=False)
        if frame["case_id"].duplicated().any():
            raise ValueError(f"Prediction contains duplicate case IDs: {path}")
        systems = sorted(set(frame["system_name"].str.strip()))
        if len(systems) != 1 or systems[0] != manifest.get("system"):
            raise ValueError(f"Prediction system differs from manifest: {path}")
        if manifest.get("prediction_sha256") != _sha256(path):
            raise ValueError(f"Prediction hash differs from manifest: {path}")
        _validate_generation_provenance(manifest, frame, path)
        inventory.append(
            {
                "file": path.name,
                "status": "FROZEN",
                "system": systems[0],
                "row_count": len(frame),
                "api_backed": bool(manifest.get("model")),
                "model": manifest.get("model"),
            }
        )
    return inventory


def _judge_replicates_ready(judge_runs: list[dict[str, Any]]) -> bool:
    models = {str(item["model"]) for item in judge_runs}
    replicates = {int(item["replicate"]) for item in judge_runs}
    return len(judge_runs) >= 2 and len(models) == 1 and len(replicates) >= 2


def project_status(
    config_path: str | Path,
    output_path: str | Path | None = None,
    *,
    require_final_ready: bool = False,
) -> dict[str, Any]:
    """Report human progress and enforce the prerequisites for the final run."""
    config_file = Path(config_path).resolve()
    root = config_file.parent.parent
    config = json.loads(config_file.read_text(encoding="utf-8"))
    submission_package = validate_submission_package(root, config)

    report_path = _resolve(root, config.get("report_file"))
    if report_path is not None and report_path.is_file():
        report_status = validate_submission_report(
            report_path,
            max_words=int(config.get("report_max_words", 2400)),
            candidate_registry_path=_resolve(root, config.get("registry_file")),
        )
    else:
        report_status = {
            "file": report_path.name if report_path is not None else "REPORT.md",
            "declared_status": "MISSING",
            "word_count": 0,
            "max_words": int(config.get("report_max_words", 2400)),
            "required_section_count": 0,
            "verified_failure_mode_count": 0,
            "is_submission_ready": False,
        }

    round_one_path = _resolve(
        root, config.get("round_one_annotation_file", config.get("gold_file"))
    )
    round_two_path = _resolve(root, config.get("second_annotation_file"))
    if round_one_path is None or round_two_path is None:
        raise ValueError("Status config requires both independent annotation files")
    round_one = validate_annotations(round_one_path)
    round_two = validate_annotations(round_two_path)
    annotation_manifest = _read_manifest(_resolve(root, config.get("annotation_manifest")))
    annotation_status = str(annotation_manifest.get("label_status", "MISSING"))
    if annotation_status in {
        "HUMAN_LABELED_UNADJUDICATED",
        "HUMAN_LABELED_ADJUDICATED",
    }:
        _require_frozen_hash(
            round_one_path,
            annotation_manifest,
            "round_one_sha256",
            "Round-one labels",
        )
        _require_frozen_hash(
            round_two_path,
            annotation_manifest,
            "round_two_sha256",
            "Round-two labels",
        )

    adjudication_path = _resolve(
        root, config.get("adjudication_file", "data/golden/adjudication.csv")
    )
    if adjudication_path is not None and adjudication_path.is_file():
        adjudication = validate_adjudication(adjudication_path)
        adjudication["status"] = (
            "COMPLETE" if adjudication["is_complete"] else "IN_PROGRESS"
        )
    else:
        adjudication = _missing_progress("adjudication.csv")

    gold_path = _resolve(root, config.get("gold_file"))
    gold_is_distinct = bool(
        gold_path is not None and gold_path != round_one_path and gold_path.is_file()
    )
    gold_progress = (
        validate_annotations(gold_path, require_blind=True) if gold_is_distinct else None
    )
    if annotation_status == "HUMAN_LABELED_ADJUDICATED" and gold_is_distinct:
        assert gold_path is not None
        _require_frozen_hash(
            gold_path,
            annotation_manifest,
            "final_sha256",
            "Final gold labels",
        )

    retrieval_path = _resolve(root, config.get("retrieval_review_file"))
    retrieval_manifest = _read_manifest(
        _resolve(root, config.get("retrieval_review_manifest"))
    )
    if retrieval_path is not None and retrieval_path.is_file():
        retrieval = validate_retrieval_review(retrieval_path)
        if retrieval_manifest.get("label_status") == "HUMAN_LABELED":
            _require_frozen_hash(
                retrieval_path,
                retrieval_manifest,
                "review_sha256",
                "Retrieval review",
            )
            retrieval_key = _resolve(root, config.get("retrieval_review_key"))
            if retrieval_key is None or not retrieval_key.is_file():
                raise ValueError("Frozen retrieval review lacks its identity key")
            _require_frozen_hash(
                retrieval_key,
                retrieval_manifest,
                "identity_key_sha256",
                "Retrieval identity key",
            )
        retrieval["status"] = (
            "FROZEN"
            if retrieval_manifest.get("label_status") == "HUMAN_LABELED"
            else "COMPLETE_UNFROZEN"
            if retrieval["is_complete"]
            else "IN_PROGRESS"
        )
    else:
        retrieval = _missing_progress("retrieval_relevance.csv")

    predictions = _prediction_inventory(config, root)
    frozen_predictions = [item for item in predictions if item["status"] == "FROZEN"]
    api_systems = {
        str(item["system"])
        for item in frozen_predictions
        if item["api_backed"] and item["row_count"] == round_one["row_count"]
    }
    configured_systems = {str(item["system"]) for item in frozen_predictions}
    headline_system = config.get("headline_system")

    reply_path = _resolve(root, config.get("reply_review_file"))
    reply_manifest = _read_manifest(_resolve(root, config.get("reply_review_manifest")))
    if reply_path is not None and reply_path.is_file():
        reply_review = validate_reply_ratings(reply_path)
        if reply_manifest.get("label_status") == "HUMAN_RATED":
            _require_frozen_hash(
                reply_path,
                reply_manifest,
                "review_sha256",
                "Reply review",
            )
            reply_key = _resolve(root, config.get("reply_review_key"))
            if reply_key is None or not reply_key.is_file():
                raise ValueError("Frozen reply review lacks its identity key")
            _require_frozen_hash(
                reply_key,
                reply_manifest,
                "identity_key_sha256",
                "Reply-review identity key",
            )
        reply_review["status"] = (
            "FROZEN"
            if reply_manifest.get("label_status") == "HUMAN_RATED"
            else "COMPLETE_UNFROZEN"
            if reply_review["is_complete"]
            else "IN_PROGRESS"
        )
    else:
        reply_review = _missing_progress("reply_review.csv")

    judge_runs = []
    for value in config.get("judge_output_files", []):
        path = _resolve(root, value)
        assert path is not None
        judge_runs.append(
            validate_judge_output(
                path,
                review_path=reply_path,
                manifest_path=path.with_suffix(path.suffix + ".manifest.json"),
            )
        )

    required_baselines = {"trivial_constant_v1", "simple_rules_bm25_v1"}

    gates = {
        "primary_annotations_complete": bool(round_one["is_complete"]),
        "independent_annotations_complete": bool(round_two["is_complete"]),
        "independent_rounds_frozen": annotation_status
        in {"HUMAN_LABELED_UNADJUDICATED", "HUMAN_LABELED_ADJUDICATED"},
        "categorical_adjudication_complete": bool(
            annotation_status == "HUMAN_LABELED_ADJUDICATED"
            and adjudication["is_complete"]
        ),
        "distinct_final_gold_configured": bool(
            gold_is_distinct and gold_progress and gold_progress["is_complete"]
        ),
        "retrieval_review_frozen": bool(
            retrieval["is_complete"]
            and retrieval_manifest.get("label_status") == "HUMAN_LABELED"
        ),
        "required_baselines_frozen": required_baselines.issubset(configured_systems),
        "three_systems_frozen": len(configured_systems) >= 3,
        "api_main_prediction_frozen": bool(api_systems),
        "headline_system_is_api_main": headline_system in api_systems,
        "reply_review_frozen": bool(
            reply_review["is_complete"]
            and reply_manifest.get("label_status") == "HUMAN_RATED"
        ),
        "two_judge_replicates_frozen": _judge_replicates_ready(judge_runs),
        "submission_package_complete": bool(submission_package["is_complete"]),
        "report_submission_ready": bool(report_status["is_submission_ready"]),
        "config_marked_final": config.get("status") == "FINAL",
    }
    actions = {
        "primary_annotations_complete": (
            "Complete round1_annotation.xlsx, then run labels-import-workbook "
            "with --require-complete for data/golden/round1_annotations.csv."
        ),
        "independent_annotations_complete": (
            "Have a different person complete round2_annotation.xlsx, then import it "
            "with labels-import-workbook --require-complete."
        ),
        "independent_rounds_frozen": "Run: python -m tescogpt labels-freeze",
        "categorical_adjudication_complete": (
            "Run annotation-agreement, adjudication-init, human resolution, and "
            "adjudication-finalize."
        ),
        "distinct_final_gold_configured": (
            "Point gold_file at data/golden/final_annotations.csv."
        ),
        "retrieval_review_frozen": (
            "Complete the blind workbook, run retrieval-review-import-workbook "
            "--require-complete, then retrieval-review-freeze."
        ),
        "required_baselines_frozen": "Freeze the specified trivial and simple baselines.",
        "three_systems_frozen": "Freeze two baselines and one main-system prediction file.",
        "api_main_prediction_frozen": (
            "Run main-openai over all 200 cases with an explicit model ID."
        ),
        "headline_system_is_api_main": (
            "Set headline_system to the frozen API-backed system ID."
        ),
        "reply_review_frozen": (
            "Create, complete, and freeze the system-blinded reply-quality review."
        ),
        "two_judge_replicates_frozen": (
            "Run and freeze at least two judge replicates over the same human review."
        ),
        "submission_package_complete": (
            "Restore the README, 10–15 decision log, citations, and 150–250-case registry."
        ),
        "report_submission_ready": (
            "Replace pending results and change REPORT.md to SUBMISSION_STATUS: READY."
        ),
        "config_marked_final": (
            "Finish the report, replace pending settings, and set config status to FINAL."
        ),
    }
    blockers = [
        {"gate": gate, "next_action": actions[gate]}
        for gate, passed in gates.items()
        if not passed
    ]
    final_integrity_checks = verify_artifact_integrity(config, root) if not blockers else []
    report = {
        "project_status_schema_version": 1,
        "configured_status": config.get("status"),
        "ready_for_final_reproduction": not blockers,
        "next_action": blockers[0] if blockers else None,
        "blockers": blockers,
        "gates": gates,
        "human_progress": {
            "round_one": round_one,
            "round_two": round_two,
            "adjudication": adjudication,
            "retrieval_review": retrieval,
            "reply_review": reply_review,
        },
        "annotation_manifest_status": annotation_status,
        "predictions": predictions,
        "api_systems": sorted(api_systems),
        "headline_system": headline_system,
        "judge_runs": judge_runs,
        "submission_package": submission_package,
        "report": report_status,
        "final_integrity_check_count": len(final_integrity_checks),
    }
    destination_value = output_path or config.get("status_output")
    destination = (
        _resolve(root, str(destination_value)) if destination_value is not None else None
    )
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if require_final_ready and blockers:
        raise RuntimeError(
            f"Final experiment is not ready: {blockers[0]['gate']}"
        )
    return report
