"""Offline artifact verification and final result reproduction."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from tescogpt.evaluation.agreement import annotation_agreement
from tescogpt.evaluation.judge import judge_human_agreement
from tescogpt.evaluation.labels import validate_annotations
from tescogpt.evaluation.metrics import evaluate_predictions
from tescogpt.evaluation.retrieval_review import (
    evaluate_retrieval_review,
    validate_retrieval_review,
)
from tescogpt.evaluation.safety import audit_prediction_safety


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _hash_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    path: Path,
    expected: str | None,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact does not exist: {path}")
    actual = _sha256(path)
    passed = expected is not None and actual == expected
    checks.append(
        {
            "name": name,
            "file": path.name,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "passed": passed,
        }
    )
    if not passed:
        raise ValueError(f"Artifact hash mismatch for {name}: {path}")


def verify_artifact_integrity(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    """Verify distributed inputs/outputs against their adjacent manifests."""
    checks: list[dict[str, Any]] = []
    registry_path = _resolve(root, config["registry_file"])
    registry_manifest = json.loads(
        _resolve(root, config["registry_manifest"]).read_text(encoding="utf-8")
    )
    _hash_check(
        checks,
        name="golden candidate registry",
        path=registry_path,
        expected=registry_manifest.get("candidate_sha256"),
    )
    _hash_check(
        checks,
        name="annotation codebook",
        path=_resolve(root, config["codebook_file"]),
        expected=registry_manifest.get("codebook_sha256"),
    )

    annotation_manifest = json.loads(
        _resolve(root, config["annotation_manifest"]).read_text(encoding="utf-8")
    )
    _hash_check(
        checks,
        name="round one annotations",
        path=_resolve(root, config["gold_file"]),
        expected=annotation_manifest.get("round_one_sha256"),
    )
    _hash_check(
        checks,
        name="round two annotations",
        path=_resolve(root, config["second_annotation_file"]),
        expected=annotation_manifest.get("round_two_sha256"),
    )

    registry_hash = _sha256(registry_path)
    for prediction_value in config["prediction_files"]:
        prediction_path = _resolve(root, prediction_value)
        manifest_path = prediction_path.with_suffix(
            prediction_path.suffix + ".manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _hash_check(
            checks,
            name=f"prediction {manifest.get('system', prediction_path.stem)}",
            path=prediction_path,
            expected=manifest.get("prediction_sha256"),
        )
        if manifest.get("input_sha256") != registry_hash:
            raise ValueError(f"Prediction input hash does not match registry: {prediction_path}")
        frame = pd.read_csv(prediction_path, dtype="string", keep_default_na=False)
        if set(frame["system_name"]) != {manifest.get("system")}:
            raise ValueError(f"Prediction system name differs from manifest: {prediction_path}")

    retrieval_path = _resolve(root, config["retrieval_file"])
    retrieval_manifest_path = retrieval_path.with_suffix(
        retrieval_path.suffix + ".manifest.json"
    )
    retrieval_manifest = json.loads(
        retrieval_manifest_path.read_text(encoding="utf-8")
    )
    _hash_check(
        checks,
        name="outcome-aware retrieval",
        path=retrieval_path,
        expected=retrieval_manifest.get("output_sha256"),
    )
    if retrieval_manifest.get("input_sha256") != registry_hash:
        raise ValueError("Retrieval input hash does not match the frozen registry")

    relevance_manifest_path = _resolve(root, config["retrieval_review_manifest"])
    relevance_manifest = json.loads(relevance_manifest_path.read_text(encoding="utf-8"))
    relevance_review_path = _resolve(root, config["retrieval_review_file"])
    relevance_key_path = _resolve(root, config["retrieval_review_key"])
    _hash_check(
        checks,
        name="retrieval relevance review",
        path=relevance_review_path,
        expected=relevance_manifest.get("review_sha256"),
    )
    _hash_check(
        checks,
        name="retrieval relevance identity key",
        path=relevance_key_path,
        expected=relevance_manifest.get("identity_key_sha256"),
    )
    if relevance_manifest.get("registry_sha256") != registry_hash:
        raise ValueError("Retrieval relevance review does not match the frozen registry")
    validate_retrieval_review(
        relevance_review_path,
        require_complete=relevance_manifest.get("label_status") == "HUMAN_LABELED",
    )
    return checks


def _percentage(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def write_headline_table(
    metrics: dict[str, Any],
    output_path: Path,
    headline_system: str,
) -> None:
    systems = metrics["systems"]
    if headline_system not in systems:
        raise ValueError(f"Configured headline system is absent: {headline_system}")
    lines = [
        "# Reproduced headline results",
        "",
        "All values below are computed from committed per-example predictions and human labels.",
        "",
        "| System | Slice | Intent macro-F1 | Coverage | Unsafe auto | 95% upper bound |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for system_name, system in systems.items():
        for slice_name in ("natural", "challenge", "all"):
            result = system["slices"][slice_name]
            intent = result["intent"]["macro_f1_observed_intents"]
            routing = result["routing"]
            display_name = f"**{system_name}**" if system_name == headline_system else system_name
            lines.append(
                f"| {display_name} | {slice_name} | {intent:.3f} | "
                f"{_percentage(routing['coverage'])} | "
                f"{_percentage(routing['unsafe_auto_rate_given_auto'])} | "
                f"{_percentage(routing['unsafe_auto_rate_wilson_upper_95'])} |"
            )
    lines.extend(
        [
            "",
            "Unsafe auto means the system chose AUTO_HANDLE where the human label was ESCALATE.",
            "The challenge slice is a stress test and is not a prevalence estimate.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def reproduce_project(
    config_path: str | Path,
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Verify artifacts and reproduce every available offline evaluation output."""
    started = time.perf_counter()
    config_file = Path(config_path).resolve()
    root = config_file.parent.parent
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("experiment_schema_version") != 1:
        raise ValueError("Unsupported experiment configuration schema")
    integrity_checks = verify_artifact_integrity(config, root)

    round_one = _resolve(root, config["gold_file"])
    round_two = _resolve(root, config["second_annotation_file"])
    round_one_progress = validate_annotations(round_one)
    round_two_progress = validate_annotations(round_two)
    prediction_paths = [_resolve(root, value) for value in config["prediction_files"]]
    safety = audit_prediction_safety(
        prediction_paths,
        _resolve(root, config["static_safety_output"]),
        _resolve(root, config["static_safety_details"]),
    )

    complete = round_one_progress["is_complete"] and round_two_progress["is_complete"]
    report: dict[str, Any] = {
        "reproduction_schema_version": 1,
        "configured_status": config["status"],
        "integrity_checks": integrity_checks,
        "annotation_progress": {
            "round_one": round_one_progress,
            "round_two": round_two_progress,
        },
        "static_safety_system_count": len(safety["systems"]),
    }
    relevance_manifest_path = _resolve(root, config["retrieval_review_manifest"])
    relevance_manifest = json.loads(relevance_manifest_path.read_text(encoding="utf-8"))
    report["retrieval_review_status"] = relevance_manifest["label_status"]
    if not complete:
        report["status"] = "AWAITING_HUMAN_LABELS"
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        destination = _resolve(root, config["reproduction_output"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if not allow_incomplete:
            raise RuntimeError(
                "Human annotations are incomplete; run with --allow-incomplete for readiness only"
            )
        return report

    annotation_manifest = json.loads(
        _resolve(root, config["annotation_manifest"]).read_text(encoding="utf-8")
    )
    if not str(annotation_manifest.get("label_status", "")).startswith("HUMAN_LABELED"):
        raise ValueError("Completed labels must be frozen with labels-freeze first")
    agreement = annotation_agreement(
        round_one,
        round_two,
        _resolve(root, config["annotation_agreement_output"]),
    )
    metrics = evaluate_predictions(
        round_one,
        _resolve(root, config["registry_file"]),
        prediction_paths,
        _resolve(root, config["metrics_output"]),
        bootstrap_iterations=int(config["bootstrap_iterations"]),
        seed=int(config["seed"]),
    )
    headline_system = config.get("headline_system")
    if not headline_system:
        raise ValueError("A completed experiment requires headline_system in the config")
    write_headline_table(
        metrics,
        _resolve(root, config["headline_output"]),
        headline_system,
    )
    report["annotation_agreement_overlap"] = agreement["overlap_count"]
    report["evaluated_system_count"] = metrics["system_count"]

    if relevance_manifest["label_status"] == "HUMAN_LABELED":
        retrieval_metrics = evaluate_retrieval_review(
            _resolve(root, config["retrieval_review_file"]),
            _resolve(root, config["retrieval_review_key"]),
            _resolve(root, config["retrieval_metrics_output"]),
            top_k=int(relevance_manifest["top_k"]),
            manifest_path=relevance_manifest_path,
        )
        report["retrieval_evaluated_system_count"] = len(retrieval_metrics["systems"])
    elif config["status"] == "FINAL":
        raise ValueError("FINAL config requires frozen human retrieval relevance ratings")

    human_review = config.get("reply_review_file")
    judge_outputs = config.get("judge_output_files", [])
    if human_review and judge_outputs:
        judge_report = judge_human_agreement(
            _resolve(root, human_review),
            [_resolve(root, value) for value in judge_outputs],
            _resolve(root, config["judge_agreement_output"]),
        )
        report["judge_comparison_count"] = len(judge_report["comparisons"])
    elif config["status"] == "FINAL":
        raise ValueError("FINAL config requires human reply ratings and judge outputs")

    elapsed = time.perf_counter() - started
    report["elapsed_seconds"] = round(elapsed, 3)
    report["under_15_minutes"] = elapsed < 900
    report["status"] = "COMPLETE" if config["status"] == "FINAL" else "DEVELOPMENT_COMPLETE"
    destination = _resolve(root, config["reproduction_output"])
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if elapsed >= 900:
        raise RuntimeError("Offline reproduction exceeded the 15-minute contract")
    return report
