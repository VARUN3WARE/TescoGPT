"""Traceable per-example failure evidence across unequal evaluation samples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from tescogpt.evaluation.judge import validate_judge_output
from tescogpt.evaluation.labels import validate_annotations
from tescogpt.evaluation.reply_review import (
    RATING_DIMENSIONS,
    validate_reply_ratings,
    validate_reply_review_key,
)
from tescogpt.evaluation.retrieval_review import (
    validate_retrieval_review,
    validate_retrieval_review_key,
)
from tescogpt.policy.safety import inspect_reply

EVENT_COLUMNS = (
    "event_id",
    "subsystem",
    "failure_mode",
    "severity",
    "system_name",
    "case_id",
    "sample_slice",
    "message",
    "draft_reply",
    "observed_evidence",
    "artifact_source",
)

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2}


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _append_event(
    events: list[dict[str, str]],
    *,
    subsystem: str,
    failure_mode: str,
    severity: str,
    system_name: str,
    case_id: str,
    sample_slice: str,
    message: str,
    draft_reply: str,
    observed_evidence: str,
    artifact_source: str,
) -> None:
    events.append(
        {
            "event_id": "",
            "subsystem": subsystem,
            "failure_mode": failure_mode,
            "severity": severity,
            "system_name": system_name,
            "case_id": case_id,
            "sample_slice": sample_slice,
            "message": message,
            "draft_reply": draft_reply,
            "observed_evidence": observed_evidence,
            "artifact_source": artifact_source,
        }
    )


def _core_prediction_events(
    gold: pd.DataFrame,
    registry: pd.DataFrame,
    prediction_paths: list[Path],
    events: list[dict[str, str]],
    denominators: list[dict[str, Any]],
) -> None:
    slice_map = registry.set_index("case_id")["sample_slice"]
    labelled = gold.copy()
    labelled["sample_slice"] = labelled["case_id"].map(slice_map)
    if labelled["sample_slice"].isna().any():
        raise ValueError("Every gold case must exist in the sampling registry")
    required = {
        "case_id",
        "system_name",
        "predicted_intent",
        "intent_confidence",
        "draft_reply",
        "handling_decision",
        "decision_reason",
    }
    for path in prediction_paths:
        predictions = pd.read_csv(path, dtype="string", keep_default_na=False)
        missing = sorted(required - set(predictions.columns))
        if missing:
            raise ValueError(f"Prediction is missing columns: {', '.join(missing)}")
        if predictions["case_id"].duplicated().any():
            raise ValueError(f"Prediction contains duplicate case IDs: {path}")
        if set(predictions["case_id"]) != set(labelled["case_id"]):
            raise ValueError(f"Prediction IDs do not exactly match gold: {path}")
        proposal_columns = {"proposed_draft", "draft_was_replaced"}
        present_proposal_columns = proposal_columns & set(predictions.columns)
        if present_proposal_columns and present_proposal_columns != proposal_columns:
            raise ValueError("Prediction must contain both proposal-audit columns")
        has_proposal_audit = present_proposal_columns == proposal_columns
        if has_proposal_audit:
            replacement_values = (
                predictions["draft_was_replaced"].str.strip().str.lower()
            )
            if not replacement_values.isin({"true", "false"}).all():
                raise ValueError("draft_was_replaced must contain only true or false")
        systems = sorted(set(predictions["system_name"]))
        if len(systems) != 1:
            raise ValueError(f"Prediction must contain one system: {path}")
        system = systems[0]
        predictions["intent_confidence"] = pd.to_numeric(
            predictions["intent_confidence"], errors="raise"
        )
        merged = labelled.merge(predictions, on="case_id", validate="one_to_one")
        denominators.append(
            {
                "subsystem": "core_agent",
                "system_name": system,
                "artifact_source": path.name,
                "unit": "gold_cases",
                "count": len(merged),
            }
        )
        for row in merged.to_dict(orient="records"):
            common = {
                "system_name": system,
                "case_id": str(row["case_id"]),
                "sample_slice": str(row["sample_slice"]),
                "message": str(row["message"]),
                "draft_reply": str(row["draft_reply"]),
                "artifact_source": path.name,
            }
            if has_proposal_audit and str(row["draft_was_replaced"]).lower() == "true":
                proposed_draft = str(row["proposed_draft"])
                proposal_flags = inspect_reply(proposed_draft)
                _append_event(
                    events,
                    subsystem="draft_safety",
                    failure_mode="guardrail_blocked_proposal",
                    severity="high",
                    observed_evidence=(
                        "blocked_flags="
                        + (";".join(proposal_flags) or "unclassified")
                        + f"; public_replacement={row['draft_reply']}"
                    ),
                    **{**common, "draft_reply": proposed_draft},
                )
            if row["intent_label"] != row["predicted_intent"]:
                _append_event(
                    events,
                    subsystem="intent",
                    failure_mode="intent_misclassification",
                    severity="medium",
                    observed_evidence=(
                        f"gold={row['intent_label']}; predicted={row['predicted_intent']}; "
                        f"confidence={float(row['intent_confidence']):.3f}"
                    ),
                    **common,
                )
            gold_auto = row["handling_label"] == "AUTO_HANDLE"
            predicted_auto = row["handling_decision"] == "AUTO_HANDLE"
            if predicted_auto and not gold_auto:
                _append_event(
                    events,
                    subsystem="routing",
                    failure_mode="unsafe_automation",
                    severity="critical",
                    observed_evidence=(
                        f"gold={row['handling_label']}/{row['reason_code']}; "
                        f"predicted={row['handling_decision']}/{row['decision_reason']}"
                    ),
                    **common,
                )
            if gold_auto and not predicted_auto:
                _append_event(
                    events,
                    subsystem="routing",
                    failure_mode="needless_escalation",
                    severity="medium",
                    observed_evidence=(
                        f"gold={row['handling_label']}/{row['reason_code']}; "
                        f"predicted={row['handling_decision']}/{row['decision_reason']}"
                    ),
                    **common,
                )
            if row["reason_code"] != row["decision_reason"]:
                _append_event(
                    events,
                    subsystem="decision_reason",
                    failure_mode="decision_reason_mismatch",
                    severity="medium",
                    observed_evidence=(
                        f"gold={row['reason_code']}; predicted={row['decision_reason']}; "
                        f"route_correct={row['handling_label'] == row['handling_decision']}"
                    ),
                    **common,
                )
            safety_flags = inspect_reply(str(row["draft_reply"]))
            if safety_flags:
                _append_event(
                    events,
                    subsystem="draft_safety",
                    failure_mode="static_draft_warning",
                    severity="critical" if predicted_auto else "high",
                    observed_evidence="flags=" + ";".join(safety_flags),
                    **common,
                )


def _reply_rating_events(
    review_path: Path,
    key_path: Path,
    events: list[dict[str, str]],
    denominators: list[dict[str, Any]],
) -> None:
    validate_reply_ratings(review_path, require_complete=True)
    review = pd.read_csv(review_path, dtype="string", keep_default_na=False)
    key = pd.read_csv(key_path, dtype="string", keep_default_na=False)
    systems = validate_reply_review_key(review, key)
    merged = review.merge(key, on=["review_id", "case_id"], validate="one_to_one")
    compared = merged.loc[merged["row_role"].eq("compared")].copy()
    for dimension in RATING_DIMENSIONS:
        compared[dimension] = pd.to_numeric(compared[dimension], errors="raise")
    for system in systems:
        count = int(compared["system_name"].eq(system).sum())
        denominators.append(
            {
                "subsystem": "human_reply_review",
                "system_name": system,
                "artifact_source": review_path.name,
                "unit": "reviewed_replies",
                "count": count,
            }
        )
    failed = compared.loc[compared["overall_pass"].eq("FAIL")]
    for row in failed.to_dict(orient="records"):
        critical = str(row["critical_error_tags"]).strip()
        score_total = sum(int(row[dimension]) for dimension in RATING_DIMENSIONS)
        _append_event(
            events,
            subsystem="reply_quality",
            failure_mode="human_reply_failure",
            severity="critical" if critical else "high",
            system_name=str(row["system_name"]),
            case_id=str(row["case_id"]),
            sample_slice=str(row["sample_slice"]),
            message=str(row["message"]),
            draft_reply=str(row["draft_reply"]),
            observed_evidence=(
                f"rubric_total={score_total}/12; privacy_safety={row['privacy_safety']}; "
                f"routing_fit={row['routing_fit']}; critical_tags={critical or 'none'}"
            ),
            artifact_source=review_path.name,
        )


def _retrieval_events(
    review_path: Path,
    key_path: Path,
    events: list[dict[str, str]],
    denominators: list[dict[str, Any]],
    *,
    top_k: int,
) -> None:
    validate_retrieval_review(review_path, require_complete=True)
    review = pd.read_csv(review_path, dtype="string", keep_default_na=False)
    key = pd.read_csv(key_path, dtype="string", keep_default_na=False)
    validate_retrieval_review_key(review, key, top_k=top_k)
    merged = review.merge(
        key,
        on=["retrieval_review_id", "query_case_id"],
        validate="one_to_one",
    )
    for system, rank_column in (("bm25", "bm25_rank"), ("outcome", "outcome_rank")):
        rank_one = merged.loc[merged[rank_column].eq("1")]
        denominators.append(
            {
                "subsystem": "retrieval_relevance",
                "system_name": system,
                "artifact_source": review_path.name,
                "unit": "reviewed_queries",
                "count": len(rank_one),
            }
        )
        irrelevant = rank_one.loc[rank_one["relevance_grade"].eq("0")]
        for row in irrelevant.to_dict(orient="records"):
            _append_event(
                events,
                subsystem="retrieval",
                failure_mode="irrelevant_top1_precedent",
                severity="high",
                system_name=system,
                case_id=str(row["query_case_id"]),
                sample_slice=str(row["sample_slice"]),
                message=str(row["query_message"]),
                draft_reply=str(row["candidate_reply"]),
                observed_evidence=(
                    f"candidate_case_id={row['candidate_case_id']}; relevance_grade=0"
                ),
                artifact_source=review_path.name,
            )


def _judge_disagreement_events(
    human_path: Path,
    key_path: Path,
    judge_paths: list[Path],
    events: list[dict[str, str]],
    denominators: list[dict[str, Any]],
) -> None:
    human = pd.read_csv(human_path, dtype="string", keep_default_na=False)
    key = pd.read_csv(key_path, dtype="string", keep_default_na=False)
    validate_reply_review_key(human, key)
    compared_key = key.loc[key["row_role"].eq("compared")]
    human_compared = human.merge(
        compared_key,
        on=["review_id", "case_id"],
        validate="one_to_one",
    )
    for path in judge_paths:
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        validation = validate_judge_output(
            path,
            review_path=human_path,
            manifest_path=manifest_path if manifest_path.is_file() else None,
        )
        judge = pd.read_csv(path, dtype="string", keep_default_na=False)
        merged = human_compared.merge(
            judge,
            on="review_id",
            suffixes=("_human", "_judge"),
            validate="one_to_one",
        )
        for system_name, system_rows in merged.groupby("system_name", sort=True):
            denominators.append(
                {
                    "subsystem": "judge_validation",
                    "system_name": str(system_name),
                    "artifact_source": path.name,
                    "evaluator": (
                        f"{validation['model']}:replicate-{validation['replicate']}"
                    ),
                    "unit": "compared_review_rows",
                    "count": len(system_rows),
                }
            )
        for row in merged.to_dict(orient="records"):
            gaps = {
                dimension: abs(int(row[f"{dimension}_human"]) - int(row[f"{dimension}_judge"]))
                for dimension in RATING_DIMENSIONS
            }
            pass_disagrees = row["overall_pass_human"] != row["overall_pass_judge"]
            severe_gaps = sorted(name for name, gap in gaps.items() if gap >= 2)
            if not pass_disagrees and not severe_gaps:
                continue
            _append_event(
                events,
                subsystem="judge_validity",
                failure_mode="judge_human_disagreement",
                severity="high" if pass_disagrees else "medium",
                system_name=str(row["system_name"]),
                case_id=str(row["case_id"]),
                sample_slice=str(row["sample_slice"]),
                message=str(row["message"]),
                draft_reply=str(row["draft_reply"]),
                observed_evidence=(
                    f"judge_file={path.name}; human_pass={row['overall_pass_human']}; "
                    f"judge_pass={row['overall_pass_judge']}; "
                    f"two_point_gaps={';'.join(severe_gaps) or 'none'}"
                ),
                artifact_source=path.name,
            )


def build_failure_evidence(
    gold_path: str | Path,
    registry_path: str | Path,
    prediction_paths: list[str | Path],
    events_path: str | Path,
    summary_path: str | Path,
    *,
    reply_review_path: str | Path | None = None,
    reply_key_path: str | Path | None = None,
    retrieval_review_path: str | Path | None = None,
    retrieval_key_path: str | Path | None = None,
    retrieval_top_k: int = 3,
    judge_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Build one evidence ledger while preserving each subsystem's denominator."""
    if not prediction_paths:
        raise ValueError("Failure analysis requires at least one prediction file")
    if (reply_review_path is None) != (reply_key_path is None):
        raise ValueError("Reply review and reply key must be supplied together")
    if (retrieval_review_path is None) != (retrieval_key_path is None):
        raise ValueError("Retrieval review and retrieval key must be supplied together")
    if judge_paths and (reply_review_path is None or reply_key_path is None):
        raise ValueError("Judge failure analysis requires the human reply review and key")

    gold_file = Path(gold_path)
    registry_file = Path(registry_path)
    prediction_files = [Path(path) for path in prediction_paths]
    validate_annotations(gold_file, require_complete=True, require_blind=True)
    gold = pd.read_csv(gold_file, dtype="string", keep_default_na=False)
    registry = pd.read_csv(registry_file, dtype="string", keep_default_na=False)
    events: list[dict[str, str]] = []
    denominators: list[dict[str, Any]] = []
    _core_prediction_events(gold, registry, prediction_files, events, denominators)

    if reply_review_path is not None and reply_key_path is not None:
        _reply_rating_events(
            Path(reply_review_path), Path(reply_key_path), events, denominators
        )
    if retrieval_review_path is not None and retrieval_key_path is not None:
        _retrieval_events(
            Path(retrieval_review_path),
            Path(retrieval_key_path),
            events,
            denominators,
            top_k=retrieval_top_k,
        )
    judge_files = [Path(path) for path in judge_paths or []]
    if judge_files:
        assert reply_review_path is not None and reply_key_path is not None
        _judge_disagreement_events(
            Path(reply_review_path),
            Path(reply_key_path),
            judge_files,
            events,
            denominators,
        )

    frame = pd.DataFrame(events, columns=EVENT_COLUMNS)
    if not frame.empty:
        frame["_severity_order"] = frame["severity"].map(_SEVERITY_ORDER)
        frame = frame.sort_values(
            ["_severity_order", "subsystem", "failure_mode", "system_name", "case_id"],
            kind="stable",
        ).drop(columns="_severity_order")
        frame = frame.reset_index(drop=True)
        frame["event_id"] = [f"failure-{index:04d}" for index in range(1, len(frame) + 1)]
    event_file = Path(events_path)
    event_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(event_file, index=False, lineterminator="\n")

    if frame.empty:
        counts: list[dict[str, Any]] = []
    else:
        counts = (
            frame.groupby(
                ["subsystem", "failure_mode", "system_name", "artifact_source"],
                sort=True,
            )
            .agg(event_count=("event_id", "size"), unique_case_count=("case_id", "nunique"))
            .reset_index()
            .to_dict(orient="records")
        )
    input_files = [gold_file, registry_file, *prediction_files]
    for optional in (reply_review_path, reply_key_path, retrieval_review_path, retrieval_key_path):
        if optional is not None:
            input_files.append(Path(optional))
    input_files.extend(judge_files)
    report = {
        "failure_analysis_schema_version": 1,
        "event_count": len(frame),
        "counts": counts,
        "denominators": denominators,
        "events_file": event_file.name,
        "events_sha256": _sha256(event_file),
        "input_artifacts": [
            {"file": path.name, "sha256": _sha256(path)} for path in input_files
        ],
        "important_limitation": (
            "Counts from the 200-case gold set, 30-case reply review, 25-query "
            "retrieval review, and judge rows have different denominators and must "
            "not be ranked against each other as if they came from one sample."
        ),
    }
    destination = Path(summary_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report
