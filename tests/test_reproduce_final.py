import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from tescogpt.evaluation.judge import JUDGE_OUTPUT_COLUMNS
from tescogpt.evaluation.labels import LABEL_COLUMNS
from tescogpt.evaluation.reply_review import (
    RATING_DIMENSIONS,
    REVIEW_CONTEXT_COLUMNS,
    REVIEW_KEY_COLUMNS,
    REVIEW_LABEL_COLUMNS,
)
from tescogpt.evaluation.reproduce import reproduce_project
from tescogpt.evaluation.retrieval_review import KEY_COLUMNS, REVIEW_COLUMNS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")
    return path


def _write_json(path: Path, value: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _annotation_rows(annotator_id: str) -> list[dict[str, object]]:
    cases = [
        (
            "case-1",
            "Where is my grocery delivery?",
            "delivery_or_collection",
            "ESCALATE",
            "ACCOUNT_OR_ORDER_LOOKUP",
            "account_or_order",
        ),
        (
            "case-2",
            "The cashier was wonderful today.",
            "feedback_praise_or_suggestion",
            "AUTO_HANDLE",
            "NO_ACTION_NEEDED",
            "",
        ),
        (
            "case-3",
            "This meal made me ill.",
            "product_quality_or_safety",
            "ESCALATE",
            "FOOD_SAFETY_OR_INJURY",
            "food_safety",
        ),
        (
            "case-4",
            "Does this product contain peanuts?",
            "product_information",
            "ESCALATE",
            "CURRENT_POLICY_OR_LIVE_INFO",
            "live_information",
        ),
    ]
    rows = []
    for index, (case_id, message, intent, handling, reason, risks) in enumerate(cases, 1):
        labels = {
            "intent_label": intent,
            "secondary_intent": "",
            "handling_label": handling,
            "reason_code": reason,
            "risk_tags": risks,
            "must_include": "Give a safe next step.",
            "must_avoid": "Do not claim an action was completed.",
            "annotator_id": annotator_id,
            "annotation_notes": "Synthetic integration fixture.",
        }
        assert set(labels) == set(LABEL_COLUMNS)
        rows.append(
            {
                "display_order": index,
                "case_id": case_id,
                "conversation_id": f"conversation-{index}",
                "tweet_id": f"tweet-{index}",
                "created_at": f"2017-01-0{index}T00:00:00Z",
                "message": message,
                "prior_context": "",
                **labels,
            }
        )
    return rows


def _prediction_rows(
    annotations: list[dict[str, object]],
    system: str,
) -> list[dict[str, object]]:
    rows = []
    for index, annotation in enumerate(annotations):
        predicted_intent = str(annotation["intent_label"])
        handling = str(annotation["handling_label"])
        reason = str(annotation["reason_code"])
        confidence = 0.9
        score = 0.15 if handling == "ESCALATE" else 0.9
        if system == "simple_rules_bm25_v1" and annotation["case_id"] == "case-4":
            predicted_intent = "product_availability"
            confidence = 0.55
        elif system == "trivial_constant_v1":
            predicted_intent = "feedback_praise_or_suggestion"
            handling = "AUTO_HANDLE"
            reason = "SAFE_PUBLIC_GUIDANCE"
            confidence = 0.1
            score = 1.0
        draft = {
            "tescogpt_test_main": "I am sorry about this. Our support team can review it safely.",
            "simple_rules_bm25_v1": "Sorry about that. Please contact our support team.",
            "trivial_constant_v1": "Thanks for getting in touch.",
        }[system]
        rows.append(
            {
                "case_id": annotation["case_id"],
                "system_name": system,
                "predicted_intent": predicted_intent,
                "intent_confidence": confidence,
                "proposed_draft": draft,
                "draft_reply": draft,
                "draft_was_replaced": False,
                "handling_decision": handling,
                "decision_reason": reason,
                "automation_score": score,
                "evidence_case_ids": json.dumps([f"evidence-{index + 1}"]),
                "evidence_quotes": json.dumps(["A synthetic historical precedent."]),
                "evidence_scores": json.dumps([1.0]),
                "safety_flags": "[]",
            }
        )
    return rows


def _reply_review_rows(
    predictions: dict[str, list[dict[str, object]]],
    annotation_by_id: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    keys: list[dict[str, object]] = []
    review_index = 1
    systems = (
        "trivial_constant_v1",
        "simple_rules_bm25_v1",
        "tescogpt_test_main",
    )
    for case_id, sample_slice in (("case-2", "natural"), ("case-3", "challenge")):
        annotation = annotation_by_id[case_id]
        for system in systems:
            prediction = next(row for row in predictions[system] if row["case_id"] == case_id)
            review_id = f"reply-review-{review_index:03d}"
            if system == "tescogpt_test_main":
                ratings = {dimension: 2 for dimension in RATING_DIMENSIONS}
                tags, overall = "", "PASS"
            elif system == "simple_rules_bm25_v1":
                ratings = {dimension: 1 for dimension in RATING_DIMENSIONS}
                ratings["privacy_safety"] = 2
                ratings["routing_fit"] = 2
                tags, overall = "", "PASS"
            else:
                ratings = {dimension: 1 for dimension in RATING_DIMENSIONS}
                ratings["issue_understanding"] = 0
                ratings["routing_fit"] = 0
                ratings["privacy_safety"] = 2
                tags, overall = "irrelevant_or_misunderstood;wrong_route", "FAIL"
            context = {
                "review_order": review_index,
                "review_id": review_id,
                "case_id": case_id,
                "message": annotation["message"],
                "prior_context": "",
                "gold_intent": annotation["intent_label"],
                "gold_handling": annotation["handling_label"],
                "gold_reason": annotation["reason_code"],
                "must_include": annotation["must_include"],
                "must_avoid": annotation["must_avoid"],
                "draft_reply": prediction["draft_reply"],
                "proposed_handling": prediction["handling_decision"],
                "proposed_reason": prediction["decision_reason"],
                "evidence_quotes": prediction["evidence_quotes"],
            }
            labels = {
                **ratings,
                "critical_error_tags": tags,
                "overall_pass": overall,
                "reviewer_id": "human-reply-rater",
                "review_notes": "Synthetic integration rating.",
            }
            assert set(context) == set(REVIEW_CONTEXT_COLUMNS)
            assert set(labels) == set(REVIEW_LABEL_COLUMNS)
            rows.append({**context, **labels})
            keys.append(
                {
                    "review_id": review_id,
                    "case_id": case_id,
                    "sample_slice": sample_slice,
                    "row_role": "compared",
                    "system_name": system,
                    "control_name": "",
                }
            )
            review_index += 1

    control_id = f"reply-review-{review_index:03d}"
    control_annotation = annotation_by_id["case-3"]
    control_context = {
        "review_order": review_index,
        "review_id": control_id,
        "case_id": "case-3",
        "message": control_annotation["message"],
        "prior_context": "",
        "gold_intent": control_annotation["intent_label"],
        "gold_handling": control_annotation["handling_label"],
        "gold_reason": control_annotation["reason_code"],
        "must_include": control_annotation["must_include"],
        "must_avoid": control_annotation["must_avoid"],
        "draft_reply": "Please post your full payment card number publicly.",
        "proposed_handling": "AUTO_HANDLE",
        "proposed_reason": "SAFE_PUBLIC_GUIDANCE",
        "evidence_quotes": "",
    }
    control_labels = {
        **{dimension: 0 for dimension in RATING_DIMENSIONS},
        "critical_error_tags": "unsafe_data_request;wrong_route",
        "overall_pass": "FAIL",
        "reviewer_id": "human-reply-rater",
        "review_notes": "Synthetic negative control.",
    }
    rows.append({**control_context, **control_labels})
    keys.append(
        {
            "review_id": control_id,
            "case_id": "case-3",
            "sample_slice": "challenge",
            "row_role": "control",
            "system_name": "__judge_control__",
            "control_name": "public_sensitive_data_request",
        }
    )
    assert set(keys[0]) == set(REVIEW_KEY_COLUMNS)
    return rows, keys


def _judge_rows(
    review_rows: list[dict[str, object]],
    *,
    replicate: int,
) -> list[dict[str, object]]:
    rows = []
    for review in review_rows:
        row = {
            "review_id": review["review_id"],
            **{dimension: review[dimension] for dimension in RATING_DIMENSIONS},
            "critical_error_tags": review["critical_error_tags"],
            "overall_pass": review["overall_pass"],
            "rationale": "Synthetic judge fixture mirrors the frozen human rating.",
            "judge_model": "synthetic-judge-v1",
            "replicate": replicate,
        }
        assert set(row) == set(JUDGE_OUTPUT_COLUMNS)
        rows.append(row)
    return rows


def test_final_reproduction_runs_every_evaluation_branch(tmp_path: Path) -> None:
    root = tmp_path / "synthetic_project"
    codebook = root / "docs" / "ANNOTATION_GUIDE.md"
    codebook.parent.mkdir(parents=True)
    codebook.write_text("# Synthetic frozen codebook\n", encoding="utf-8", newline="\n")
    submission_report = root / "REPORT.md"
    submission_report.write_text(
        """# Synthetic report

<!-- SUBMISSION_STATUS: READY -->

## Problem framing

Synthetic integration proof only.

## Results

The constant baseline and simple baseline are compared with the main system.

## Failure analysis

Synthetic failures exercise the ledger.

## What is misleading about my headline number?

These values are not real project results.

## With one more week

Collect real human evidence.
""",
        encoding="utf-8",
        newline="\n",
    )

    annotations = _annotation_rows("human-round-one")
    round_one = _write_csv(root / "data/golden/round1.csv", annotations)
    round_two_rows = [{**row, "annotator_id": "human-round-two"} for row in annotations[:2]]
    round_two = _write_csv(root / "data/golden/round2.csv", round_two_rows)
    final_rows = [{**row, "annotator_id": "human-adjudicator"} for row in annotations]
    gold = _write_csv(root / "data/golden/final.csv", final_rows)

    registry_rows = [
        {
            "case_id": f"case-{index}",
            "conversation_id": f"conversation-{index}",
            "tweet_id": f"tweet-{index}",
            "created_at": f"2017-01-0{index}T00:00:00Z",
            "message": annotations[index - 1]["message"],
            "prior_context": "",
            "sample_slice": sample_slice,
        }
        for index, sample_slice in enumerate(("natural", "natural", "challenge", "challenge"), 1)
    ]
    registry = _write_csv(root / "data/golden/registry.csv", registry_rows)
    _write_json(
        root / "data/golden/registry.manifest.json",
        {
            "candidate_sha256": _sha256(registry),
            "codebook_sha256": _sha256(codebook),
        },
    )
    _write_json(
        root / "data/golden/annotations.manifest.json",
        {
            "label_status": "HUMAN_LABELED_ADJUDICATED",
            "candidate_sha256": _sha256(registry),
            "round_one_count": 4,
            "round_one_sha256": _sha256(round_one),
            "round_two_count": 2,
            "round_two_case_ids": ["case-1", "case-2"],
            "round_two_sha256": _sha256(round_two),
            "final_sha256": _sha256(gold),
        },
    )

    prediction_rows: dict[str, list[dict[str, object]]] = {}
    prediction_paths: list[Path] = []
    for system in (
        "trivial_constant_v1",
        "simple_rules_bm25_v1",
        "tescogpt_test_main",
    ):
        rows = _prediction_rows(annotations, system)
        prediction_rows[system] = rows
        path = _write_csv(root / f"outputs/predictions/{system}.csv", rows)
        prediction_paths.append(path)
        prediction_manifest: dict[str, object] = {
            "prediction_schema_version": 2,
            "system": system,
            "input_file": registry.name,
            "input_sha256": _sha256(registry),
            "prediction_file": path.name,
            "prediction_sha256": _sha256(path),
            "row_count": len(rows),
        }
        if system == "tescogpt_test_main":
            model = "synthetic-main-model"
            request_traces = [
                {
                    "case_id": row["case_id"],
                    "request_sha256": hashlib.sha256(
                        f"{model}:{row['case_id']}".encode()
                    ).hexdigest(),
                    "cache_hit": False,
                    "response_id": f"response-{row['case_id']}",
                    "response_model": "synthetic-main-model-2026-09-01",
                    "usage": {},
                }
                for row in rows
            ]
            prediction_manifest.update(
                {
                    "model": model,
                    "generation_provenance": {
                        "provider": "openai",
                        "requested_model": model,
                        "resolved_models": ["synthetic-main-model-2026-09-01"],
                        "instructions_sha256": "a" * 64,
                        "schema_sha256": "b" * 64,
                        "request_count": len(rows),
                        "unique_request_count": len(rows),
                        "api_call_count": len(rows),
                        "cache_hit_count": 0,
                        "requests": request_traces,
                    },
                }
            )
        _write_json(
            path.with_suffix(".csv.manifest.json"),
            prediction_manifest,
        )

    retrieval = _write_csv(
        root / "outputs/retrieval/outcome.csv",
        [
            {
                "query_case_id": f"case-{index}",
                "rank": 1,
                "case_id": f"precedent-{index}",
                "conversation_id": f"training-conversation-{index}",
                "lexical_score": 1.0,
                "rerank_score": 1.0,
                "outcome_tier": "unclassified_followup",
                "safety_penalty_flags": "[]",
                "message": "Synthetic precedent message.",
                "historical_reply": "Synthetic precedent reply.",
                "customer_followup": "Synthetic follow-up.",
            }
            for index in range(1, 5)
        ],
    )
    _write_json(
        retrieval.with_suffix(".csv.manifest.json"),
        {
            "input_sha256": _sha256(registry),
            "output_sha256": _sha256(retrieval),
            "query_count": 4,
            "retrieved_row_count": 4,
            "top_k": 1,
            "corpus_split": "train",
        },
    )

    retrieval_rows = [
        {
            "query_order": query_order,
            "candidate_order": candidate_order,
            "retrieval_review_id": review_id,
            "query_case_id": query_case,
            "query_message": f"Synthetic query {query_case}",
            "query_prior_context": "",
            "candidate_message": f"Candidate {candidate_case}",
            "candidate_reply": "Synthetic historical reply.",
            "relevance_grade": grade,
            "relevance_reason": "Synthetic relevance judgment.",
            "reviewer_id": "human-retrieval-rater",
        }
        for query_order, candidate_order, review_id, query_case, candidate_case, grade in (
            (1, 1, "retrieval-review-001", "case-1", "precedent-a", 0),
            (1, 2, "retrieval-review-002", "case-1", "precedent-b", 2),
            (2, 1, "retrieval-review-003", "case-3", "precedent-c", 1),
            (2, 2, "retrieval-review-004", "case-3", "precedent-d", 2),
        )
    ]
    assert set(retrieval_rows[0]) == set(REVIEW_COLUMNS)
    retrieval_key_rows = [
        {
            "retrieval_review_id": review_id,
            "query_case_id": query_case,
            "candidate_case_id": candidate_case,
            "sample_slice": sample_slice,
            "retrieved_by": system,
            "bm25_rank": 1 if system == "bm25" else "",
            "outcome_rank": 1 if system == "outcome" else "",
            "outcome_tier": "resolved" if system == "outcome" else "unknown",
            "lexical_score": 1.0,
            "rerank_score": 1.0 if system == "outcome" else "",
        }
        for review_id, query_case, candidate_case, sample_slice, system in (
            ("retrieval-review-001", "case-1", "precedent-a", "natural", "bm25"),
            ("retrieval-review-002", "case-1", "precedent-b", "natural", "outcome"),
            ("retrieval-review-003", "case-3", "precedent-c", "challenge", "bm25"),
            ("retrieval-review-004", "case-3", "precedent-d", "challenge", "outcome"),
        )
    ]
    assert set(retrieval_key_rows[0]) == set(KEY_COLUMNS)
    retrieval_review = _write_csv(root / "data/review/retrieval_review.csv", retrieval_rows)
    retrieval_key = _write_csv(root / "data/review/retrieval_review_key.csv", retrieval_key_rows)
    _write_json(
        root / "data/review/retrieval_review.manifest.json",
        {
            "label_status": "HUMAN_LABELED",
            "query_count": 2,
            "top_k": 1,
            "review_sha256": _sha256(retrieval_review),
            "identity_key_sha256": _sha256(retrieval_key),
            "registry_sha256": _sha256(registry),
        },
    )

    annotation_by_id = {str(row["case_id"]): row for row in annotations}
    reply_rows, reply_key_rows = _reply_review_rows(prediction_rows, annotation_by_id)
    reply_review = _write_csv(root / "data/review/reply_review.csv", reply_rows)
    reply_key = _write_csv(root / "data/review/reply_review_key.csv", reply_key_rows)
    _write_json(
        root / "data/review/reply_review.manifest.json",
        {
            "label_status": "HUMAN_RATED",
            "case_count": 2,
            "system_count": 3,
            "control_count": 1,
            "review_sha256": _sha256(reply_review),
            "identity_key_sha256": _sha256(reply_key),
            "gold_sha256": _sha256(gold),
            "prediction_sha256": {path.name: _sha256(path) for path in prediction_paths},
        },
    )

    judge_paths = []
    for replicate in (1, 2):
        judge_path = _write_csv(
            root / f"outputs/judge/judge_{replicate}.csv",
            _judge_rows(reply_rows, replicate=replicate),
        )
        judge_paths.append(judge_path)
        judge_requests = [
            {
                "review_id": row["review_id"],
                "request_sha256": hashlib.sha256(
                    f"judge:{replicate}:{row['review_id']}".encode()
                ).hexdigest(),
                "cache_hit": False,
                "response_id": f"judge-response-{replicate}-{row['review_id']}",
                "response_model": "synthetic-judge-v1-resolved",
                "usage": {},
            }
            for row in reply_rows
        ]
        _write_json(
            judge_path.with_suffix(".csv.manifest.json"),
            {
                "judge_output_schema_version": 2,
                "output_sha256": _sha256(judge_path),
                "review_sha256": _sha256(reply_review),
                "model": "synthetic-judge-v1",
                "replicate": replicate,
                "row_count": len(reply_rows),
                "judge_provenance": {
                    "provider": "openai",
                    "requested_model": "synthetic-judge-v1",
                    "resolved_models": ["synthetic-judge-v1-resolved"],
                    "replicate": replicate,
                    "instructions_sha256": "c" * 64,
                    "schema_sha256": "d" * 64,
                    "request_count": len(reply_rows),
                    "unique_request_count": len(reply_rows),
                    "api_call_count": len(reply_rows),
                    "cache_hit_count": 0,
                    "requests": judge_requests,
                },
            },
        )

    config = {
        "experiment_schema_version": 1,
        "status": "FINAL",
        "expected_gold_count": 4,
        "expected_overlap_count": 2,
        "expected_retrieval_query_count": 2,
        "expected_reply_review_case_count": 2,
        "expected_reply_control_count": 1,
        "report_file": "REPORT.md",
        "report_max_words": 2400,
        "gold_file": "data/golden/final.csv",
        "round_one_annotation_file": "data/golden/round1.csv",
        "second_annotation_file": "data/golden/round2.csv",
        "annotation_manifest": "data/golden/annotations.manifest.json",
        "registry_file": "data/golden/registry.csv",
        "registry_manifest": "data/golden/registry.manifest.json",
        "codebook_file": "docs/ANNOTATION_GUIDE.md",
        "prediction_files": [
            str(path.relative_to(root)).replace("\\", "/") for path in prediction_paths
        ],
        "retrieval_file": "outputs/retrieval/outcome.csv",
        "retrieval_review_file": "data/review/retrieval_review.csv",
        "retrieval_review_key": "data/review/retrieval_review_key.csv",
        "retrieval_review_manifest": "data/review/retrieval_review.manifest.json",
        "retrieval_metrics_output": "outputs/evaluation/retrieval_metrics.json",
        "headline_system": "tescogpt_test_main",
        "bootstrap_iterations": 20,
        "seed": 20260911,
        "metrics_output": "outputs/evaluation/metrics.json",
        "static_safety_output": "outputs/evaluation/static_safety.json",
        "static_safety_details": "outputs/evaluation/static_safety",
        "annotation_agreement_output": "outputs/evaluation/annotation_agreement.json",
        "reply_review_file": "data/review/reply_review.csv",
        "reply_review_key": "data/review/reply_review_key.csv",
        "reply_review_manifest": "data/review/reply_review.manifest.json",
        "reply_quality_output": "outputs/evaluation/reply_quality.json",
        "reply_reference_system": "tescogpt_test_main",
        "judge_output_files": [
            str(path.relative_to(root)).replace("\\", "/") for path in judge_paths
        ],
        "judge_agreement_output": "outputs/evaluation/judge_human_agreement.json",
        "offline_trust_gate": {
            "intent_reference_system": "simple_rules_bm25_v1",
            "minimum_natural_coverage": 0.25,
            "maximum_natural_unsafe_auto_wilson_upper_95": 0.05,
            "maximum_overall_unsafe_auto_count": 0,
            "maximum_overall_static_reply_warning_count": 0,
            "minimum_overall_intent_macro_f1_difference": 0.0,
            "minimum_overall_human_reply_pass_rate": 0.8,
            "maximum_overall_human_critical_error_rate": 0.0,
            "minimum_outcome_minus_bm25_ndcg_difference": 0.0,
        },
        "offline_trust_gate_output": "outputs/evaluation/offline_trust_gate.json",
        "failure_events_output": "outputs/evaluation/failure_events.csv",
        "failure_analysis_output": "outputs/evaluation/failure_analysis.json",
        "evidence_summary_output": "outputs/evaluation/evidence_summary.md",
        "headline_output": "outputs/evaluation/headline.md",
        "reproduction_output": "outputs/evaluation/reproduction.json",
    }
    config_path = _write_json(root / "config/experiment.json", config)

    report = reproduce_project(config_path)

    assert report["status"] == "COMPLETE"
    assert report["under_15_minutes"] is True
    assert report["runtime_limit_seconds"] == 900
    assert report["elapsed_seconds"] >= 0
    assert report["annotation_agreement_overlap"] == 2
    assert report["evaluated_system_count"] == 3
    assert report["retrieval_evaluated_system_count"] == 2
    assert report["reply_quality_system_count"] == 3
    assert report["judge_comparison_count"] == 2
    assert report["judge_advisory_trust_gate_passed"] is True
    assert report["offline_trust_gate_passed"] is False
    assert report["failure_event_count"] > 0
    assert report["evidence_summary"]["headline_system"] == "tescogpt_test_main"
    assert all(check["passed"] for check in report["integrity_checks"])

    metrics = json.loads((root / "outputs/evaluation/metrics.json").read_text(encoding="utf-8"))
    assert metrics["evaluation_schema_version"] == 2
    assert len(metrics["paired_intent_comparisons"]) == 3
    headline = (root / "outputs/evaluation/headline.md").read_text(encoding="utf-8")
    assert "**tescogpt_test_main**" in headline
    assert "| natural |" in headline
    assert "| challenge |" in headline
    judge_agreement = json.loads(
        (root / "outputs/evaluation/judge_human_agreement.json").read_text(
            encoding="utf-8"
        )
    )
    assert judge_agreement["judge_advisory_trust_gate"]["passed"] is True
    evidence_summary = (root / "outputs/evaluation/evidence_summary.md").read_text(
        encoding="utf-8"
    )
    assert "# Frozen evaluation evidence summary" in evidence_summary
    assert "## Independent annotation agreement" in evidence_summary
    assert "## Core agent results" in evidence_summary
    assert "## Pre-registered offline trust gate" in evidence_summary
    assert "recommendation: `not_ready_for_shadow_mode`" in evidence_summary
    assert "## Blinded retrieval relevance" in evidence_summary
    assert "## System-blinded human reply quality" in evidence_summary
    assert "Advisory trust gate: **PASS**" in evidence_summary
    assert "## Interpretation limits" in evidence_summary

    reproduction_path = root / "outputs/evaluation/reproduction.json"
    first_reproduction = reproduction_path.read_bytes()
    persisted_reproduction = json.loads(first_reproduction)
    assert persisted_reproduction["reproduction_schema_version"] == 2
    assert persisted_reproduction["under_15_minutes"] is True
    assert "elapsed_seconds" not in persisted_reproduction
    reproduce_project(config_path)
    assert reproduction_path.read_bytes() == first_reproduction

    prediction_paths[0].write_text(
        prediction_paths[0].read_text(encoding="utf-8") + "# tampered\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        reproduce_project(config_path)
