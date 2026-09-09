from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from tescogpt.agent.intent import classify_intent
from tescogpt.agent.schema import AgentOutput
from tescogpt.baselines.simple import SimpleBaseline, route_case
from tescogpt.baselines.trivial import TrivialBaseline
from tescogpt.prediction import (
    run_predictions,
    smoke_test_openai,
    validate_prediction_artifact,
)
from tescogpt.retrieval.bm25 import BM25Index


def _corpus() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "train-1",
                "conversation_id": "10",
                "message": "my grocery delivery is late",
                "historical_reply": "Sorry about the delay. Please contact our delivery team.",
                "split": "train",
            },
            {
                "case_id": "train-2",
                "conversation_id": "20",
                "message": "thank you to the lovely cashier",
                "historical_reply": "Thanks for the lovely feedback!",
                "split": "train",
            },
        ]
    )


def test_agent_output_rejects_incompatible_route() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        AgentOutput(
            case_id="case-1",
            system_name="broken",
            predicted_intent="delivery_or_collection",
            intent_confidence=0.5,
            draft_reply="A reply",
            handling_decision="AUTO_HANDLE",
            decision_reason="MONEY_OR_COMMITMENT",
        )


@pytest.mark.parametrize(
    ("evidence_case_ids", "evidence_quotes", "evidence_scores", "match"),
    [
        (("train-1", "train-1"), ("one", "two"), (1.0, 0.5), "unique"),
        (("",), ("one",), (1.0,), "non-empty strings"),
        (("train-1",), ("",), (1.0,), "quotes"),
        (("train-1",), ("one",), (float("nan"),), "finite"),
    ],
)
def test_agent_output_rejects_malformed_evidence(
    evidence_case_ids: tuple[str, ...],
    evidence_quotes: tuple[str, ...],
    evidence_scores: tuple[float, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        AgentOutput(
            case_id="case-1",
            system_name="broken",
            predicted_intent="feedback_praise_or_suggestion",
            intent_confidence=0.9,
            draft_reply="Thanks.",
            handling_decision="AUTO_HANDLE",
            decision_reason="NO_ACTION_NEEDED",
            evidence_case_ids=evidence_case_ids,
            evidence_quotes=evidence_quotes,
            evidence_scores=evidence_scores,
        )


def test_trivial_baseline_always_escalates() -> None:
    result = TrivialBaseline().predict({"case_id": "case-1"})
    assert result.predicted_intent == "other_or_unclear"
    assert result.handling_decision == "ESCALATE"
    assert not result.evidence_case_ids


def test_simple_baseline_routes_safety_and_cites_retrieval() -> None:
    result = SimpleBaseline(_corpus()).predict(
        {
            "case_id": "test-1",
            "conversation_id": "99",
            "message": "My delivery contained raw chicken and made me ill",
            "prior_context": "",
        }
    )
    assert result.predicted_intent == "product_quality_or_safety"
    assert result.handling_decision == "ESCALATE"
    assert result.decision_reason == "FOOD_SAFETY_OR_INJURY"
    assert result.evidence_case_ids == ("train-1",)


def test_bm25_excludes_entire_conversation() -> None:
    index = BM25Index(_corpus())
    results = index.search("delivery late", exclude_conversation_id="10")
    assert results == []


def test_intent_priority_keeps_safety_above_refund() -> None:
    intent, _ = classify_intent("I need a refund because this chicken made me sick")
    assert intent == "product_quality_or_safety"


def test_simple_store_rule_uses_human_judgment_without_distress() -> None:
    handling, reason, flags = route_case(
        "store_or_staff_experience",
        "The queue in your shop was very long",
    )

    assert handling == "ESCALATE"
    assert reason == "HUMAN_JUDGMENT_REQUIRED"
    assert "repeated_failure" not in flags
    assert "strong_distress" not in flags


def test_prediction_runner_writes_rows_and_manifest(tmp_path: Path) -> None:
    cases = pd.DataFrame(
        [
            {
                "case_id": "case-1",
                "conversation_id": "99",
                "message": "Thanks Tesco",
                "prior_context": "",
            }
        ]
    )
    input_path = tmp_path / "cases.csv"
    output_path = tmp_path / "predictions.csv"
    cases.to_csv(input_path, index=False)

    manifest = run_predictions("trivial", input_path, output_path)

    output = pd.read_csv(output_path)
    saved_manifest = json.loads(
        output_path.with_suffix(".csv.manifest.json").read_text(encoding="utf-8")
    )
    assert len(output) == 1
    assert output.loc[0, "system_name"] == "trivial_constant_v1"
    assert output.loc[0, "proposed_draft"] == output.loc[0, "draft_reply"]
    assert not bool(output.loc[0, "draft_was_replaced"])
    assert manifest == saved_manifest


def test_prediction_runner_refuses_unpartitioned_retrieval_corpus(
    tmp_path: Path,
) -> None:
    cases = pd.DataFrame(
        [
            {
                "case_id": "case-1",
                "conversation_id": "99",
                "message": "Where is my delivery?",
                "prior_context": "",
            }
        ]
    )
    input_path = tmp_path / "cases.csv"
    corpus_path = tmp_path / "corpus.csv"
    cases.to_csv(input_path, index=False)
    _corpus().drop(columns="split").to_csv(corpus_path, index=False)

    with pytest.raises(ValueError, match="explicit split"):
        run_predictions(
            "simple",
            input_path,
            tmp_path / "predictions.csv",
            corpus_path,
        )


class _SmokeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="smoke-response-1",
            model="resolved-smoke-model",
            output_text=json.dumps(
                {
                    "predicted_intent": "feedback_praise_or_suggestion",
                    "intent_confidence": 0.95,
                    "draft_reply": "Thanks for sharing your feedback with Tesco.",
                    "used_evidence_case_ids": [],
                }
            ),
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


def test_api_smoke_runs_one_case_and_warms_the_cache(tmp_path: Path) -> None:
    cases = pd.DataFrame(
        [
            {
                "case_id": "case-1",
                "conversation_id": "99",
                "message": "Thanks Tesco",
                "prior_context": "",
            },
            {
                "case_id": "case-2",
                "conversation_id": "98",
                "message": "Thank you",
                "prior_context": "",
            },
        ]
    )
    corpus = _corpus().assign(
        customer_followup="",
        private_handoff_proxy="false",
        positive_outcome_proxy="false",
        unresolved_outcome_proxy="false",
    )
    input_path = tmp_path / "cases.csv"
    corpus_path = tmp_path / "corpus.csv"
    cache_dir = tmp_path / "cache"
    cases.to_csv(input_path, index=False)
    corpus.to_csv(corpus_path, index=False)
    responses = _SmokeResponses()
    client = SimpleNamespace(responses=responses)

    first = smoke_test_openai(
        input_path,
        corpus_path,
        model="smoke-model",
        case_id="case-2",
        cache_dir=cache_dir,
        client=client,
    )
    second = smoke_test_openai(
        input_path,
        corpus_path,
        model="smoke-model",
        case_id="case-2",
        cache_dir=cache_dir,
        client=client,
    )
    batch = run_predictions(
        "main-openai",
        input_path,
        tmp_path / "predictions.csv",
        corpus_path,
        model="smoke-model",
        cache_dir=cache_dir,
        client=client,
    )

    assert len(responses.calls) == 2
    assert first["case_id"] == "case-2"
    assert first["generation_provenance"]["api_call_count"] == 1
    assert second["generation_provenance"]["api_call_count"] == 0
    assert second["generation_provenance"]["cache_hit_count"] == 1
    assert batch["generation_provenance"]["request_count"] == 2
    assert batch["generation_provenance"]["api_call_count"] == 1
    assert batch["generation_provenance"]["cache_hit_count"] == 1


def test_validator_rejects_hash_consistent_missing_prediction(tmp_path: Path) -> None:
    cases = pd.DataFrame(
        [
            {
                "case_id": case_id,
                "conversation_id": f"conversation-{case_id}",
                "message": "Thanks Tesco",
                "prior_context": "",
            }
            for case_id in ("case-1", "case-2")
        ]
    )
    input_path = tmp_path / "cases.csv"
    output_path = tmp_path / "predictions.csv"
    manifest_path = output_path.with_suffix(".csv.manifest.json")
    cases.to_csv(input_path, index=False)
    run_predictions("trivial", input_path, output_path)

    predictions = pd.read_csv(output_path).iloc[:1]
    predictions.to_csv(output_path, index=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prediction_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    manifest["row_count"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="case IDs do not exactly match"):
        validate_prediction_artifact(input_path, output_path, manifest_path)
