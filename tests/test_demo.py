from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from tescogpt.data.demo import build_demo_corpus, validate_demo_corpus_artifact
from tescogpt.demo import run_demo
from tescogpt.evaluation.labels import INTENT_LABELS

INTENT_MESSAGES = {
    "delivery_or_collection": "My delivery has a missing bag",
    "product_quality_or_safety": "I found glass in this product",
    "product_availability": "Is this item available in stock?",
    "pricing_promotion_or_clubcard": "My Clubcard offer price is wrong",
    "refund_return_or_exchange": "How do I return this for a refund?",
    "online_account_or_checkout": "I cannot login to the website",
    "store_or_staff_experience": "The store staff were unhelpful",
    "product_information": "Does this contain gluten?",
    "feedback_praise_or_suggestion": "Thanks for the brilliant service",
    "other_or_unclear": "Hello there",
}


def _source(path: Path, rows_per_intent: int = 2) -> Path:
    rows = []
    for intent, message in INTENT_MESSAGES.items():
        for index in range(rows_per_intent):
            suffix = hashlib.sha256(f"{intent}-{index}".encode()).hexdigest()[:8]
            rows.append(
                {
                    "case_id": f"case-{suffix}",
                    "conversation_id": f"conversation-{suffix}",
                    "created_at": "2017-01-01T00:00:00Z",
                    "message": f"{message} ref{index}",
                    "prior_context": "",
                    "historical_reply": "A historical Tesco reply.",
                    "customer_followup": "Thanks",
                    "private_handoff_proxy": "False",
                    "positive_outcome_proxy": "True",
                    "unresolved_outcome_proxy": "False",
                    "split": "train",
                }
            )
    rows.append({**rows[0], "case_id": "held-out", "split": "test"})
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")
    return path


def test_demo_corpus_is_balanced_deterministic_and_training_only(tmp_path: Path) -> None:
    source = _source(tmp_path / "cases.csv")
    output = tmp_path / "demo.csv"
    manifest = tmp_path / "demo.manifest.json"

    first = build_demo_corpus(
        source,
        output,
        manifest,
        examples_per_intent=1,
        seed=17,
    )
    first_bytes = output.read_bytes()
    second = build_demo_corpus(
        source,
        output,
        manifest,
        examples_per_intent=1,
        seed=17,
    )
    corpus = pd.read_csv(output, dtype="string", keep_default_na=False)

    assert first == second
    assert output.read_bytes() == first_bytes
    assert len(corpus) == len(INTENT_LABELS)
    assert corpus["split"].eq("train").all()
    assert set(corpus["selection_intent"]) == set(INTENT_LABELS)
    assert not set(corpus["case_id"]) & {"held-out"}
    assert first["selection_uses_outcome_fields"] is False
    assert validate_demo_corpus_artifact(output, manifest)["intent_count"] == 10

    changed_source = tmp_path / "cases-with-changed-outcomes.csv"
    changed = pd.read_csv(source, dtype="string", keep_default_na=False)
    changed["positive_outcome_proxy"] = "False"
    changed["unresolved_outcome_proxy"] = "True"
    changed.to_csv(changed_source, index=False, lineterminator="\n")
    changed_output = tmp_path / "changed-demo.csv"
    build_demo_corpus(
        changed_source,
        changed_output,
        tmp_path / "changed-demo.manifest.json",
        examples_per_intent=1,
        seed=17,
    )
    assert pd.read_csv(changed_output)["case_id"].tolist() == corpus["case_id"].tolist()


def test_demo_corpus_validator_rejects_a_changed_artifact(tmp_path: Path) -> None:
    source = _source(tmp_path / "cases.csv")
    output = tmp_path / "demo.csv"
    manifest = tmp_path / "demo.manifest.json"
    build_demo_corpus(source, output, manifest, examples_per_intent=1)
    output.write_text(output.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output_sha256"):
        validate_demo_corpus_artifact(output, manifest)


def test_committed_clone_and_run_corpus_matches_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    status = validate_demo_corpus_artifact(
        root / "data/sample/tesco_demo_corpus.csv",
        root / "data/sample/tesco_demo_corpus.manifest.json",
    )

    assert status["row_count"] == 120
    assert status["intent_count"] == 10
    assert status["source_split"] == "train"


def test_offline_demo_returns_policy_and_historical_evidence(tmp_path: Path) -> None:
    source = _source(tmp_path / "cases.csv")
    corpus = tmp_path / "demo.csv"
    manifest = tmp_path / "demo.manifest.json"
    build_demo_corpus(source, corpus, manifest, examples_per_intent=1)

    result = run_demo(
        "I found glass in this food",
        corpus,
        corpus_manifest_path=manifest,
    )

    prediction = result["prediction"]
    assert prediction["predicted_intent"] == "product_quality_or_safety"
    assert prediction["handling_decision"] == "ESCALATE"
    assert prediction["decision_reason"] == "FOOD_SAFETY_OR_INJURY"
    assert prediction["evidence_case_ids"] == ()
    assert result["retrieved_precedents"][0]["historical_customer_message"]
    assert result["retrieved_precedents"][0]["used_by_drafter"] is False
    assert "not a measured evaluation result" in result["notices"][0]
    json.dumps(result)


def test_demo_requires_message_and_explicit_api_model(tmp_path: Path) -> None:
    corpus = tmp_path / "missing.csv"
    with pytest.raises(ValueError, match="must not be blank"):
        run_demo("  ", corpus)
    with pytest.raises(ValueError, match="explicit model"):
        run_demo("Hello", corpus, system="main-openai")
