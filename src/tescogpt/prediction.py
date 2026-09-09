"""Run comparable support systems and persist auditable predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from tescogpt.agent.drafting import OpenAIDrafter, SafeTemplateDrafter
from tescogpt.agent.main import EvidencePolicyAgent
from tescogpt.agent.schema import PREDICTION_COLUMNS, AgentOutput
from tescogpt.baselines import SimpleBaseline, TrivialBaseline


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _json_array(value: str, column: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Prediction {column} must be a JSON array") from error
    if not isinstance(parsed, list):
        raise ValueError(f"Prediction {column} must be a JSON array")
    return parsed


def validate_prediction_artifact(
    input_path: str | Path,
    prediction_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Validate frozen prediction coverage, row schema, and manifest provenance."""
    source = Path(input_path)
    destination = Path(prediction_path)
    manifest_file = Path(manifest_path)
    cases = pd.read_csv(source, dtype="string", keep_default_na=False)
    predictions = pd.read_csv(destination, dtype="string", keep_default_na=False)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    if "case_id" not in cases.columns:
        raise ValueError("Prediction input is missing case_id")
    if cases["case_id"].duplicated().any():
        raise ValueError("Prediction input contains duplicate case IDs")
    missing = sorted(set(PREDICTION_COLUMNS) - set(predictions.columns))
    if missing:
        raise ValueError("Prediction artifact is missing columns: " + ", ".join(missing))
    if predictions["case_id"].duplicated().any():
        raise ValueError("Prediction artifact contains duplicate case IDs")
    if set(predictions["case_id"]) != set(cases["case_id"]):
        raise ValueError("Prediction case IDs do not exactly match the frozen input")

    for row in predictions.to_dict(orient="records"):
        replaced_text = str(row["draft_was_replaced"]).strip().lower()
        if replaced_text not in {"true", "false"}:
            raise ValueError("Prediction draft_was_replaced must be true or false")
        evidence_ids = _json_array(str(row["evidence_case_ids"]), "evidence_case_ids")
        evidence_quotes = _json_array(str(row["evidence_quotes"]), "evidence_quotes")
        evidence_scores = _json_array(str(row["evidence_scores"]), "evidence_scores")
        safety_flags = _json_array(str(row["safety_flags"]), "safety_flags")
        if not all(isinstance(item, str) for item in evidence_ids):
            raise ValueError("Prediction evidence_case_ids must contain strings")
        if not all(isinstance(item, str) for item in evidence_quotes):
            raise ValueError("Prediction evidence_quotes must contain strings")
        if not all(isinstance(item, int | float) for item in evidence_scores):
            raise ValueError("Prediction evidence_scores must contain numbers")
        if not all(isinstance(item, str) for item in safety_flags):
            raise ValueError("Prediction safety_flags must contain strings")
        AgentOutput(
            case_id=str(row["case_id"]),
            system_name=str(row["system_name"]),
            predicted_intent=str(row["predicted_intent"]),
            intent_confidence=float(row["intent_confidence"]),
            proposed_draft=str(row["proposed_draft"]),
            draft_reply=str(row["draft_reply"]),
            draft_was_replaced=replaced_text == "true",
            handling_decision=str(row["handling_decision"]),
            decision_reason=str(row["decision_reason"]),
            automation_score=float(row["automation_score"]),
            evidence_case_ids=tuple(evidence_ids),
            evidence_quotes=tuple(evidence_quotes),
            evidence_scores=tuple(float(score) for score in evidence_scores),
            safety_flags=tuple(safety_flags),
        )

    systems = sorted(set(predictions["system_name"]))
    if len(systems) != 1:
        raise ValueError("Prediction artifact must contain exactly one system")
    expected_manifest = {
        "prediction_schema_version": 2,
        "system": systems[0],
        "input_file": source.name,
        "input_sha256": _sha256(source),
        "prediction_file": destination.name,
        "prediction_sha256": _sha256(destination),
        "row_count": len(predictions),
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise ValueError(f"Prediction manifest {field} does not match the artifact")
    return {
        "system": systems[0],
        "row_count": len(predictions),
        "input_case_count": len(cases),
    }


def run_predictions(
    system: str,
    input_path: str | Path,
    output_path: str | Path,
    corpus_path: str | Path | None = None,
    model: str | None = None,
    cache_dir: str | Path = "artifacts/cache/openai_drafts",
    client: Any | None = None,
) -> dict[str, Any]:
    """Run one system over a case CSV and write predictions plus provenance."""
    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Prediction input does not exist: {source}")
    cases = pd.read_csv(source, dtype="string", keep_default_na=False)
    required = {"case_id", "conversation_id", "message", "prior_context"}
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ValueError(f"Prediction input is missing columns: {', '.join(missing)}")

    corpus_file: Path | None = None
    corpus: pd.DataFrame | None = None
    openai_drafter: OpenAIDrafter | None = None
    needs_corpus = system in {"simple", "main-template", "main-openai"}
    if needs_corpus:
        if corpus_path is None:
            raise ValueError(f"The {system} system requires a retrieval corpus")
        corpus_file = Path(corpus_path)
        if not corpus_file.is_file():
            raise FileNotFoundError(f"Retrieval corpus does not exist: {corpus_file}")
        corpus = pd.read_csv(corpus_file, dtype="string", keep_default_na=False)
        if "split" not in corpus.columns:
            raise ValueError("Retrieval corpus must contain an explicit split column")
        corpus = corpus.loc[corpus["split"].eq("train")].copy()
        if corpus.empty:
            raise ValueError("Retrieval corpus contains no train rows")

    if system == "trivial":
        agent = TrivialBaseline()
    elif system == "simple":
        assert corpus is not None
        agent = SimpleBaseline(corpus)
    elif system == "main-template":
        assert corpus is not None
        agent = EvidencePolicyAgent(corpus, SafeTemplateDrafter())
    elif system == "main-openai":
        assert corpus is not None
        if model is None:
            raise ValueError("main-openai requires an explicit model ID")
        openai_drafter = OpenAIDrafter(model, cache_dir, client=client)
        agent = EvidencePolicyAgent(corpus, openai_drafter)
    else:
        raise ValueError(f"Unknown system: {system}")

    records = [agent.predict(row).to_record() for row in cases.to_dict(orient="records")]
    predictions = pd.DataFrame(records)
    if predictions["case_id"].duplicated().any():
        raise AssertionError("Predictions must contain unique case IDs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False, lineterminator="\n")

    manifest: dict[str, Any] = {
        "prediction_schema_version": 2,
        "system": agent.name,
        "input_file": source.name,
        "input_sha256": _sha256(source),
        "prediction_file": destination.name,
        "prediction_sha256": _sha256(destination),
        "row_count": len(predictions),
    }
    if corpus_file is not None:
        assert corpus is not None
        manifest["corpus_file"] = corpus_file.name
        manifest["corpus_sha256"] = _sha256(corpus_file)
        manifest["corpus_split"] = "train"
        manifest["corpus_row_count"] = len(corpus)
    if model is not None:
        manifest["model"] = model
    if openai_drafter is not None:
        manifest["generation_provenance"] = openai_drafter.provenance()
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    validate_prediction_artifact(source, destination, manifest_path)
    return manifest


def smoke_test_openai(
    input_path: str | Path,
    corpus_path: str | Path,
    *,
    model: str,
    case_id: str | None = None,
    cache_dir: str | Path = "artifacts/cache/openai_drafts",
    client: Any | None = None,
) -> dict[str, Any]:
    """Run one frozen case through the API path and warm the full-run cache."""
    source = Path(input_path)
    corpus_file = Path(corpus_path)
    if not source.is_file():
        raise FileNotFoundError(f"Prediction input does not exist: {source}")
    if not corpus_file.is_file():
        raise FileNotFoundError(f"Retrieval corpus does not exist: {corpus_file}")

    cases = pd.read_csv(source, dtype="string", keep_default_na=False)
    required = {"case_id", "conversation_id", "message", "prior_context"}
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ValueError(f"Prediction input is missing columns: {', '.join(missing)}")
    if cases.empty:
        raise ValueError("Prediction input contains no cases")
    if cases["case_id"].duplicated().any():
        raise ValueError("Prediction input contains duplicate case IDs")
    if case_id is None:
        selected = cases.iloc[0]
    else:
        matches = cases.loc[cases["case_id"].eq(case_id)]
        if matches.empty:
            raise ValueError(f"Smoke-test case ID is not in the input: {case_id}")
        selected = matches.iloc[0]

    corpus = pd.read_csv(corpus_file, dtype="string", keep_default_na=False)
    if "split" not in corpus.columns:
        raise ValueError("Retrieval corpus must contain an explicit split column")
    train = corpus.loc[corpus["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("Retrieval corpus contains no train rows")

    drafter = OpenAIDrafter(model, cache_dir, client=client)
    agent = EvidencePolicyAgent(train, drafter)
    prediction = agent.predict(selected.to_dict()).to_record()
    return {
        "smoke_test_schema_version": 1,
        "case_id": str(selected["case_id"]),
        "system": agent.name,
        "input_file": source.name,
        "input_sha256": _sha256(source),
        "corpus_file": corpus_file.name,
        "corpus_sha256": _sha256(corpus_file),
        "corpus_split": "train",
        "corpus_row_count": len(train),
        "prediction": prediction,
        "generation_provenance": drafter.provenance(),
    }
