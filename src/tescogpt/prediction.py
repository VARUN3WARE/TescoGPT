"""Run comparable support systems and persist auditable predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from tescogpt.agent.drafting import OpenAIDrafter, SafeTemplateDrafter
from tescogpt.agent.main import EvidencePolicyAgent
from tescogpt.baselines import SimpleBaseline, TrivialBaseline


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def run_predictions(
    system: str,
    input_path: str | Path,
    output_path: str | Path,
    corpus_path: str | Path | None = None,
    model: str | None = None,
    cache_dir: str | Path = "artifacts/cache/openai_drafts",
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
        openai_drafter = OpenAIDrafter(model, cache_dir)
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
    return manifest
