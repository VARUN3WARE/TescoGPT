"""Deterministic, outcome-blind mini-corpus for a clone-and-run agent demo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from tescogpt.agent.intent import classify_intent
from tescogpt.evaluation.labels import INTENT_LABELS

DEMO_COLUMNS = (
    "case_id",
    "conversation_id",
    "created_at",
    "message",
    "prior_context",
    "historical_reply",
    "customer_followup",
    "private_handoff_proxy",
    "positive_outcome_proxy",
    "unresolved_outcome_proxy",
    "split",
    "selection_intent",
)
SELECTION_METHOD = "sha256_rank_within_deterministic_baseline_intent"


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _selection_key(case_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest()


def validate_demo_corpus_artifact(
    corpus_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Validate a tracked demo corpus without requiring the ignored full dataset."""
    corpus_file = Path(corpus_path)
    manifest_file = Path(manifest_path)
    corpus = pd.read_csv(corpus_file, dtype="string", keep_default_na=False)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    missing = sorted(set(DEMO_COLUMNS) - set(corpus.columns))
    if missing:
        raise ValueError("Demo corpus is missing columns: " + ", ".join(missing))
    if corpus.empty:
        raise ValueError("Demo corpus must not be empty")
    if corpus["case_id"].duplicated().any():
        raise ValueError("Demo corpus contains duplicate case IDs")
    if not corpus["split"].eq("train").all():
        raise ValueError("Demo corpus may contain training rows only")
    if corpus[list(DEMO_COLUMNS)].apply(lambda column: column.str.contains("\r")).any().any():
        raise ValueError("Demo corpus contains carriage returns")

    intent_counts = {
        label: int((corpus["selection_intent"] == label).sum()) for label in INTENT_LABELS
    }
    if any(count <= 0 for count in intent_counts.values()):
        raise ValueError("Demo corpus must represent every configured intent")
    counts = set(intent_counts.values())
    if len(counts) != 1:
        raise ValueError("Demo corpus must use the same number of rows for every intent")

    expected = {
        "demo_corpus_schema_version": 1,
        "selection_method": SELECTION_METHOD,
        "examples_per_intent": counts.pop(),
        "output_file": corpus_file.name,
        "output_sha256": _sha256(corpus_file),
        "output_row_count": len(corpus),
        "intent_counts": intent_counts,
        "source_split": "train",
        "selection_uses_outcome_fields": False,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"Demo corpus manifest {field} does not match the artifact")
    return {
        "row_count": len(corpus),
        "intent_count": len(intent_counts),
        "source_split": "train",
        "output_sha256": expected["output_sha256"],
    }


def build_demo_corpus(
    source_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    examples_per_intent: int = 12,
    seed: int = 20260914,
) -> dict[str, Any]:
    """Select a deterministic hash-ranked sample within baseline intent strata."""
    if examples_per_intent <= 0:
        raise ValueError("examples_per_intent must be positive")
    source_file = Path(source_path)
    destination = Path(output_path)
    manifest_file = Path(manifest_path)
    source = pd.read_csv(source_file, dtype="string", keep_default_na=False)
    required = set(DEMO_COLUMNS) - {"selection_intent"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError("Demo source is missing columns: " + ", ".join(missing))
    if source["case_id"].duplicated().any():
        raise ValueError("Demo source contains duplicate case IDs")

    train = source.loc[source["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("Demo source contains no training rows")
    query = (train["prior_context"] + " " + train["message"]).str.strip()
    train["selection_intent"] = query.map(lambda text: classify_intent(text)[0])
    train["_selection_key"] = train["case_id"].map(
        lambda case_id: _selection_key(str(case_id), seed)
    )

    selected_groups: list[pd.DataFrame] = []
    for intent in INTENT_LABELS:
        group = train.loc[train["selection_intent"].eq(intent)].sort_values(
            ["_selection_key", "case_id"], kind="stable"
        )
        if len(group) < examples_per_intent:
            raise ValueError(
                f"Demo source has only {len(group)} rows for {intent}; "
                f"need {examples_per_intent}"
            )
        selected_groups.append(group.head(examples_per_intent))
    selected = pd.concat(selected_groups, ignore_index=True)[list(DEMO_COLUMNS)]

    destination.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(destination, index=False, lineterminator="\n")
    intent_counts = {
        label: int((selected["selection_intent"] == label).sum()) for label in INTENT_LABELS
    }
    manifest: dict[str, Any] = {
        "demo_corpus_schema_version": 1,
        "selection_method": SELECTION_METHOD,
        "selection_uses_outcome_fields": False,
        "selection_seed": seed,
        "examples_per_intent": examples_per_intent,
        "source_file": source_file.name,
        "source_sha256": _sha256(source_file),
        "source_row_count": len(source),
        "source_split": "train",
        "source_train_row_count": len(train),
        "output_file": destination.name,
        "output_sha256": _sha256(destination),
        "output_row_count": len(selected),
        "intent_counts": intent_counts,
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    validate_demo_corpus_artifact(destination, manifest_file)
    return manifest
