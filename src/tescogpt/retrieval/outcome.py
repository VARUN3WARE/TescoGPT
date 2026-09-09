"""Outcome-aware reranking and auditable retrieval artifact generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tescogpt.policy.safety import inspect_reply
from tescogpt.retrieval.bm25 import BM25Index


@dataclass(frozen=True)
class Precedent:
    case_id: str
    conversation_id: str
    lexical_score: float
    rerank_score: float
    outcome_tier: str
    safety_penalty_flags: tuple[str, ...]
    message: str
    historical_reply: str
    customer_followup: str


def outcome_tier(row: pd.Series) -> str:
    """Describe available follow-up evidence without claiming true resolution."""
    positive = str(row.get("positive_outcome_proxy", "")).lower() == "true"
    unresolved = str(row.get("unresolved_outcome_proxy", "")).lower() == "true"
    private = str(row.get("private_handoff_proxy", "")).lower() == "true"
    followup = str(row.get("customer_followup", "")).strip()
    if positive and unresolved:
        return "conflicting_followup_proxy"
    if unresolved:
        return "unresolved_followup_proxy"
    if positive:
        return "positive_followup_proxy"
    if private:
        return "private_handoff_without_outcome"
    if followup:
        return "unclassified_followup"
    return "no_customer_followup"


_OUTCOME_ADJUSTMENT = {
    "positive_followup_proxy": 0.18,
    "unclassified_followup": 0.0,
    "no_customer_followup": -0.03,
    "private_handoff_without_outcome": -0.08,
    "conflicting_followup_proxy": -0.20,
    "unresolved_followup_proxy": -0.25,
}


class OutcomeAwareRetriever:
    """Retrieve lexically similar cases, then rerank by weak outcome evidence."""

    def __init__(self, corpus: pd.DataFrame, candidate_pool: int = 50) -> None:
        required = {
            "case_id",
            "conversation_id",
            "message",
            "historical_reply",
            "customer_followup",
            "private_handoff_proxy",
            "positive_outcome_proxy",
            "unresolved_outcome_proxy",
        }
        missing = sorted(required - set(corpus.columns))
        if missing:
            raise ValueError(f"Outcome corpus is missing columns: {', '.join(missing)}")
        if candidate_pool <= 0:
            raise ValueError("candidate_pool must be positive")
        self._candidate_pool = candidate_pool
        records = corpus.copy()
        records["case_id"] = records["case_id"].astype(str)
        self._rows = records.set_index("case_id", drop=False)
        self._bm25 = BM25Index(records)

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        exclude_case_id: str | None = None,
        exclude_conversation_id: str | int | None = None,
    ) -> list[Precedent]:
        if top_k <= 0 or top_k > self._candidate_pool:
            raise ValueError("top_k must be positive and no larger than candidate_pool")
        lexical = self._bm25.search(
            query,
            top_k=self._candidate_pool,
            exclude_case_id=exclude_case_id,
            exclude_conversation_id=exclude_conversation_id,
        )
        if not lexical:
            return []
        maximum = lexical[0].score or 1.0
        reranked: list[Precedent] = []
        for result in lexical:
            row = self._rows.loc[result.case_id]
            tier = outcome_tier(row)
            safety_flags = inspect_reply(result.historical_reply)
            safety_penalty = min(0.05 * len(safety_flags), 0.20)
            score = (
                0.82 * result.score / maximum
                + _OUTCOME_ADJUSTMENT[tier]
                - safety_penalty
            )
            reranked.append(
                Precedent(
                    case_id=result.case_id,
                    conversation_id=result.conversation_id,
                    lexical_score=result.score,
                    rerank_score=score,
                    outcome_tier=tier,
                    safety_penalty_flags=safety_flags,
                    message=result.message,
                    historical_reply=result.historical_reply,
                    customer_followup=str(row["customer_followup"]),
                )
            )
        reranked.sort(key=lambda item: (-item.rerank_score, item.case_id))
        return reranked[:top_k]


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_retrieval_artifact(
    input_path: str | Path,
    corpus_path: str | Path,
    output_path: str | Path,
    *,
    top_k: int = 3,
    candidate_pool: int = 50,
) -> dict[str, Any]:
    """Retrieve precedents for every input case and save scores and raw evidence."""
    source = Path(input_path)
    corpus_file = Path(corpus_path)
    destination = Path(output_path)
    cases = pd.read_csv(source, dtype="string", keep_default_na=False)
    corpus = pd.read_csv(corpus_file, dtype="string", keep_default_na=False)
    if "split" not in corpus.columns:
        raise ValueError("Retrieval corpus must contain an explicit split column")
    train = corpus.loc[corpus["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("Retrieval corpus contains no train rows")
    retriever = OutcomeAwareRetriever(train, candidate_pool=candidate_pool)

    records: list[dict[str, Any]] = []
    for case in cases.to_dict(orient="records"):
        query = f"{case.get('prior_context', '')} {case['message']}".strip()
        precedents = retriever.search(
            query,
            top_k=top_k,
            exclude_case_id=str(case["case_id"]),
            exclude_conversation_id=case.get("conversation_id"),
        )
        for rank, precedent in enumerate(precedents, start=1):
            record = asdict(precedent)
            record["safety_penalty_flags"] = json.dumps(
                record["safety_penalty_flags"]
            )
            records.append({"query_case_id": case["case_id"], "rank": rank, **record})
    artifact = pd.DataFrame(records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact.to_csv(destination, index=False, lineterminator="\n")
    manifest = {
        "retrieval_schema_version": 1,
        "method": "bm25_outcome_rerank_v1",
        "input_file": source.name,
        "input_sha256": _sha256(source),
        "corpus_file": corpus_file.name,
        "corpus_sha256": _sha256(corpus_file),
        "corpus_split": "train",
        "corpus_row_count": len(train),
        "candidate_pool": candidate_pool,
        "top_k": top_k,
        "query_count": len(cases),
        "retrieved_row_count": len(artifact),
        "output_file": destination.name,
        "output_sha256": _sha256(destination),
    }
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
