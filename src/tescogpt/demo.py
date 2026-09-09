"""Single-message agent demonstration with inspectable historical evidence."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from tescogpt.agent.drafting import OpenAIDrafter, SafeTemplateDrafter
from tescogpt.agent.main import EvidencePolicyAgent
from tescogpt.data.demo import validate_demo_corpus_artifact
from tescogpt.retrieval.outcome import outcome_tier


def run_demo(
    message: str,
    corpus_path: str | Path,
    *,
    prior_context: str = "",
    system: str = "main-template",
    model: str | None = None,
    cache_dir: str | Path = "artifacts/cache/openai_drafts",
    corpus_manifest_path: str | Path | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Run one unlabelled message through retrieval, drafting, and policy gating."""
    clean_message = message.strip()
    clean_context = prior_context.strip()
    if not clean_message:
        raise ValueError("Demo message must not be blank")
    if system not in {"main-template", "main-openai"}:
        raise ValueError("Demo system must be main-template or main-openai")
    if system == "main-openai" and not model:
        raise ValueError("main-openai demo requires an explicit model ID")

    corpus_file = Path(corpus_path)
    if corpus_manifest_path is not None:
        validate_demo_corpus_artifact(corpus_file, corpus_manifest_path)
    corpus = pd.read_csv(corpus_file, dtype="string", keep_default_na=False)
    if "split" not in corpus.columns:
        raise ValueError("Demo corpus must contain an explicit split column")
    train = corpus.loc[corpus["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("Demo corpus contains no training rows")

    if system == "main-template":
        drafter = SafeTemplateDrafter()
    else:
        assert model is not None
        drafter = OpenAIDrafter(model, cache_dir, client=client)
    agent = EvidencePolicyAgent(train, drafter)
    input_digest = hashlib.sha256(
        f"{clean_context}\n{clean_message}".encode()
    ).hexdigest()[:12]
    prediction, retrieved = agent.predict_with_trace(
        {
            "case_id": f"demo-{input_digest}",
            "conversation_id": f"demo-conversation-{input_digest}",
            "message": clean_message,
            "prior_context": clean_context,
        }
    )

    indexed = train.set_index("case_id", drop=False)
    used_ids = set(prediction.evidence_case_ids)
    evidence = []
    for precedent in retrieved:
        row = indexed.loc[precedent.case_id]
        evidence.append(
            {
                "case_id": precedent.case_id,
                "used_by_drafter": precedent.case_id in used_ids,
                "historical_customer_message": str(row["message"]),
                "historical_tesco_reply": str(row["historical_reply"]),
                "visible_customer_followup": str(row["customer_followup"]),
                "outcome_tier": outcome_tier(row),
                "rerank_score": round(float(precedent.rerank_score), 6),
            }
        )

    return {
        "demo_schema_version": 1,
        "input": {
            "message": clean_message,
            "prior_context": clean_context,
        },
        "prediction": asdict(prediction),
        "retrieved_precedents": evidence,
        "notices": [
            "This is a demonstration, not a measured evaluation result.",
            "Historical replies are evidence of past handling, not current Tesco policy.",
            "The offline template displays retrieval but does not claim it used "
            "precedent text; main-openai records only evidence it actually used.",
            "AUTO_HANDLE authorizes only the displayed public draft; "
            "it performs no backend action.",
        ],
    }
