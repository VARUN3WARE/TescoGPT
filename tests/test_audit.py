from __future__ import annotations

import json
from pathlib import Path

from tescogpt.data.audit import audit_conversations
from tescogpt.data.threads import extract_brand_conversations

FIXTURE = Path(__file__).parent / "fixtures" / "twcs_mini.csv"


def test_audit_reports_structure_and_labels_proxies(tmp_path: Path) -> None:
    prepared = tmp_path / "tesco.csv"
    manifest = extract_brand_conversations(FIXTURE, prepared, brand="Tesco", chunk_size=3)
    manifest_path = prepared.with_suffix(".csv.manifest.json")
    output = tmp_path / "audit.json"
    markdown = tmp_path / "audit.md"

    audit = audit_conversations(prepared, output, markdown, manifest_path=manifest_path)

    assert audit["structure"]["message_count"] == 8
    assert audit["structure"]["conversation_count"] == 2
    assert audit["structure"]["max_thread_depth"] == 4
    assert audit["brand_reply_behavior"]["direct_reply_to_customer_count"] == 2
    assert audit["brand_reply_behavior"]["private_handoff_count"] == 1
    assert audit["customer_followup_proxies"]["followup_count"] == 3
    assert audit["customer_followup_proxies"]["positive_cue_count"] == 2
    assert audit["date_distribution"] == {"2017-11": 8}
    assert audit["provenance"]["source_sha256"] == manifest["source_sha256"]
    assert json.loads(output.read_text(encoding="utf-8")) == audit
    assert "Positive language is **not** a resolution label" in markdown.read_text(
        encoding="utf-8"
    )


def test_audit_rejects_wrong_brand(tmp_path: Path) -> None:
    prepared = tmp_path / "tesco.csv"
    extract_brand_conversations(FIXTURE, prepared, brand="Tesco")

    try:
        audit_conversations(
            prepared,
            tmp_path / "audit.json",
            tmp_path / "audit.md",
            brand="MissingBrand",
        )
    except ValueError as error:
        assert "contains no messages" in str(error)
    else:
        raise AssertionError("audit_conversations accepted a missing brand")
