from pathlib import Path

import pandas as pd

from tescogpt.data.taxonomy_audit import audit_taxonomy_themes


def test_taxonomy_audit_is_training_only_overlapping_and_deterministic(
    tmp_path: Path,
) -> None:
    cases = tmp_path / "cases.csv"
    pd.DataFrame(
        [
            {"case_id": "train-1", "message": "My delivery is late", "split": "train"},
            {
                "case_id": "train-2",
                "message": "Thanks, the delivery arrived",
                "split": "train",
            },
            {
                "case_id": "train-3",
                "message": "Does this contain vegan ingredients?",
                "split": "train",
            },
            {"case_id": "test-1", "message": "I need a refund", "split": "test"},
        ]
    ).to_csv(cases, index=False)
    output = tmp_path / "audit.json"
    markdown = tmp_path / "audit.md"

    first = audit_taxonomy_themes(cases, output, markdown, seed=7, examples_per_theme=1)
    first_json = output.read_bytes()
    first_markdown = markdown.read_bytes()
    second = audit_taxonomy_themes(cases, output, markdown, seed=7, examples_per_theme=1)

    assert first["training_case_count"] == 3
    assert first["themes"]["delivery_or_collection"]["match_count"] == 2
    assert first["themes"]["feedback_praise_or_suggestion"]["match_count"] == 1
    assert first["themes"]["refund_return_or_exchange"]["match_count"] == 0
    assert first["multi_theme_case_count"] == 1
    assert first_json == output.read_bytes()
    assert first_markdown == markdown.read_bytes()
    assert first == second
    assert "not unsupervised discovery" in markdown.read_text(encoding="utf-8")
