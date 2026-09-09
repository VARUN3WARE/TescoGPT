"""Validate assignment-level submission artifacts beyond metric correctness."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

_DECISION_PATTERN = re.compile(r"^(\d+)\.\s+\*\*", flags=re.MULTILINE)
_README_REQUIREMENTS = {
    "editable install command": "python -m pip install -e .",
    "headline reproduction command": "python -m tescogpt reproduce",
    "15-minute contract": "15-minute",
    "report link": "REPORT.md",
    "decision log link": "DECISIONS.md",
    "citation record link": "CITATIONS.md",
}
_CITATION_REQUIREMENTS = {
    "primary dataset citation": "kaggle.com/datasets/thoughtvector/customer-support-on-twitter",
    "AI-assistance disclosure": "AI coding assistant",
    "borrowed-method citation": "doi.org",
}


def _resolve(root: Path, value: str | None, default: str) -> Path:
    path = Path(value or default)
    return path if path.is_absolute() else root / path


def _read_text(path: Path, label: str, issues: list[str]) -> str:
    if not path.is_file():
        issues.append(f"Missing {label}: {path.name}")
        return ""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        issues.append(f"Empty {label}: {path.name}")
    return text


def validate_submission_package(
    root: str | Path,
    config: dict[str, Any],
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Check the repository-level artifacts named by the take-home brief."""
    project_root = Path(root)
    issues: list[str] = []
    readme_path = _resolve(project_root, config.get("readme_file"), "README.md")
    decisions_path = _resolve(
        project_root, config.get("decision_log_file"), "DECISIONS.md"
    )
    citations_path = _resolve(
        project_root, config.get("citations_file"), "CITATIONS.md"
    )
    registry_path = _resolve(
        project_root, config.get("registry_file"), "data/golden/golden_candidates.csv"
    )

    readme = _read_text(readme_path, "README", issues)
    for label, snippet in _README_REQUIREMENTS.items():
        if snippet not in readme:
            issues.append(f"README lacks {label}")

    decision_text = _read_text(decisions_path, "decision log", issues)
    decision_numbers = [int(value) for value in _DECISION_PATTERN.findall(decision_text)]
    decision_count = len(decision_numbers)
    if not 10 <= decision_count <= 15:
        issues.append("Decision log must contain 10 to 15 numbered decisions")
    if decision_numbers != list(range(1, decision_count + 1)):
        issues.append("Decision log numbering must be unique and consecutive from 1")

    citations = _read_text(citations_path, "citation record", issues)
    for label, snippet in _CITATION_REQUIREMENTS.items():
        if snippet.lower() not in citations.lower():
            issues.append(f"Citation record lacks {label}")

    expected_gold_count = int(config.get("expected_gold_count", 0))
    if not 150 <= expected_gold_count <= 250:
        issues.append("Configured golden-set size must be between 150 and 250")
    registry_count = 0
    if not registry_path.is_file():
        issues.append(f"Missing golden candidate registry: {registry_path.name}")
    else:
        registry = pd.read_csv(registry_path, dtype="string", keep_default_na=False)
        registry_count = len(registry)
        if "case_id" not in registry.columns:
            issues.append("Golden candidate registry lacks case_id")
        elif registry["case_id"].duplicated().any():
            issues.append("Golden candidate registry contains duplicate case IDs")
        if registry_count != expected_gold_count:
            issues.append("Golden candidate count differs from the configured count")

    result = {
        "submission_package_schema_version": 1,
        "is_complete": not issues,
        "issues": issues,
        "readme_file": readme_path.name,
        "decision_log_file": decisions_path.name,
        "decision_count": decision_count,
        "citations_file": citations_path.name,
        "registry_file": registry_path.name,
        "golden_candidate_count": registry_count,
        "configured_gold_count": expected_gold_count,
    }
    if require_complete and issues:
        raise ValueError("Submission package is incomplete: " + "; ".join(issues))
    return result
