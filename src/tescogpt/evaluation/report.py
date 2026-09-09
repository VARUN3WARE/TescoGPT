"""Submission-report structure and readiness validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

STATUS_PATTERN = re.compile(r"<!--\s*SUBMISSION_STATUS:\s*(PENDING|READY)\s*-->")
REQUIRED_SECTION_PHRASES = (
    "problem framing",
    "results",
    "failure analysis",
    "what is misleading about my headline number?",
    "with one more week",
)
REQUIRED_BASELINE_PHRASES = ("constant baseline", "simple baseline")


def validate_submission_report(
    path: str | Path,
    *,
    max_words: int = 2400,
    require_ready: bool = False,
) -> dict[str, Any]:
    """Check the report's required sections, size guard, and explicit status."""
    source = Path(path)
    if max_words <= 0:
        raise ValueError("Report max_words must be positive")
    text = source.read_text(encoding="utf-8")
    statuses = STATUS_PATTERN.findall(text)
    if len(statuses) != 1:
        raise ValueError("Report must contain exactly one SUBMISSION_STATUS marker")
    headings = [
        line.lstrip("#").strip().lower()
        for line in text.splitlines()
        if line.startswith("## ")
    ]
    missing_sections = [
        phrase
        for phrase in REQUIRED_SECTION_PHRASES
        if not any(phrase in heading for heading in headings)
    ]
    if missing_sections:
        raise ValueError("Report is missing required sections: " + ", ".join(missing_sections))
    lowered = text.lower()
    missing_baselines = [
        phrase for phrase in REQUIRED_BASELINE_PHRASES if phrase not in lowered
    ]
    if missing_baselines:
        raise ValueError("Report is missing required baselines: " + ", ".join(missing_baselines))
    word_count = len(re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE))
    if word_count > max_words:
        raise ValueError(f"Report has {word_count} words, exceeding the {max_words}-word guard")
    declared_status = statuses[0]
    if require_ready and declared_status != "READY":
        raise ValueError("Report still declares SUBMISSION_STATUS: PENDING")
    return {
        "file": source.name,
        "declared_status": declared_status,
        "word_count": word_count,
        "max_words": max_words,
        "required_section_count": len(REQUIRED_SECTION_PHRASES),
        "is_submission_ready": declared_status == "READY",
    }
