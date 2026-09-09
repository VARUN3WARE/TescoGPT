"""Submission-report structure and readiness validation."""

from __future__ import annotations

import csv
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
REQUIRED_RESULTS_SOURCE = "evidence_summary.md"
UNRESOLVED_READY_PATTERNS = (
    ("placeholder token", re.compile(r"\b(?:pending|tbd|todo|fixme|placeholder)\b", re.I)),
    ("awaiting work", re.compile(r"\bawait(?:s|ed|ing)?\b", re.I)),
    (
        "future result work",
        re.compile(
            r"\b(?:will|must) (?:be )?"
            r"(?:generat|report|replac|select|adjudicat|collect|compar|run)\w*\b",
            re.I,
        ),
    ),
    (
        "unavailable result",
        re.compile(
            r"\b(?:results?|ratings?|labels?|agreement|metrics?) "
            r"(?:are|is) not (?:yet )?available\b",
            re.I,
        ),
    ),
)


def _section(text: str, phrase: str) -> str:
    """Return one level-two Markdown section whose heading contains ``phrase``."""
    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", text))
    for index, match in enumerate(matches):
        if phrase in match.group(1).lower():
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            return text[match.end() : end]
    return ""


def _numbered_items(section: str) -> list[tuple[int, str]]:
    starts = list(re.finditer(r"(?m)^\s*(\d+)\.\s+", section))
    return [
        (
            int(match.group(1)),
            section[match.end() : starts[index + 1].start()]
            if index + 1 < len(starts)
            else section[match.end() :],
        )
        for index, match in enumerate(starts)
    ]


def _registry_case_ids(path: str | Path) -> set[str]:
    source = Path(path)
    with source.open(encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if reader.fieldnames is None or "case_id" not in reader.fieldnames:
            raise ValueError("Golden registry must contain a case_id column")
        case_ids = {
            value
            for row in reader
            if (value := str(row.get("case_id") or "").strip())
        }
    if not case_ids:
        raise ValueError("Golden registry contains no case IDs")
    return case_ids


def _validate_ready_content(text: str, candidate_registry_path: str | Path | None) -> int:
    unresolved = [label for label, pattern in UNRESOLVED_READY_PATTERNS if pattern.search(text)]
    if unresolved:
        raise ValueError(
            "READY report contains unresolved language: " + ", ".join(unresolved)
        )

    results = _section(text, "results")
    missing_results_baselines = [
        phrase for phrase in REQUIRED_BASELINE_PHRASES if phrase not in results.lower()
    ]
    if missing_results_baselines:
        raise ValueError(
            "READY report Results section is missing baselines: "
            + ", ".join(missing_results_baselines)
        )
    if REQUIRED_RESULTS_SOURCE not in results:
        raise ValueError(
            "READY report Results section must cite outputs/evaluation/evidence_summary.md"
        )

    failures = _numbered_items(_section(text, "failure analysis"))
    if [number for number, _ in failures[:5]] != [1, 2, 3, 4, 5]:
        raise ValueError("READY report must contain five consecutively numbered failure modes")
    for number, body in failures[:5]:
        if "hypothesis" not in body.lower():
            raise ValueError(f"READY report failure mode {number} has no hypothesis")

    if candidate_registry_path is not None:
        case_ids = _registry_case_ids(candidate_registry_path)
        for number, body in failures[:5]:
            if not any(case_id in body for case_id in case_ids):
                raise ValueError(
                    f"READY report failure mode {number} has no example from the golden registry"
                )
    return 5


def validate_submission_report(
    path: str | Path,
    *,
    max_words: int = 2400,
    require_ready: bool = False,
    candidate_registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Check report structure and prevent a stale draft from declaring readiness."""
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
    verified_failure_modes = 0
    if declared_status == "READY":
        verified_failure_modes = _validate_ready_content(text, candidate_registry_path)
    return {
        "file": source.name,
        "declared_status": declared_status,
        "word_count": word_count,
        "max_words": max_words,
        "required_section_count": len(REQUIRED_SECTION_PHRASES),
        "verified_failure_mode_count": verified_failure_modes,
        "is_submission_ready": declared_status == "READY",
    }
