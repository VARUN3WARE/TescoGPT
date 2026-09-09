from pathlib import Path

import pytest

from tescogpt.evaluation.report import validate_submission_report


def _report(path: Path, status: str = "PENDING") -> Path:
    path.write_text(
        f"""# Test report

<!-- SUBMISSION_STATUS: {status} -->

## Problem framing

Scope.

## Results

The constant baseline and simple baseline are both reported.

## Failure analysis

Five evidence-backed failures.

## What is misleading about my headline number?

Limitations.

## With one more week

Next steps.
""",
        encoding="utf-8",
    )
    return path


def test_report_requires_explicit_ready_marker_for_final_run(tmp_path: Path) -> None:
    path = _report(tmp_path / "REPORT.md")

    status = validate_submission_report(path)

    assert status["declared_status"] == "PENDING"
    assert not status["is_submission_ready"]
    with pytest.raises(ValueError, match="still declares"):
        validate_submission_report(path, require_ready=True)


def test_ready_report_enforces_required_sections_and_size(tmp_path: Path) -> None:
    path = _report(tmp_path / "REPORT.md", status="READY")
    assert validate_submission_report(path, require_ready=True)["is_submission_ready"]

    path.write_text(path.read_text(encoding="utf-8") + " word" * 20, encoding="utf-8")
    with pytest.raises(ValueError, match="exceeding"):
        validate_submission_report(path, max_words=10)
