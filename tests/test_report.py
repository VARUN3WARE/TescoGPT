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

The constant baseline and simple baseline are both reported. Aggregate claims
come from `outputs/evaluation/evidence_summary.md`.

## Failure analysis

1. `case-1` demonstrates mode one. Hypothesis: improve one.
2. `case-2` demonstrates mode two. Hypothesis: improve two.
3. `case-3` demonstrates mode three. Hypothesis: improve three.
4. `case-4` demonstrates mode four. Hypothesis: improve four.
5. `case-5` demonstrates mode five. Hypothesis: improve five.

## What is misleading about my headline number?

Limitations.

## With one more week

Next steps.
""",
        encoding="utf-8",
    )
    return path


def _registry(path: Path) -> Path:
    path.write_text(
        "case_id,message\n" + "".join(f"case-{index},message {index}\n" for index in range(1, 6)),
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
    registry = _registry(tmp_path / "registry.csv")
    status = validate_submission_report(
        path,
        require_ready=True,
        candidate_registry_path=registry,
    )
    assert status["is_submission_ready"]
    assert status["verified_failure_mode_count"] == 5

    path.write_text(path.read_text(encoding="utf-8") + " word" * 20, encoding="utf-8")
    with pytest.raises(ValueError, match="exceeding"):
        validate_submission_report(path, max_words=10)


def test_ready_report_rejects_unresolved_result_language(tmp_path: Path) -> None:
    path = _report(tmp_path / "REPORT.md", status="READY")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "Aggregate claims", "Results are pending. Aggregate claims"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unresolved language"):
        validate_submission_report(path, require_ready=True)


def test_ready_report_requires_traceable_failure_examples(tmp_path: Path) -> None:
    path = _report(tmp_path / "REPORT.md", status="READY")
    registry = _registry(tmp_path / "registry.csv")
    path.write_text(
        path.read_text(encoding="utf-8").replace("`case-3`", "`unknown-case`"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="failure mode 3"):
        validate_submission_report(path, candidate_registry_path=registry)
