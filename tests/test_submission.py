from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tescogpt.evaluation.submission import validate_submission_package


def _package(tmp_path: Path, *, decision_count: int = 10) -> dict[str, object]:
    (tmp_path / "README.md").write_text(
        """# Project

[Report](REPORT.md) · [Decisions](DECISIONS.md) · [Sources](CITATIONS.md)

Reproduce within the 15-minute contract:

    python -m pip install -e .
    python -m tescogpt reproduce
""",
        encoding="utf-8",
    )
    decisions = "# Decision log\n\n" + "\n".join(
        f"{index}. **Decision {index}.** Reason." for index in range(1, decision_count + 1)
    )
    (tmp_path / "DECISIONS.md").write_text(decisions + "\n", encoding="utf-8")
    (tmp_path / "CITATIONS.md").write_text(
        """# Sources

Customer Support on Twitter:
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

An AI coding assistant was used. Borrowed method: https://doi.org/example
""",
        encoding="utf-8",
    )
    registry = tmp_path / "data" / "golden" / "golden_candidates.csv"
    registry.parent.mkdir(parents=True)
    pd.DataFrame({"case_id": [f"case-{index}" for index in range(150)]}).to_csv(
        registry, index=False
    )
    return {
        "expected_gold_count": 150,
        "registry_file": "data/golden/golden_candidates.csv",
    }


def test_submission_package_covers_assignment_level_artifacts(tmp_path: Path) -> None:
    config = _package(tmp_path)

    result = validate_submission_package(tmp_path, config, require_complete=True)

    assert result["is_complete"]
    assert result["decision_count"] == 10
    assert result["golden_candidate_count"] == 150


def test_submission_package_rejects_false_ready_state(tmp_path: Path) -> None:
    config = _package(tmp_path, decision_count=16)
    (tmp_path / "CITATIONS.md").unlink()

    with pytest.raises(ValueError, match="Submission package is incomplete"):
        validate_submission_package(tmp_path, config, require_complete=True)

    result = validate_submission_package(tmp_path, config)
    assert not result["is_complete"]
    assert any("10 to 15" in issue for issue in result["issues"])
    assert any("Missing citation" in issue for issue in result["issues"])
