"""Agreement statistics for annotation and reply-judge validation."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tescogpt.evaluation.labels import validate_annotations


def cohen_kappa(first: pd.Series, second: pd.Series) -> float:
    """Unweighted Cohen's kappa with explicit degenerate-case behavior."""
    if len(first) != len(second) or len(first) == 0:
        raise ValueError("Kappa inputs must be equally sized and non-empty")
    left = first.astype(str).reset_index(drop=True)
    right = second.astype(str).reset_index(drop=True)
    labels = sorted(set(left) | set(right))
    observed = float((left == right).mean())
    expected = sum(float((left == label).mean() * (right == label).mean()) for label in labels)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def weighted_kappa(
    first: pd.Series,
    second: pd.Series,
    *,
    categories: tuple[int, ...] = (0, 1, 2),
) -> float:
    """Quadratic-weighted kappa for ordinal rubric scores."""
    if len(first) != len(second) or len(first) == 0:
        raise ValueError("Kappa inputs must be equally sized and non-empty")
    left = pd.to_numeric(first, errors="raise").astype(int)
    right = pd.to_numeric(second, errors="raise").astype(int)
    invalid = (set(left) | set(right)) - set(categories)
    if invalid:
        raise ValueError(f"Unknown ordinal categories: {sorted(invalid)}")
    size = len(categories)
    lookup = {value: index for index, value in enumerate(categories)}
    observed = np.zeros((size, size), dtype=float)
    for left_value, right_value in zip(left, right, strict=True):
        observed[lookup[left_value], lookup[right_value]] += 1
    observed /= len(left)
    left_histogram = observed.sum(axis=1)
    right_histogram = observed.sum(axis=0)
    expected = np.outer(left_histogram, right_histogram)
    denominator_scale = max(1, (size - 1) ** 2)
    weights = np.fromfunction(
        lambda i, j: (i - j) ** 2 / denominator_scale,
        (size, size),
    )
    observed_disagreement = float((weights * observed).sum())
    expected_disagreement = float((weights * expected).sum())
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else 0.0
    return 1 - observed_disagreement / expected_disagreement


def _tag_set(value: str) -> set[str]:
    return {item.strip() for item in str(value).split(";") if item.strip()}


def annotation_agreement(
    round_one_path: str | Path,
    round_two_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Compare independent labels on the frozen overlap before adjudication."""
    validate_annotations(round_one_path, require_complete=True, require_blind=True)
    validate_annotations(round_two_path, require_complete=True, require_blind=True)
    first = pd.read_csv(round_one_path, dtype="string", keep_default_na=False)
    second = pd.read_csv(round_two_path, dtype="string", keep_default_na=False)
    if not set(second["case_id"]).issubset(set(first["case_id"])):
        raise ValueError("Round two must be a subset of round one")
    merged = first.merge(second, on="case_id", suffixes=("_one", "_two"), validate="one_to_one")
    categorical = ("intent_label", "handling_label", "reason_code")
    fields = {
        field: {
            "raw_agreement": float((merged[f"{field}_one"] == merged[f"{field}_two"]).mean()),
            "cohen_kappa": cohen_kappa(
                merged[f"{field}_one"], merged[f"{field}_two"]
            ),
        }
        for field in categorical
    }
    risk_jaccard = []
    for left, right in zip(merged["risk_tags_one"], merged["risk_tags_two"], strict=True):
        left_tags = _tag_set(left)
        right_tags = _tag_set(right)
        union = left_tags | right_tags
        risk_jaccard.append(len(left_tags & right_tags) / len(union) if union else 1.0)
    report = {
        "annotation_agreement_schema_version": 1,
        "overlap_count": len(merged),
        "fields": fields,
        "risk_tag_mean_jaccard": float(np.mean(risk_jaccard)),
        "disagreement_case_ids": sorted(
            merged.loc[
                (
                    merged["intent_label_one"].ne(merged["intent_label_two"])
                    | merged["handling_label_one"].ne(merged["handling_label_two"])
                    | merged["reason_code_one"].ne(merged["reason_code_two"])
                ),
                "case_id",
            ].astype(str)
        ),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def pairwise_repeatability(
    rating_frames: dict[str, pd.DataFrame],
    dimensions: tuple[str, ...],
) -> dict[str, Any]:
    """Compare repeated judge runs without averaging away instability."""
    results: dict[str, Any] = {}
    for (left_name, left), (right_name, right) in combinations(rating_frames.items(), 2):
        merged = left.merge(
            right,
            on="review_id",
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        results[f"{left_name}__vs__{right_name}"] = {
            dimension: {
                "exact_agreement": float(
                    (merged[f"{dimension}_left"] == merged[f"{dimension}_right"]).mean()
                ),
                "quadratic_weighted_kappa": weighted_kappa(
                    merged[f"{dimension}_left"], merged[f"{dimension}_right"]
                ),
            }
            for dimension in dimensions
        }
    return results
