"""Chronological conversation splits and deterministic golden-set sampling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tescogpt.data.cases import build_cases_from_csv
from tescogpt.evaluation.labels import LABEL_COLUMNS, validate_annotations

ANNOTATION_VISIBLE_COLUMNS = [
    "display_order",
    "case_id",
    "sample_slice",
    "conversation_id",
    "tweet_id",
    "created_at",
    "message",
    "prior_context",
    "challenge_flags",
    *LABEL_COLUMNS,
]

_CHALLENGE_PRIORITY = (
    "food_safety",
    "injury_or_allergy",
    "repeated_failure",
    "strong_distress",
    "money",
    "account_or_order",
    "image_or_link_dependent",
    "context_dependent",
    "short_message",
    "punctuation_heavy",
)


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _split_summary(conversation_frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_name, group in conversation_frame.groupby("split", sort=False):
        summary[str(split_name)] = {
            "conversation_count": len(group),
            "earliest_end_at": group["conversation_end_at"].min().isoformat(),
            "latest_end_at": group["conversation_end_at"].max().isoformat(),
        }
    return summary


def assign_chronological_splits(
    messages: pd.DataFrame,
    train_fraction: float = 0.70,
    dev_fraction: float = 0.15,
) -> tuple[pd.Series, dict[str, Any]]:
    """Assign complete conversations to train/dev/test in chronological order."""
    if train_fraction < 0 or dev_fraction < 0 or train_fraction + dev_fraction >= 1:
        raise ValueError("Split fractions must be non-negative and leave a non-empty test share")
    created = pd.to_datetime(
        messages["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        utc=True,
        errors="coerce",
    )
    if created.isna().any():
        raise ValueError("Every prepared message must have a parseable created_at timestamp")

    conversation_frame = (
        pd.DataFrame(
            {
                "conversation_id": messages["conversation_id"].astype("int64"),
                "created_at": created,
            }
        )
        .groupby("conversation_id", as_index=False)["created_at"]
        .max()
        .rename(columns={"created_at": "conversation_end_at"})
        .sort_values(["conversation_end_at", "conversation_id"], kind="stable")
        .reset_index(drop=True)
    )
    count = len(conversation_frame)
    train_end = int(np.floor(count * train_fraction))
    dev_end = train_end + int(np.floor(count * dev_fraction))
    conversation_frame["split"] = "test"
    conversation_frame.loc[: train_end - 1, "split"] = "train"
    conversation_frame.loc[train_end : dev_end - 1, "split"] = "dev"
    mapping = conversation_frame.set_index("conversation_id")["split"]
    assigned = messages["conversation_id"].map(mapping)
    if assigned.isna().any():
        raise AssertionError("Every message must inherit its conversation split")
    return assigned, {
        "train_fraction": train_fraction,
        "dev_fraction": dev_fraction,
        "test_fraction": round(1 - train_fraction - dev_fraction, 10),
        "splits": _split_summary(conversation_frame),
    }


def _natural_candidates(test_cases: pd.DataFrame) -> pd.DataFrame:
    timestamps = pd.to_datetime(
        test_cases["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        utc=True,
        errors="coerce",
    )
    ordered = test_cases.assign(_created_at=timestamps).sort_values(
        ["conversation_id", "_created_at", "tweet_id"], kind="stable"
    )
    return ordered.drop_duplicates("conversation_id", keep="first")


def _sample_challenge(
    test_cases: pd.DataFrame,
    excluded_conversations: set[int],
    count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    remaining = test_cases.loc[
        ~test_cases["conversation_id"].isin(excluded_conversations)
    ].copy()
    selections: list[int] = []
    selected_conversations: set[int] = set(excluded_conversations)
    candidates_by_flag: dict[str, list[int]] = {}
    for flag in _CHALLENGE_PRIORITY:
        indices = remaining.index[
            remaining["challenge_flags"]
            .fillna("")
            .str.split(";")
            .map(lambda values, wanted=flag: wanted in values)
        ].to_numpy()
        candidates_by_flag[flag] = list(rng.permutation(indices))

    while len(selections) < count:
        made_progress = False
        for flag in _CHALLENGE_PRIORITY:
            queue = candidates_by_flag[flag]
            while queue:
                index = int(queue.pop())
                conversation_id = int(remaining.at[index, "conversation_id"])
                if conversation_id in selected_conversations:
                    continue
                selections.append(index)
                selected_conversations.add(conversation_id)
                made_progress = True
                break
            if len(selections) == count:
                break
        if not made_progress:
            break

    if len(selections) < count:
        fallback = remaining.loc[
            ~remaining["conversation_id"].isin(selected_conversations)
            & remaining["challenge_flags"].fillna("").ne("")
        ]
        for index in rng.permutation(fallback.index.to_numpy()):
            conversation_id = int(remaining.at[index, "conversation_id"])
            if conversation_id in selected_conversations:
                continue
            selections.append(int(index))
            selected_conversations.add(conversation_id)
            if len(selections) == count:
                break

    if len(selections) < count:
        raise ValueError(
            f"Only {len(selections)} distinct challenge conversations available; requested {count}"
        )
    return remaining.loc[selections].copy()


def sample_golden_candidates(
    prepared_messages_path: str | Path,
    cases_output_path: str | Path,
    candidates_output_path: str | Path,
    manifest_output_path: str | Path,
    codebook_path: str | Path,
    natural_count: int = 150,
    challenge_count: int = 50,
    seed: int = 20260909,
    train_fraction: float = 0.70,
    dev_fraction: float = 0.15,
) -> dict[str, Any]:
    """Build cases, split conversations, and write an unlabelled blind sheet."""
    prepared_path = Path(prepared_messages_path)
    cases_path = Path(cases_output_path)
    candidates_path = Path(candidates_output_path)
    manifest_path = Path(manifest_output_path)
    codebook = Path(codebook_path)
    if natural_count < 0 or challenge_count < 0 or natural_count + challenge_count == 0:
        raise ValueError("At least one natural or challenge candidate is required")
    if not codebook.is_file():
        raise FileNotFoundError(f"Annotation codebook does not exist: {codebook}")

    cases = build_cases_from_csv(prepared_path, cases_path)
    messages = pd.read_csv(prepared_path)
    message_splits, split_manifest = assign_chronological_splits(
        messages,
        train_fraction=train_fraction,
        dev_fraction=dev_fraction,
    )
    conversation_splits = pd.DataFrame(
        {
            "conversation_id": messages["conversation_id"].astype("int64"),
            "split": message_splits,
        }
    ).drop_duplicates()
    split_map = conversation_splits.set_index("conversation_id")["split"]
    cases["split"] = cases["conversation_id"].map(split_map)
    cases.to_csv(cases_path, index=False, lineterminator="\n")

    test_cases = cases.loc[cases["split"].eq("test")].copy()
    natural_pool = _natural_candidates(test_cases)
    if len(natural_pool) < natural_count:
        raise ValueError(
            f"Only {len(natural_pool)} test conversations available; "
            f"requested {natural_count} natural"
        )
    rng = np.random.default_rng(seed)
    natural_indices = rng.choice(natural_pool.index.to_numpy(), size=natural_count, replace=False)
    natural = natural_pool.loc[natural_indices].copy()
    natural["sample_slice"] = "natural"
    natural_conversations = set(natural["conversation_id"].astype(int))

    challenge = _sample_challenge(
        test_cases,
        excluded_conversations=natural_conversations,
        count=challenge_count,
        rng=rng,
    )
    challenge["sample_slice"] = "challenge"
    selected = pd.concat([natural, challenge], ignore_index=True)
    visible_source_columns = [
        column for column in selected.columns if not column.startswith("_")
    ]
    selected = selected.loc[:, visible_source_columns]
    for column in LABEL_COLUMNS:
        selected[column] = ""
    selected["display_order"] = rng.permutation(np.arange(1, len(selected) + 1))
    selected = selected.sort_values("display_order", kind="stable")
    blind = selected.loc[:, ANNOTATION_VISIBLE_COLUMNS]

    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    blind.to_csv(candidates_path, index=False, lineterminator="\n")
    validation = validate_annotations(candidates_path)
    if validation["completed_count"] != 0:
        raise AssertionError("New candidate sheet must not contain labels")
    if blind["conversation_id"].duplicated().any():
        raise AssertionError("Golden candidates must use distinct conversations")

    challenge_counts: dict[str, int] = {}
    for value in challenge["challenge_flags"].fillna(""):
        for flag in value.split(";"):
            if flag:
                challenge_counts[flag] = challenge_counts.get(flag, 0) + 1

    manifest: dict[str, Any] = {
        "candidate_schema_version": 1,
        "label_status": "UNLABELED",
        "seed": seed,
        "prepared_messages_file": prepared_path.name,
        "prepared_messages_sha256": _sha256(prepared_path),
        "cases_file": cases_path.name,
        "cases_sha256": _sha256(cases_path),
        "codebook_file": codebook.name,
        "codebook_sha256": _sha256(codebook),
        "candidate_file": candidates_path.name,
        "candidate_sha256": _sha256(candidates_path),
        "eligible_case_count": len(cases),
        "eligible_test_case_count": len(test_cases),
        "natural_pool_conversation_count": len(natural_pool),
        "natural_count": natural_count,
        "challenge_count": challenge_count,
        "total_count": len(blind),
        "unique_conversation_count": int(blind["conversation_id"].nunique()),
        "challenge_flag_counts": dict(sorted(challenge_counts.items())),
        "chronological_split": split_manifest,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
