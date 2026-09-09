"""Reconstruct complete brand conversations from the TWCS reply graph.

The source file is too large to assume it fits comfortably in laptop memory.
This module performs two chunked CSV passes while retaining only compact NumPy
arrays for graph structure. It selects an entire rooted reply tree whenever any
node in that tree was authored by the requested support brand.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STRUCTURE_COLUMNS = ["tweet_id", "author_id", "in_response_to_tweet_id"]
OUTPUT_SOURCE_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]
OUTPUT_COLUMNS = [
    "conversation_id",
    "thread_depth",
    "tweet_id",
    "parent_tweet_id",
    "response_tweet_id",
    "author_id",
    "author_role",
    "inbound",
    "created_at",
    "text",
]
OUTPUT_DTYPES = {
    "tweet_id": "int64",
    "author_id": "string",
    "inbound": "boolean",
    "created_at": "string",
    "text": "string",
    "response_tweet_id": "string",
    "in_response_to_tweet_id": "Int64",
}

_MISSING_PARENT = -1
_INITIAL_CAPACITY = 1_000_000
_MAX_POINTER_JUMPS = 64
_MAX_THREAD_DEPTH = 4096


class ConversationDataError(ValueError):
    """Raised when the reply graph violates reconstruction assumptions."""


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _grow(array: np.ndarray, capacity: int, fill_value: int | bool) -> np.ndarray:
    grown = np.full(capacity, fill_value, dtype=array.dtype)
    grown[: len(array)] = array
    return grown


def _next_capacity(current: int, required_index: int) -> int:
    capacity = current
    while capacity <= required_index:
        capacity *= 2
    return capacity


def _load_graph(
    input_path: Path,
    brand: str,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    parent = np.full(_INITIAL_CAPACITY, _MISSING_PARENT, dtype=np.int64)
    exists = np.zeros(_INITIAL_CAPACITY, dtype=np.bool_)
    is_brand = np.zeros(_INITIAL_CAPACITY, dtype=np.bool_)
    row_count = 0

    for chunk in pd.read_csv(
        input_path,
        usecols=STRUCTURE_COLUMNS,
        chunksize=chunk_size,
        dtype={"tweet_id": "int64", "author_id": "string"},
    ):
        if chunk.empty:
            continue

        tweet_ids = chunk["tweet_id"].to_numpy(dtype=np.int64)
        if (tweet_ids < 0).any():
            raise ConversationDataError("tweet_id values must be non-negative")
        if len(np.unique(tweet_ids)) != len(tweet_ids):
            raise ConversationDataError("Duplicate tweet_id found within one CSV chunk")

        required_index = int(tweet_ids.max())
        if required_index >= len(parent):
            capacity = _next_capacity(len(parent), required_index)
            parent = _grow(parent, capacity, _MISSING_PARENT)
            exists = _grow(exists, capacity, False)
            is_brand = _grow(is_brand, capacity, False)

        if exists[tweet_ids].any():
            raise ConversationDataError("Duplicate tweet_id found across CSV chunks")

        parent_values = (
            pd.to_numeric(chunk["in_response_to_tweet_id"], errors="coerce")
            .fillna(_MISSING_PARENT)
            .to_numpy(dtype=np.int64)
        )
        parent[tweet_ids] = parent_values
        exists[tweet_ids] = True
        is_brand[tweet_ids] = chunk["author_id"].eq(brand).fillna(False).to_numpy()
        row_count += len(chunk)

    if not is_brand.any():
        raise ConversationDataError(f"No messages authored by brand {brand!r} were found")

    return parent, exists, is_brand, row_count


def _compute_roots(parent: np.ndarray, exists: np.ndarray) -> np.ndarray:
    """Return the first available ancestor for every possible tweet ID."""
    ancestors = np.arange(len(parent), dtype=np.int64)
    nodes = np.flatnonzero(exists)
    candidate_parents = parent[nodes]
    valid_parent = (candidate_parents >= 0) & (candidate_parents < len(parent))

    step_nodes = nodes[valid_parent]
    step_parents = candidate_parents[valid_parent]
    parent_exists = exists[step_parents]
    ancestors[step_nodes[parent_exists]] = step_parents[parent_exists]
    has_existing_parent = np.zeros(len(parent), dtype=np.bool_)
    has_existing_parent[step_nodes[parent_exists]] = True

    for _ in range(_MAX_POINTER_JUMPS):
        jumped = ancestors[ancestors]
        if np.array_equal(jumped, ancestors):
            cycle_node = exists & has_existing_parent & (ancestors == np.arange(len(parent)))
            if cycle_node.any():
                raise ConversationDataError("Reply graph contains a parent cycle")
            return ancestors
        ancestors = jumped

    raise ConversationDataError(
        "Reply graph did not converge to roots; it may contain a parent cycle"
    )


def _compute_depths(
    selected_ids: np.ndarray,
    parent: np.ndarray,
    exists: np.ndarray,
) -> np.ndarray:
    current = selected_ids.copy()
    depths = np.zeros(len(selected_ids), dtype=np.int32)
    active = np.ones(len(selected_ids), dtype=np.bool_)

    for _ in range(_MAX_THREAD_DEPTH):
        active_indices = np.flatnonzero(active)
        if len(active_indices) == 0:
            return depths

        candidate_parents = parent[current[active_indices]]
        valid = (candidate_parents >= 0) & (candidate_parents < len(parent))
        valid_indices = active_indices[valid]
        valid_parents = candidate_parents[valid]
        parent_exists = exists[valid_parents]

        advancing = valid_indices[parent_exists]
        stopped = np.ones(len(active_indices), dtype=np.bool_)
        stopped[np.flatnonzero(valid)[parent_exists]] = False
        active[active_indices[stopped]] = False

        if len(advancing):
            current[advancing] = valid_parents[parent_exists]
            depths[advancing] += 1

    raise ConversationDataError(
        f"Conversation depth exceeded the {_MAX_THREAD_DEPTH}-message traversal limit"
    )


def _format_messages(
    messages: pd.DataFrame,
    roots: np.ndarray,
    depths_by_id: np.ndarray,
    brand: str,
) -> pd.DataFrame:
    tweet_ids = messages["tweet_id"].to_numpy(dtype=np.int64)
    parent_ids = (
        pd.to_numeric(messages["in_response_to_tweet_id"], errors="coerce")
        .fillna(_MISSING_PARENT)
        .astype("int64")
    )

    formatted = pd.DataFrame(
        {
            "conversation_id": roots[tweet_ids],
            "thread_depth": depths_by_id[tweet_ids],
            "tweet_id": tweet_ids,
            "parent_tweet_id": parent_ids,
            "response_tweet_id": messages["response_tweet_id"].fillna(""),
            "author_id": messages["author_id"].fillna(""),
            "author_role": np.where(messages["author_id"].eq(brand), "brand", "customer_or_other"),
            "inbound": messages["inbound"].astype(bool),
            "created_at": messages["created_at"].fillna(""),
            "text": messages["text"].fillna(""),
        }
    )
    formatted["_created_at_sort"] = pd.to_datetime(
        formatted["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        utc=True,
        errors="coerce",
    )
    formatted = formatted.sort_values(
        ["conversation_id", "_created_at_sort", "thread_depth", "tweet_id"],
        kind="stable",
        na_position="last",
    ).drop(columns="_created_at_sort")
    return formatted[OUTPUT_COLUMNS]


def extract_brand_conversations(
    input_path: str | Path,
    output_path: str | Path,
    brand: str = "Tesco",
    chunk_size: int = 250_000,
) -> dict[str, Any]:
    """Extract full rooted reply trees containing at least one brand message.

    The output is a long-form CSV with graph identifiers. A deterministic JSON
    manifest is written next to it as ``<output>.manifest.json``.
    """
    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {source}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not brand.strip():
        raise ValueError("brand must not be empty")

    parent, exists, is_brand, source_rows = _load_graph(source, brand, chunk_size)
    roots = _compute_roots(parent, exists)
    brand_roots = np.unique(roots[np.flatnonzero(is_brand)])
    selected_mask = exists & np.isin(roots, brand_roots)
    selected_ids = np.flatnonzero(selected_mask)
    selected_lookup = np.zeros(len(exists), dtype=np.bool_)
    selected_lookup[selected_ids] = True

    selected_depths = _compute_depths(selected_ids, parent, exists)
    depths_by_id = np.zeros(len(exists), dtype=np.int32)
    depths_by_id[selected_ids] = selected_depths

    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        source,
        usecols=OUTPUT_SOURCE_COLUMNS,
        chunksize=chunk_size,
        dtype=OUTPUT_DTYPES,
    ):
        ids = chunk["tweet_id"].to_numpy(dtype=np.int64)
        keep = (ids < len(selected_lookup)) & selected_lookup[ids]
        if keep.any():
            frames.append(chunk.loc[keep].copy())

    if not frames:
        raise ConversationDataError("Brand roots were found but no output messages were selected")

    messages = pd.concat(frames, ignore_index=True)
    output = _format_messages(messages, roots, depths_by_id, brand)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False, lineterminator="\n")

    missing_parent_count = int(
        (
            (output["parent_tweet_id"] >= 0)
            & ~output["parent_tweet_id"].isin(output["tweet_id"])
        ).sum()
    )
    manifest: dict[str, Any] = {
        "brand": brand,
        "source_file": source.name,
        "source_rows": source_rows,
        "source_sha256": _sha256(source),
        "output_file": destination.name,
        "output_rows": len(output),
        "output_sha256": _sha256(destination),
        "conversation_count": int(output["conversation_id"].nunique()),
        "brand_message_count": int((output["author_role"] == "brand").sum()),
        "missing_parent_count": missing_parent_count,
        "max_thread_depth": int(output["thread_depth"].max()),
    }
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
