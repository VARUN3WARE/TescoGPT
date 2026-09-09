"""Small dependency-free BM25 implementation for the sparse baseline."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "can",
    "customer",
    "do",
    "get",
    "for",
    "from",
    "i",
    "have",
    "help",
    "hi",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "sorry",
    "tesco",
    "that",
    "the",
    "this",
    "thanks",
    "to",
    "was",
    "with",
    "you",
    "your",
}


def tokenize(text: str) -> list[str]:
    """Tokenize support text while discarding brand boilerplate."""
    return [
        token
        for token in _TOKEN.findall(str(text).lower())
        if token not in _STOPWORDS and not token.isdigit()
    ]


@dataclass(frozen=True)
class SearchResult:
    case_id: str
    conversation_id: str
    score: float
    message: str
    historical_reply: str


class BM25Index:
    """In-memory BM25 index with deterministic ranking and exclusions."""

    def __init__(
        self,
        corpus: pd.DataFrame,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        required = {"case_id", "conversation_id", "message", "historical_reply"}
        missing = sorted(required - set(corpus.columns))
        if missing:
            raise ValueError(f"Retrieval corpus is missing columns: {', '.join(missing)}")
        if corpus.empty:
            raise ValueError("Retrieval corpus must not be empty")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")

        usable = corpus.loc[corpus["historical_reply"].fillna("").str.strip().ne("")].copy()
        if usable.empty:
            raise ValueError("Retrieval corpus contains no historical replies")
        usable["case_id"] = usable["case_id"].astype(str)
        usable["conversation_id"] = usable["conversation_id"].astype(str)
        usable = usable.sort_values("case_id", kind="stable").reset_index(drop=True)

        self._corpus = usable
        self._case_ids = usable["case_id"].tolist()
        self._conversation_ids = usable["conversation_id"].tolist()
        self._messages = usable["message"].astype(str).tolist()
        self._replies = usable["historical_reply"].astype(str).tolist()
        self._k1 = k1
        self._b = b
        self._term_counts = [Counter(tokenize(value)) for value in usable["message"]]
        self._lengths = [sum(counts.values()) for counts in self._term_counts]
        self._average_length = sum(self._lengths) / len(self._lengths) or 1.0
        document_frequency: Counter[str] = Counter()
        postings: dict[str, list[int]] = defaultdict(list)
        for index, counts in enumerate(self._term_counts):
            document_frequency.update(counts.keys())
            for term in counts:
                postings[term].append(index)
        size = len(self._term_counts)
        self._idf = {
            term: math.log(1 + (size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        self._postings = dict(postings)

    def search(
        self,
        query: str,
        *,
        top_k: int = 1,
        exclude_case_id: str | None = None,
        exclude_conversation_id: str | int | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_terms = set(tokenize(query))
        excluded_conversation = (
            str(exclude_conversation_id) if exclude_conversation_id is not None else None
        )
        candidate_indices = sorted(
            {
                index
                for term in query_terms
                for index in self._postings.get(term, ())
            }
        )
        ranked: list[tuple[float, str, int]] = []
        for index in candidate_indices:
            counts = self._term_counts[index]
            case_id = self._case_ids[index]
            conversation_id = self._conversation_ids[index]
            if exclude_case_id is not None and case_id == str(exclude_case_id):
                continue
            if excluded_conversation is not None and conversation_id == excluded_conversation:
                continue
            length_norm = 1 - self._b + self._b * self._lengths[index] / self._average_length
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                score += self._idf.get(term, 0.0) * (
                    frequency * (self._k1 + 1)
                    / (frequency + self._k1 * length_norm)
                )
            if score > 0:
                ranked.append((score, case_id, index))
        ranked.sort(key=lambda item: (-item[0], item[1]))

        results: list[SearchResult] = []
        for score, _, index in ranked[:top_k]:
            results.append(
                SearchResult(
                    case_id=self._case_ids[index],
                    conversation_id=self._conversation_ids[index],
                    score=score,
                    message=self._messages[index],
                    historical_reply=self._replies[index],
                )
            )
        return results
