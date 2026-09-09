"""Training-only lexical audit supporting the Tesco intent-taxonomy rationale."""

from __future__ import annotations

import hashlib
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

THEME_PATTERNS: dict[str, tuple[str, ...]] = {
    "delivery_or_collection": (
        r"\bdeliver(?:y|ed|ies)?\b",
        r"\bclick\s*(?:&|and)?\s*collect\b",
        r"\bdelivery slot\b",
        r"\bsubstitut(?:e|ed|ion)s?\b",
        r"\bmissing (?:bag|item|order)s?\b",
    ),
    "product_quality_or_safety": (
        r"\b(?:ill|sick|poison(?:ed|ing)?)\b",
        r"\b(?:mould|mold|rotten|raw|stale)\b",
        r"\b(?:glass|bone|foreign object)\b",
        r"\b(?:damag(?:e|ed)|broken|leak(?:ing|ed)?)\b",
        r"\b(?:allergic reaction|chok(?:e|ed|ing)|injur(?:y|ed))\b",
    ),
    "product_availability": (
        r"\b(?:in|out of) stock\b",
        r"\b(?:stock|sell|sold|available|availability)\b",
        r"\bdiscontinu(?:e|ed)\b",
        r"\bbring (?:it|them|this|these)?\s*back\b",
    ),
    "pricing_promotion_or_clubcard": (
        r"\bclubcard\b",
        r"\b(?:price|priced|charged|cost)\b",
        r"\b(?:offer|promotion|promo|discount)\b",
        r"\b(?:voucher|coupon|points|gift card)\b",
    ),
    "refund_return_or_exchange": (
        r"\brefund(?:ed|ing|s)?\b",
        r"\breturn(?:ed|ing|s)?\b",
        r"\breplac(?:e|ed|ement|ing)s?\b",
        r"\bexchange(?:d|s)?\b",
        r"\bmoney back\b",
    ),
    "online_account_or_checkout": (
        r"\b(?:login|log in|password|account)\b",
        r"\b(?:website|web site|app|online)\b",
        r"\b(?:checkout|check out|basket|payment)\b",
        r"\b(?:error|signs? me out|crash(?:ed|es|ing)?)\b",
    ),
    "store_or_staff_experience": (
        r"\b(?:staff|cashier|colleague|manager)\b",
        r"\b(?:queue|trolley|parking|toilet|cleanliness)\b",
        r"\b(?:store|shop|supermarket)\b",
        r"\bopening (?:hour|time)s?\b",
    ),
    "product_information": (
        r"\b(?:ingredient|nutrition|calorie)s?\b",
        r"\b(?:vegan|vegetarian|allergen|allergy)\b",
        r"\b(?:contain|suitable for|free from)\b",
        r"\b(?:weight|size|cook(?:ing)?|prepar(?:e|ation))\b",
    ),
    "feedback_praise_or_suggestion": (
        r"\b(?:thank you|thanks|thx)\b",
        r"\b(?:love|great|brilliant|amazing|excellent)\b",
        r"\b(?:suggestion|feedback|well done)\b",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _example_order(case_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest()


def _markdown_text(value: str) -> str:
    return " ".join(str(value).split()).replace("|", "\\|")


def audit_taxonomy_themes(
    input_path: str | Path,
    output_path: str | Path,
    markdown_path: str | Path,
    *,
    seed: int = 20260913,
    examples_per_theme: int = 3,
) -> dict[str, Any]:
    """Count overlapping lexical issue themes on training cases only."""
    if examples_per_theme <= 0:
        raise ValueError("examples_per_theme must be positive")
    source = Path(input_path)
    cases = pd.read_csv(source, dtype="string", keep_default_na=False)
    required = {"case_id", "message", "split"}
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ValueError("Taxonomy audit input is missing columns: " + ", ".join(missing))
    training = cases.loc[cases["split"].eq("train"), ["case_id", "message"]].copy()
    if training.empty:
        raise ValueError("Taxonomy audit requires at least one training case")
    if training["case_id"].duplicated().any():
        raise ValueError("Taxonomy audit input contains duplicate training case IDs")

    theme_hits = pd.DataFrame(index=training.index)
    for theme, fragments in THEME_PATTERNS.items():
        pattern = re.compile("|".join(f"(?:{fragment})" for fragment in fragments), re.I)
        theme_hits[theme] = training["message"].str.contains(pattern, na=False)
    training["matched_theme_count"] = theme_hits.sum(axis=1)

    themes: dict[str, Any] = {}
    for theme, fragments in THEME_PATTERNS.items():
        matched = training.loc[theme_hits[theme]].copy()
        matched["_example_order"] = matched["case_id"].map(
            lambda case_id: _example_order(str(case_id), seed)
        )
        examples = (
            matched.sort_values("_example_order", kind="stable")
            .head(examples_per_theme)
            .loc[:, ["case_id", "message"]]
            .to_dict(orient="records")
        )
        themes[theme] = {
            "match_count": len(matched),
            "match_share": len(matched) / len(training),
            "patterns": list(fragments),
            "examples": examples,
        }

    overlap_rows = []
    for left, right in combinations(THEME_PATTERNS, 2):
        count = int((theme_hits[left] & theme_hits[right]).sum())
        if count:
            overlap_rows.append({"left_theme": left, "right_theme": right, "count": count})
    overlap_rows.sort(key=lambda row: (-row["count"], row["left_theme"], row["right_theme"]))
    audit = {
        "taxonomy_audit_schema_version": 1,
        "scope": "training cases only; lexical diagnostics, not labels",
        "input_file": source.name,
        "input_sha256": _sha256(source),
        "seed": seed,
        "training_case_count": len(training),
        "theme_count": len(THEME_PATTERNS),
        "unmatched_case_count": int(training["matched_theme_count"].eq(0).sum()),
        "multi_theme_case_count": int(training["matched_theme_count"].gt(1).sum()),
        "themes": themes,
        "top_theme_overlaps": overlap_rows[:15],
        "important_limitation": (
            "Patterns make broad issue families inspectable but are neither discovered "
            "clusters nor gold labels. Counts overlap and do not estimate intent accuracy."
        ),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    lines = [
        "# Tesco intent-taxonomy derivation audit",
        "",
        "> Generated from training-period cases only. This is an overlapping lexical ",
        "> coverage audit, not unsupervised discovery and not human ground truth.",
        "",
        "## Provenance",
        "",
        f"- Input: `{source.name}`",
        f"- Input SHA-256: `{audit['input_sha256']}`",
        f"- Deterministic example seed: `{seed}`",
        "",
        "## Why these intent boundaries",
        "",
        "The taxonomy uses operational support actions rather than product categories. "
        "Closely related surface forms are consolidated when they need the same queue or "
        "public response: pricing, promotions and Clubcard form one family; login, app and "
        "checkout failures form another. Product defects and safety reports share one intent, "
        "while risk tags and escalation reasons preserve the safety distinction. Standalone "
        "refund process questions remain separate, but a refund caused by delivery or a defect "
        "is secondary to that underlying issue. `other_or_unclear` is the residual, not a "
        "dictionary theme.",
        "",
        "## Training-data coverage",
        "",
        f"Training cases: **{len(training):,}**. Cases matching no theme: "
        f"**{audit['unmatched_case_count']:,}**. Cases matching multiple themes: "
        f"**{audit['multi_theme_case_count']:,}**.",
        "",
        "| Candidate intent family | Lexical matches | Share of training cases |",
        "|---|---:|---:|",
    ]
    for theme, result in themes.items():
        lines.append(
            f"| `{theme}` | {result['match_count']:,} | {100 * result['match_share']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Counts overlap. High overlap is evidence for explicit tie-break rules, not a reason "
            "to report these matches as classifier performance.",
            "",
            "## Deterministic examples",
        ]
    )
    for theme, result in themes.items():
        lines.extend(["", f"### `{theme}`", ""])
        for example in result["examples"]:
            lines.append(
                f"- `{example['case_id']}` — {_markdown_text(str(example['message']))}"
            )
    lines.extend(
        [
            "",
            "## What this does not prove",
            "",
            "These dictionaries were written to audit chosen operational families; they do not "
            "claim the taxonomy emerged automatically. Polysemy, negation, multi-intent cases "
            "and polite words create false matches. The blind 200-case human annotation is the "
            "only source of evaluation labels, and its disagreement analysis is the test of "
            "whether these boundaries are usable.",
        ]
    )
    report_path = Path(markdown_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return audit
