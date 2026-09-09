"""Command-line entry points for reproducible project workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from tescogpt.data.audit import audit_conversations
from tescogpt.data.threads import extract_brand_conversations


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tescogpt",
        description="Build and evaluate the TescoGPT support agent.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Extract complete conversations involving one support brand.",
    )
    prepare.add_argument("--input", type=Path, required=True, help="Path to twcs.csv")
    prepare.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/tesco_messages.csv"),
        help="Destination for the long-form conversation table.",
    )
    prepare.add_argument("--brand", default="Tesco", help="Exact support author_id")
    prepare.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help="CSV rows read at once; reduce this on memory-constrained machines.",
    )

    audit = subparsers.add_parser(
        "audit",
        help="Profile a reconstructed brand conversation table.",
    )
    audit.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/tesco_messages.csv"),
        help="Long-form CSV created by the prepare command.",
    )
    audit.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional prepare manifest to embed as provenance.",
    )
    audit.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/data/tesco_audit.json"),
        help="Destination for machine-readable aggregate statistics.",
    )
    audit.add_argument(
        "--markdown",
        type=Path,
        default=Path("docs/TESCO_DATA_AUDIT.md"),
        help="Destination for the human-readable audit report.",
    )
    audit.add_argument("--brand", default="Tesco", help="Exact support author_id")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        manifest = extract_brand_conversations(
            input_path=args.input,
            output_path=args.output,
            brand=args.brand,
            chunk_size=args.chunk_size,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.command == "audit":
        audit = audit_conversations(
            input_path=args.input,
            output_path=args.output,
            markdown_path=args.markdown,
            brand=args.brand,
            manifest_path=args.manifest,
        )
        print(json.dumps(audit, indent=2, sort_keys=True))
        return

    parser.error(f"Unknown command: {args.command}")
