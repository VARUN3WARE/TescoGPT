"""Command-line entry points for reproducible project workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

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

    parser.error(f"Unknown command: {args.command}")
