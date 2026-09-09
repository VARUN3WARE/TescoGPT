"""Command-line entry points for reproducible project workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from tescogpt.data.audit import audit_conversations
from tescogpt.data.threads import extract_brand_conversations
from tescogpt.evaluation.labels import initialize_annotation_rounds, validate_annotations
from tescogpt.evaluation.sampling import sample_golden_candidates
from tescogpt.prediction import run_predictions
from tescogpt.retrieval.outcome import write_retrieval_artifact


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

    sample = subparsers.add_parser(
        "sample",
        help="Build cases and create a deterministic blind annotation sheet.",
    )
    sample.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/tesco_messages.csv"),
        help="Prepared long-form conversation CSV.",
    )
    sample.add_argument(
        "--cases-output",
        type=Path,
        default=Path("data/processed/tesco_cases.csv"),
        help="Derived case table; contains hidden historical outcomes.",
    )
    sample.add_argument(
        "--candidates-output",
        type=Path,
        default=Path("data/golden/golden_candidates.csv"),
        help="Blind CSV to be labelled by a human.",
    )
    sample.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("data/golden/golden_candidates.manifest.json"),
    )
    sample.add_argument(
        "--codebook",
        type=Path,
        default=Path("docs/ANNOTATION_GUIDE.md"),
    )
    sample.add_argument("--natural-count", type=int, default=150)
    sample.add_argument("--challenge-count", type=int, default=50)
    sample.add_argument("--seed", type=int, default=20260909)

    labels_check = subparsers.add_parser(
        "labels-check",
        help="Validate labels and show annotation completion progress.",
    )
    labels_check.add_argument("--input", type=Path, required=True)
    labels_check.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any row is not fully labelled.",
    )

    labels_init = subparsers.add_parser(
        "labels-init",
        help="Create blank independent annotation rounds from frozen candidates.",
    )
    labels_init.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/golden/golden_candidates.csv"),
    )
    labels_init.add_argument(
        "--round-one",
        type=Path,
        default=Path("data/golden/round1_annotations.csv"),
    )
    labels_init.add_argument(
        "--round-two",
        type=Path,
        default=Path("data/golden/round2_annotations.csv"),
    )
    labels_init.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/golden/annotation_rounds.manifest.json"),
    )
    labels_init.add_argument("--second-annotation-count", type=int, default=60)
    labels_init.add_argument("--seed", type=int, default=20260910)

    predict = subparsers.add_parser(
        "predict",
        help="Run a baseline over a case sheet using the shared output contract.",
    )
    predict.add_argument("--system", choices=("trivial", "simple"), required=True)
    predict.add_argument(
        "--input",
        type=Path,
        default=Path("data/golden/golden_candidates.csv"),
    )
    predict.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/processed/tesco_cases.csv"),
        help="Historical case corpus; required by the simple baseline.",
    )
    predict.add_argument("--output", type=Path, required=True)

    retrieve = subparsers.add_parser(
        "retrieve",
        help="Write auditable outcome-aware precedents for each input case.",
    )
    retrieve.add_argument(
        "--input",
        type=Path,
        default=Path("data/golden/golden_candidates.csv"),
    )
    retrieve.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/processed/tesco_cases.csv"),
    )
    retrieve.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/retrieval/outcome_aware.csv"),
    )
    retrieve.add_argument("--top-k", type=int, default=3)
    retrieve.add_argument("--candidate-pool", type=int, default=50)

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

    if args.command == "sample":
        manifest = sample_golden_candidates(
            prepared_messages_path=args.input,
            cases_output_path=args.cases_output,
            candidates_output_path=args.candidates_output,
            manifest_output_path=args.manifest_output,
            codebook_path=args.codebook,
            natural_count=args.natural_count,
            challenge_count=args.challenge_count,
            seed=args.seed,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.command == "labels-check":
        progress = validate_annotations(args.input, require_complete=args.require_complete)
        print(json.dumps(progress, indent=2, sort_keys=True))
        return

    if args.command == "labels-init":
        manifest = initialize_annotation_rounds(
            candidates_path=args.candidates,
            round_one_path=args.round_one,
            round_two_path=args.round_two,
            manifest_path=args.manifest,
            second_annotation_count=args.second_annotation_count,
            seed=args.seed,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.command == "predict":
        manifest = run_predictions(
            system=args.system,
            input_path=args.input,
            output_path=args.output,
            corpus_path=args.corpus if args.system == "simple" else None,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.command == "retrieve":
        manifest = write_retrieval_artifact(
            input_path=args.input,
            corpus_path=args.corpus,
            output_path=args.output,
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    parser.error(f"Unknown command: {args.command}")
