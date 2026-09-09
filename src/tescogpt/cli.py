"""Command-line entry points for reproducible project workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from tescogpt.data.audit import audit_conversations
from tescogpt.data.taxonomy_audit import audit_taxonomy_themes
from tescogpt.data.threads import extract_brand_conversations
from tescogpt.evaluation.adjudication import (
    finalize_adjudication,
    initialize_adjudication,
    validate_adjudication,
)
from tescogpt.evaluation.agreement import annotation_agreement
from tescogpt.evaluation.failure_analysis import build_failure_evidence
from tescogpt.evaluation.judge import judge_human_agreement, judge_review_sheet
from tescogpt.evaluation.labels import (
    freeze_human_annotations,
    initialize_annotation_rounds,
    validate_annotations,
)
from tescogpt.evaluation.metrics import evaluate_predictions
from tescogpt.evaluation.reply_review import (
    evaluate_reply_quality,
    freeze_reply_review,
    initialize_reply_review,
    validate_reply_ratings,
)
from tescogpt.evaluation.reproduce import reproduce_project
from tescogpt.evaluation.retrieval_review import (
    evaluate_retrieval_review,
    freeze_retrieval_review,
    initialize_retrieval_review,
    validate_retrieval_review,
)
from tescogpt.evaluation.safety import audit_prediction_safety
from tescogpt.evaluation.sampling import sample_golden_candidates
from tescogpt.evaluation.status import project_status
from tescogpt.prediction import run_predictions, smoke_test_openai
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

    taxonomy_audit = subparsers.add_parser(
        "taxonomy-audit",
        help="Audit overlapping intent themes on training cases only.",
    )
    taxonomy_audit.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/tesco_cases.csv"),
    )
    taxonomy_audit.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/data/taxonomy_audit.json"),
    )
    taxonomy_audit.add_argument(
        "--markdown",
        type=Path,
        default=Path("docs/TAXONOMY_DERIVATION.md"),
    )
    taxonomy_audit.add_argument("--seed", type=int, default=20260913)
    taxonomy_audit.add_argument("--examples-per-theme", type=int, default=3)

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

    labels_freeze = subparsers.add_parser(
        "labels-freeze",
        help="Verify completed human rounds and freeze their hashes.",
    )
    labels_freeze.add_argument(
        "--candidates", type=Path, default=Path("data/golden/golden_candidates.csv")
    )
    labels_freeze.add_argument(
        "--round-one", type=Path, default=Path("data/golden/round1_annotations.csv")
    )
    labels_freeze.add_argument(
        "--round-two", type=Path, default=Path("data/golden/round2_annotations.csv")
    )
    labels_freeze.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/golden/annotation_rounds.manifest.json"),
    )
    labels_freeze.add_argument(
        "--codebook", type=Path, default=Path("docs/ANNOTATION_GUIDE.md")
    )

    adjudication_init = subparsers.add_parser(
        "adjudication-init",
        help="Create a blind adjudication sheet for independent label disagreements.",
    )
    adjudication_init.add_argument(
        "--round-one", type=Path, default=Path("data/golden/round1_annotations.csv")
    )
    adjudication_init.add_argument(
        "--round-two", type=Path, default=Path("data/golden/round2_annotations.csv")
    )
    adjudication_init.add_argument(
        "--round-manifest",
        type=Path,
        default=Path("data/golden/annotation_rounds.manifest.json"),
    )
    adjudication_init.add_argument(
        "--output", type=Path, default=Path("data/golden/adjudication.csv")
    )
    adjudication_init.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/golden/adjudication.manifest.json"),
    )

    adjudication_check = subparsers.add_parser(
        "adjudication-check",
        help="Validate adjudicated overlap labels and show resolution progress.",
    )
    adjudication_check.add_argument(
        "--input", type=Path, default=Path("data/golden/adjudication.csv")
    )
    adjudication_check.add_argument("--require-complete", action="store_true")

    adjudication_finalize = subparsers.add_parser(
        "adjudication-finalize",
        help="Freeze adjudication and create the distinct final 200-case gold file.",
    )
    adjudication_finalize.add_argument(
        "--round-one", type=Path, default=Path("data/golden/round1_annotations.csv")
    )
    adjudication_finalize.add_argument(
        "--round-two", type=Path, default=Path("data/golden/round2_annotations.csv")
    )
    adjudication_finalize.add_argument(
        "--round-manifest",
        type=Path,
        default=Path("data/golden/annotation_rounds.manifest.json"),
    )
    adjudication_finalize.add_argument(
        "--adjudication", type=Path, default=Path("data/golden/adjudication.csv")
    )
    adjudication_finalize.add_argument(
        "--adjudication-manifest",
        type=Path,
        default=Path("data/golden/adjudication.manifest.json"),
    )
    adjudication_finalize.add_argument(
        "--output", type=Path, default=Path("data/golden/final_annotations.csv")
    )

    predict = subparsers.add_parser(
        "predict",
        help="Run a baseline over a case sheet using the shared output contract.",
    )
    predict.add_argument(
        "--system",
        choices=("trivial", "simple", "main-template", "main-openai"),
        required=True,
    )
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
    predict.add_argument(
        "--model",
        default=None,
        help="Explicit Responses API model ID; required only for main-openai.",
    )
    predict.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/cache/openai_drafts"),
    )

    api_smoke = subparsers.add_parser(
        "api-smoke",
        help="Run one frozen case through the OpenAI path before a paid batch.",
    )
    api_smoke.add_argument(
        "--input",
        type=Path,
        default=Path("data/golden/golden_candidates.csv"),
    )
    api_smoke.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/processed/tesco_cases.csv"),
    )
    api_smoke.add_argument("--model", required=True, help="Explicit Responses API model ID.")
    api_smoke.add_argument(
        "--case-id",
        default=None,
        help="Frozen case ID to test; defaults to the first input row.",
    )
    api_smoke.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/cache/openai_drafts"),
    )

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

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate prediction files against a complete human-labelled set.",
    )
    evaluate.add_argument("--gold", type=Path, required=True)
    evaluate.add_argument(
        "--registry",
        type=Path,
        default=Path("data/golden/golden_candidates.csv"),
    )
    evaluate.add_argument("--predictions", type=Path, nargs="+", required=True)
    evaluate.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/metrics.json"),
    )
    evaluate.add_argument("--bootstrap-iterations", type=int, default=2000)
    evaluate.add_argument("--seed", type=int, default=20260911)

    safety_audit = subparsers.add_parser(
        "safety-audit",
        help="Run label-free static checks over public reply drafts.",
    )
    safety_audit.add_argument("--predictions", type=Path, nargs="+", required=True)
    safety_audit.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/static_safety.json"),
    )
    safety_audit.add_argument(
        "--details-directory",
        type=Path,
        default=Path("outputs/evaluation/static_safety"),
    )

    annotation_agree = subparsers.add_parser(
        "annotation-agreement",
        help="Measure independent human agreement before adjudication.",
    )
    annotation_agree.add_argument(
        "--round-one", type=Path, default=Path("data/golden/round1_annotations.csv")
    )
    annotation_agree.add_argument(
        "--round-two", type=Path, default=Path("data/golden/round2_annotations.csv")
    )
    annotation_agree.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/annotation_agreement.json"),
    )

    review_init = subparsers.add_parser(
        "reply-review-init",
        help="Create a system-blinded human reply-quality review packet.",
    )
    review_init.add_argument("--gold", type=Path, required=True)
    review_init.add_argument(
        "--registry", type=Path, default=Path("data/golden/golden_candidates.csv")
    )
    review_init.add_argument("--predictions", type=Path, nargs="+", required=True)
    review_init.add_argument(
        "--review", type=Path, default=Path("data/review/reply_review.csv")
    )
    review_init.add_argument(
        "--identity-key", type=Path, default=Path("data/review/reply_review_key.csv")
    )
    review_init.add_argument(
        "--manifest", type=Path, default=Path("data/review/reply_review.manifest.json")
    )
    review_init.add_argument("--case-count", type=int, default=30)
    review_init.add_argument("--decoy-count", type=int, default=10)
    review_init.add_argument("--seed", type=int, default=20260912)

    review_check = subparsers.add_parser(
        "reply-ratings-check",
        help="Validate a human reply-quality review sheet.",
    )
    review_check.add_argument("--input", type=Path, required=True)
    review_check.add_argument("--require-complete", action="store_true")

    review_freeze = subparsers.add_parser(
        "reply-review-freeze",
        help="Freeze complete human reply ratings and verify blinded content.",
    )
    review_freeze.add_argument("--review", type=Path, required=True)
    review_freeze.add_argument("--identity-key", type=Path, required=True)
    review_freeze.add_argument("--manifest", type=Path, required=True)

    review_evaluate = subparsers.add_parser(
        "reply-evaluate",
        help="Aggregate reply ratings and derive case-matched baseline comparisons.",
    )
    review_evaluate.add_argument("--review", type=Path, required=True)
    review_evaluate.add_argument("--identity-key", type=Path, required=True)
    review_evaluate.add_argument("--manifest", type=Path)
    review_evaluate.add_argument("--reference-system", required=True)
    review_evaluate.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/reply_quality.json"),
    )

    judge_review = subparsers.add_parser(
        "judge-review",
        help="Run a structured LLM judge over a blinded review packet.",
    )
    judge_review.add_argument("--review", type=Path, required=True)
    judge_review.add_argument("--review-manifest", type=Path)
    judge_review.add_argument("--output", type=Path, required=True)
    judge_review.add_argument("--model", required=True)
    judge_review.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/cache/judge"),
    )
    judge_review.add_argument("--replicate", type=int, default=1)

    judge_agree = subparsers.add_parser(
        "judge-agreement",
        help="Compare one or more judge runs with frozen human reply ratings.",
    )
    judge_agree.add_argument("--human-review", type=Path, required=True)
    judge_agree.add_argument("--identity-key", type=Path)
    judge_agree.add_argument("--judge-outputs", type=Path, nargs="+", required=True)
    judge_agree.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/judge_human_agreement.json"),
    )

    reproduce = subparsers.add_parser(
        "reproduce",
        help="Verify committed artifacts and reproduce offline evaluation outputs.",
    )
    reproduce.add_argument(
        "--config", type=Path, default=Path("config/experiment.json")
    )
    reproduce.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Run integrity/readiness checks without pretending labels are complete.",
    )

    status = subparsers.add_parser(
        "status",
        help="Show every human/API checkpoint and the next action for the final run.",
    )
    status.add_argument(
        "--config", type=Path, default=Path("config/experiment.json")
    )
    status.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override status_output from the experiment config.",
    )
    status.add_argument("--require-final-ready", action="store_true")

    retrieval_review_init = subparsers.add_parser(
        "retrieval-review-init",
        help="Create a pooled, system-blinded retrieval relevance review.",
    )
    retrieval_review_init.add_argument(
        "--registry", type=Path, default=Path("data/golden/golden_candidates.csv")
    )
    retrieval_review_init.add_argument(
        "--corpus", type=Path, default=Path("data/processed/tesco_cases.csv")
    )
    retrieval_review_init.add_argument(
        "--review", type=Path, default=Path("data/review/retrieval_relevance.csv")
    )
    retrieval_review_init.add_argument(
        "--identity-key",
        type=Path,
        default=Path("data/review/retrieval_relevance_key.csv"),
    )
    retrieval_review_init.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/review/retrieval_relevance.manifest.json"),
    )
    retrieval_review_init.add_argument("--query-count", type=int, default=25)
    retrieval_review_init.add_argument("--top-k", type=int, default=3)
    retrieval_review_init.add_argument("--candidate-pool", type=int, default=50)
    retrieval_review_init.add_argument("--seed", type=int, default=20260913)

    retrieval_review_check = subparsers.add_parser(
        "retrieval-review-check",
        help="Validate pooled retrieval relevance judgments.",
    )
    retrieval_review_check.add_argument("--input", type=Path, required=True)
    retrieval_review_check.add_argument("--require-complete", action="store_true")

    retrieval_review_freeze = subparsers.add_parser(
        "retrieval-review-freeze",
        help="Freeze completed human relevance judgments and verify blind content.",
    )
    retrieval_review_freeze.add_argument("--review", type=Path, required=True)
    retrieval_review_freeze.add_argument("--identity-key", type=Path, required=True)
    retrieval_review_freeze.add_argument("--manifest", type=Path, required=True)

    retrieval_evaluate = subparsers.add_parser(
        "retrieval-evaluate",
        help="Compute pooled Precision@k and nDCG@k from human relevance grades.",
    )
    retrieval_evaluate.add_argument("--review", type=Path, required=True)
    retrieval_evaluate.add_argument("--identity-key", type=Path, required=True)
    retrieval_evaluate.add_argument("--manifest", type=Path)
    retrieval_evaluate.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/retrieval_metrics.json"),
    )
    retrieval_evaluate.add_argument("--top-k", type=int, default=3)

    failure_analysis = subparsers.add_parser(
        "failure-analysis",
        help="Create a per-example failure ledger with explicit sample denominators.",
    )
    failure_analysis.add_argument("--gold", type=Path, required=True)
    failure_analysis.add_argument(
        "--registry", type=Path, default=Path("data/golden/golden_candidates.csv")
    )
    failure_analysis.add_argument("--predictions", type=Path, nargs="+", required=True)
    failure_analysis.add_argument(
        "--events",
        type=Path,
        default=Path("outputs/evaluation/failure_events.csv"),
    )
    failure_analysis.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/evaluation/failure_analysis.json"),
    )
    failure_analysis.add_argument("--reply-review", type=Path)
    failure_analysis.add_argument("--reply-key", type=Path)
    failure_analysis.add_argument("--retrieval-review", type=Path)
    failure_analysis.add_argument("--retrieval-key", type=Path)
    failure_analysis.add_argument("--retrieval-top-k", type=int, default=3)
    failure_analysis.add_argument("--judge-outputs", type=Path, nargs="*", default=[])

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

    if args.command == "taxonomy-audit":
        audit = audit_taxonomy_themes(
            input_path=args.input,
            output_path=args.output,
            markdown_path=args.markdown,
            seed=args.seed,
            examples_per_theme=args.examples_per_theme,
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

    if args.command == "labels-freeze":
        manifest = freeze_human_annotations(
            candidates_path=args.candidates,
            round_one_path=args.round_one,
            round_two_path=args.round_two,
            manifest_path=args.manifest,
            codebook_path=args.codebook,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.command == "adjudication-init":
        manifest = initialize_adjudication(
            round_one_path=args.round_one,
            round_two_path=args.round_two,
            round_manifest_path=args.round_manifest,
            output_path=args.output,
            manifest_path=args.manifest,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.command == "adjudication-check":
        progress = validate_adjudication(
            args.input,
            require_complete=args.require_complete,
        )
        print(json.dumps(progress, indent=2, sort_keys=True))
        return

    if args.command == "adjudication-finalize":
        manifest = finalize_adjudication(
            round_one_path=args.round_one,
            round_two_path=args.round_two,
            round_manifest_path=args.round_manifest,
            adjudication_path=args.adjudication,
            adjudication_manifest_path=args.adjudication_manifest,
            final_output_path=args.output,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.command == "predict":
        manifest = run_predictions(
            system=args.system,
            input_path=args.input,
            output_path=args.output,
            corpus_path=args.corpus if args.system != "trivial" else None,
            model=args.model,
            cache_dir=args.cache_dir,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.command == "api-smoke":
        result = smoke_test_openai(
            input_path=args.input,
            corpus_path=args.corpus,
            model=args.model,
            case_id=args.case_id,
            cache_dir=args.cache_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
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

    if args.command == "evaluate":
        report = evaluate_predictions(
            gold_path=args.gold,
            registry_path=args.registry,
            prediction_paths=args.predictions,
            output_path=args.output,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.command == "safety-audit":
        report = audit_prediction_safety(
            prediction_paths=args.predictions,
            output_path=args.output,
            details_directory=args.details_directory,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.command == "annotation-agreement":
        report = annotation_agreement(
            args.round_one,
            args.round_two,
            args.output,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.command == "reply-review-init":
        manifest = initialize_reply_review(
            gold_path=args.gold,
            registry_path=args.registry,
            prediction_paths=args.predictions,
            review_path=args.review,
            key_path=args.identity_key,
            manifest_path=args.manifest,
            case_count=args.case_count,
            decoy_count=args.decoy_count,
            seed=args.seed,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.command == "reply-ratings-check":
        progress = validate_reply_ratings(
            args.input,
            require_complete=args.require_complete,
        )
        print(json.dumps(progress, indent=2, sort_keys=True))
        return

    if args.command == "reply-review-freeze":
        manifest = freeze_reply_review(
            review_path=args.review,
            key_path=args.identity_key,
            manifest_path=args.manifest,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.command == "reply-evaluate":
        report = evaluate_reply_quality(
            review_path=args.review,
            key_path=args.identity_key,
            output_path=args.output,
            reference_system=args.reference_system,
            manifest_path=args.manifest,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.command == "judge-review":
        manifest = judge_review_sheet(
            review_path=args.review,
            output_path=args.output,
            model=args.model,
            cache_dir=args.cache_dir,
            replicate=args.replicate,
            review_manifest_path=args.review_manifest,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.command == "judge-agreement":
        report = judge_human_agreement(
            human_review_path=args.human_review,
            judge_paths=args.judge_outputs,
            output_path=args.output,
            identity_key_path=args.identity_key,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.command == "reproduce":
        report = reproduce_project(
            args.config,
            allow_incomplete=args.allow_incomplete,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.command == "status":
        report = project_status(
            args.config,
            args.output,
            require_final_ready=args.require_final_ready,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.command == "retrieval-review-init":
        manifest = initialize_retrieval_review(
            registry_path=args.registry,
            corpus_path=args.corpus,
            review_path=args.review,
            key_path=args.identity_key,
            manifest_path=args.manifest,
            query_count=args.query_count,
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
            seed=args.seed,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.command == "retrieval-review-check":
        progress = validate_retrieval_review(
            args.input,
            require_complete=args.require_complete,
        )
        print(json.dumps(progress, indent=2, sort_keys=True))
        return

    if args.command == "retrieval-review-freeze":
        manifest = freeze_retrieval_review(
            review_path=args.review,
            key_path=args.identity_key,
            manifest_path=args.manifest,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.command == "retrieval-evaluate":
        report = evaluate_retrieval_review(
            review_path=args.review,
            key_path=args.identity_key,
            output_path=args.output,
            top_k=args.top_k,
            manifest_path=args.manifest,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.command == "failure-analysis":
        report = build_failure_evidence(
            gold_path=args.gold,
            registry_path=args.registry,
            prediction_paths=args.predictions,
            events_path=args.events,
            summary_path=args.summary,
            reply_review_path=args.reply_review,
            reply_key_path=args.reply_key,
            retrieval_review_path=args.retrieval_review,
            retrieval_key_path=args.retrieval_key,
            retrieval_top_k=args.retrieval_top_k,
            judge_paths=args.judge_outputs,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    parser.error(f"Unknown command: {args.command}")
