# TescoGPT

An evaluation-first AI support agent for Tesco conversations from the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset.

> **Current status:** data audit and blind annotation workflow complete. The
> candidate set is frozen but explicitly `UNLABELED`; no headline result is
> claimed until genuine human labels and the complete evaluation are available.

## The question this project answers

**How much public support traffic can the agent handle automatically while
keeping unsafe automatic replies below an explicit, measured limit?**

For each incoming customer message, the completed system will return:

1. one operational intent from a Tesco-specific taxonomy;
2. a draft public reply grounded in historically similar Tesco conversations;
3. `auto_handle` or `escalate`, with a machine-readable reason; and
4. an evidence trail or a useful handoff packet for the human agent.

`auto_handle` means the public draft may be sent without pre-send human review.
It does **not** mean that an order, refund, safety complaint, or account issue was
resolved in a backend system that this project cannot access.

## Why Tesco

Tesco has enough conversations for modelling while offering a meaningful safety
boundary: routine stock and website questions coexist with food-safety,
allergen, injury, refund, delivery, and privacy-sensitive cases. That makes
escalation a real decision rather than a confidence threshold added for show.

An exploratory full-dataset scan found 38,573 Tesco-authored messages. Brand
selection will be justified by a reproducible audit in the repository rather
than by volume alone. See the generated [Tesco data audit](docs/TESCO_DATA_AUDIT.md)
for provenance, conversation structure, reply patterns, and outcome-proxy caveats.

## What “good” means

- **Useful intent routing:** report macro-F1 and per-intent errors, not accuracy
  alone.
- **Evidence-backed drafting:** distinguish a visible successful outcome from a
  generic reply or private-channel handoff.
- **Selective automation:** maximize coverage subject to a safety target and
  show the entire risk–coverage trade-off.
- **Safe public behavior:** never request sensitive information publicly or
  invent refunds, credits, policies, URLs, or completed actions.
- **Reproducible proof:** frozen IDs, cached predictions, raw evaluation outputs,
  uncertainty intervals, and one offline command for the headline results.

## What this project intentionally does not build

- live Tesco account, order, payment, refund, or inventory integrations;
- a claim that 2017 Twitter replies are current Tesco policy;
- an unconstrained general-purpose chatbot;
- a large multi-agent framework where deterministic code is sufficient; or
- a UI before the data, baselines, and evaluation are trustworthy.

## Evaluation design

The frozen evaluation candidate set contains 200 examples awaiting manual review:

- **150 natural-prevalence examples** from a held-out chronological window;
- **50 challenge examples** covering safety, ambiguity, multiple intents,
  missing context, image-dependent complaints, abuse, and non-English text.

The two slices will be reported separately. A stratified 60-example subset has
been frozen for a second independent human annotation. Both annotation sheets
hide later Tesco responses, slice membership, and challenge-selection flags.
The automated reply judge will be compared with humans per rubric dimension and
will not become the headline if agreement is weak.

The ten-intent taxonomy, routing criteria, tie-breaks, and safety reason codes
are defined in the [annotation guide](docs/ANNOTATION_GUIDE.md). The sampling
manifest records input hashes, timestamps, seeds, split sizes, and candidate
hashes in [data/golden](data/golden). `golden_candidates.csv` is a sampling
registry, while `round1_annotations.csv` and `round2_annotations.csv` are the
blinded files given to annotators.

Planned primary measures:

- intent macro-F1 with bootstrap confidence intervals;
- unsafe auto-handle rate and its confidence bound;
- automation coverage at a fixed safety target;
- risk–coverage curve and cost-sensitive routing utility;
- human all-critical-dimensions reply pass rate;
- pairwise reply win/tie/loss against baselines;
- retrieval Recall@k/nDCG on human relevance labels; and
- judge–human agreement, position consistency, and repeatability.

Metric definitions and the currently runnable label-free safety audit are in
[the evaluation protocol](docs/EVALUATION.md). Human-dependent metrics are coded
but intentionally fail while the annotation sheets are incomplete.

## Baselines

1. **Trivial:** constant `other_or_unclear` intent, generic acknowledgement,
   always escalate.
2. **Simple:** ordered keyword intent rules, BM25 nearest historical reply, and
   explicit safety/policy routing rules.
3. **Ablations:** the main system without retrieval and without the outcome
   evidence filter.

Their exact frozen behavior and limitations are documented in
[the baseline definitions](docs/BASELINES.md). Both emit the same validated
schema as the final agent.

## Agent architecture

The main path combines training-only, outcome-aware retrieval with a structured
drafter and a deterministic policy gate that can override unsafe generations.
An API-backed drafter is optional and requires an explicit model ID; the
committed guarded-template output keeps development and offline verification
runnable without a secret. See [the architecture and trust boundaries](docs/AGENT.md).

## Reproducibility contract

The final `README` command must reproduce committed headline tables in under 15
minutes without downloading the full dataset or making paid API calls. Rebuilding
the corpus or regenerating model responses will remain an optional, documented
workflow.

Until that workflow exists, setup commands will not be advertised as complete.
See [the project plan](docs/PROJECT_PLAN.md) for milestones and acceptance gates.

### Current development command

The first implemented slice reconstructs complete reply trees containing Tesco
messages. It is intentionally separate from the future headline-results command:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m tescogpt prepare --input path/to/twcs.csv
python -m tescogpt audit --manifest data/processed/tesco_messages.csv.manifest.json
python -m tescogpt sample
python -m tescogpt labels-init
python -m tescogpt labels-check --input data/golden/round1_annotations.csv
python -m tescogpt retrieve
```

Raw and full processed datasets are gitignored. The command writes a long-form
conversation CSV plus a manifest containing source/output hashes and structural
counts. `sample` creates a chronological conversation-level 70/15/15 split and
selects at most one evaluated case per conversation. `labels-init` is
deterministic and should only be rerun while the sheets are blank; once human
work begins, completed files must be preserved rather than overwritten.

To finish the human-label checkpoint, annotator one completes all 200 rows in
`round1_annotations.csv`, while a different person independently completes the
60 rows in `round2_annotations.csv`. Validate either file with
`labels-check --require-complete`. AI-generated labels do not satisfy this
project's golden-set requirement.

Historical replies are retrieved only from the training period and reranked by
weak follow-up evidence plus static safety penalties. The full method, its
unfitted scoring formula, and its limitations are in
[the retrieval note](docs/RETRIEVAL.md); raw top-three evidence is committed for
all 200 candidates.

## Repository map

```text
TescoGPT/
├── data/                 # raw data ignored; small samples and gold tracked
├── docs/                 # plan, data card, annotation guide, report
├── src/tescogpt/         # ingestion, models, policy, evaluation, CLI
├── tests/                # leakage, reconstruction, policy and metric tests
├── artifacts/            # large local models/indexes ignored
├── outputs/              # committed predictions, metrics and figures
├── DECISIONS.md          # non-obvious decisions and their rationale
└── README.md
```

## Research basis

- Dataset: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- Selective prediction: [The Art of Abstention](https://aclanthology.org/2021.acl-long.84/)
- Judge bias: [Judging the Judges](https://aclanthology.org/2025.ijcnlp-long.18/)
- Product context: [Hiver on the AI trust gap](https://hiverhq.com/blog/ai-trust-gap-in-support)

All borrowed code, models, prompts, and data will be attributed at the point of
use. Customer identifiers will remain anonymized, and no attempt will be made to
reidentify dataset participants.
