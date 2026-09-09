# Outcome-aware retrieval

## Why plain similarity is insufficient

The Twitter corpus records what Tesco said, not whether the advice was correct
or the case was resolved. A nearest-neighbor system can therefore retrieve a
fluent reply that led to another complaint, only moved the customer to a private
channel, or asserted a backend action unavailable to this project.

TescoGPT keeps the source case visible but separates lexical similarity from
weak outcome evidence. The distinction is central to the project: historical
behavior is a precedent, not automatically a successful resolution.

## Method

1. BM25 retrieves the top 50 training-period customer messages.
2. Each candidate receives one non-ground-truth outcome tier:
   `positive_followup_proxy`, `unresolved_followup_proxy`,
   `conflicting_followup_proxy`, `private_handoff_without_outcome`,
   `unclassified_followup`, or `no_customer_followup`.
3. Static checks flag private-channel requests, personal-data requests,
   unsupported backend actions, financial promises, links, literal contact
   information, and replies longer than 280 characters.
4. A frozen reranker combines normalized lexical score (weight 0.82), a small
   outcome adjustment (+0.18 to -0.25), and a capped safety penalty (up to 0.20).
5. The top three precedents retain the customer message, Tesco reply, subsequent
   customer follow-up, component scores, tier, and flags for audit.

The weights are engineering priors frozen before golden labels are read. They
are not fitted on the final evaluation set, and the score is not a probability.

## Leakage controls

- Only `split == train` rows may enter the index; an unpartitioned corpus fails.
- A target case and every case in its conversation are excluded.
- Test-period historical replies and customer follow-ups are never input to
  retrieval or drafting.
- The 2017 response itself is evidence text, never a current-policy source.

## Current artifact sanity check

The committed artifact contains three precedents for each of 200 frozen cases.
Of the 200 rank-one precedents, 123 have a positive-follow-up phrase proxy and 41
still trigger at least one static reply warning. That second number is useful:
outcome reranking improves evidence selection but does not make copied text safe
to send. Whether outcome reranking improves human relevance remains pending and
will be reported before retrieval is called successful.

## Blinded pooled relevance review

The frozen review samples 25 golden-set queries in the same natural/challenge
proportion (19/6), pools the union of each system's top three, and randomizes
candidate order within each query. This produces 117 judgments. The review CSV
contains query and candidate text only; system identity, ranks, sample slice,
scores, and outcome tier remain in a separate identity key until ratings freeze.

For each query-candidate pair, assign one grade:

- `2` — same underlying issue and a compatible support action or resolution;
- `1` — related issue, but the action is missing, incomplete, or materially
  different;
- `0` — irrelevant, contradictory, or likely to mislead drafting.

Judge usefulness as historical evidence, not fluency or whether old Tesco text
can be copied verbatim today. Fill `reviewer_id` on every row and use
`relevance_reason` for borderline grades. Do not open the identity key until the
review file has been completed and frozen.

The evaluator reports macro-averaged pooled Precision@3 (grades 1 or 2 count as
relevant), pooled nDCG@3 (graded relevance), and per-query win/tie/loss for
outcome reranking versus BM25. The ideal DCG is calculated over the judged union.
These are deliberately named **pooled** metrics: because unreturned corpus items
are not judged, corpus-wide Recall@3 is not estimable. Twenty-five queries also
produce wide uncertainty, so the result is a targeted reranker diagnostic, not
a general retrieval-quality claim.

## Reproduce

```bash
python -m tescogpt retrieve \
  --input data/golden/golden_candidates.csv \
  --corpus data/processed/tesco_cases.csv \
  --output outputs/retrieval/outcome_aware.csv \
  --top-k 3 --candidate-pool 50

python -m tescogpt retrieval-review-check \
  --input data/review/retrieval_relevance.csv

# Run these only after the human review is complete.
python -m tescogpt retrieval-review-freeze \
  --review data/review/retrieval_relevance.csv \
  --identity-key data/review/retrieval_relevance_key.csv \
  --manifest data/review/retrieval_relevance.manifest.json

python -m tescogpt retrieval-evaluate \
  --review data/review/retrieval_relevance.csv \
  --identity-key data/review/retrieval_relevance_key.csv \
  --manifest data/review/retrieval_relevance.manifest.json \
  --output outputs/evaluation/retrieval_metrics.json --top-k 3
```
