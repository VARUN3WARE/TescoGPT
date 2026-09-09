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
to send. Human relevance labels and Recall@k/nDCG remain pending and will be
reported before retrieval is called successful.

## Reproduce

```bash
python -m tescogpt retrieve \
  --input data/golden/golden_candidates.csv \
  --corpus data/processed/tesco_cases.csv \
  --output outputs/retrieval/outcome_aware.csv \
  --top-k 3 --candidate-pool 50
```
