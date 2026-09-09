# Baseline definitions

Both baselines implement the same validated output contract that the final agent
must use: one intent, confidence, public draft, handling decision, reason code,
safety flags, and zero or more evidence case IDs and quotes. They are frozen
before the main system is evaluated.

## Trivial constant baseline

- always predicts `other_or_unclear` with confidence zero;
- always returns the same generic acknowledgement;
- always escalates with `OUT_OF_SCOPE_OR_UNCLEAR`; and
- retrieves no historical evidence.

This establishes zero automation coverage and exposes why routing accuracy alone
is a poor headline: always escalating cannot create an unsafe automatic send,
but it provides none of the efficiency the system is meant to deliver.

## Simple rules plus BM25 baseline

- applies ordered keyword expressions that follow the annotation guide's intent
  tie-break order;
- uses explicit safety, order, money, live-information, and distress rules for
  handling;
- retrieves one lexically similar historical customer case with BM25; and
- copies that case's historical Tesco reply as its draft.

The retrieval corpus is restricted to the chronological training split. The
runner refuses an unpartitioned corpus, excludes the target case and its complete
conversation, and records hashes for inputs, corpus, and predictions.

This is intentionally a realistic but weak comparator. Lexical similarity does
not prove that two cases need the same action, and copying a historical reply can
inherit obsolete policy, irrelevant questions, private-channel requests, or a
tone mismatch. Those failures are measured rather than quietly repaired in the
baseline.

## Reproduction

After preparing cases and freezing candidates as described in the README:

```bash
python -m tescogpt predict --system trivial \
  --input data/golden/golden_candidates.csv \
  --output outputs/predictions/trivial.csv

python -m tescogpt predict --system simple \
  --input data/golden/golden_candidates.csv \
  --corpus data/processed/tesco_cases.csv \
  --output outputs/predictions/simple.csv
```

The committed CSVs are model outputs, not evaluation results. Metrics will only
be computed after the human labels are complete and frozen.
