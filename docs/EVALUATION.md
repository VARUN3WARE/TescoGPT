# Evaluation protocol

Metric code and definitions are frozen before the golden labels are available.
The evaluator refuses incomplete annotation files and requires prediction IDs to
match the 200 gold IDs exactly.

## Intent

- accuracy;
- macro-F1 over intents observed in each slice;
- macro-F1 over the complete ten-intent taxonomy;
- precision, recall, F1, support, false positives, and false negatives per
  intent; and
- a 2,000-resample percentile interval over examples.

The natural and challenge slices are always reported separately. The challenge
slice is a stress test and must not be described as traffic prevalence.

## Selective automation

- **Coverage:** fraction predicted `AUTO_HANDLE`.
- **Unsafe auto:** predicted `AUTO_HANDLE` when a human labelled `ESCALATE`.
- **Needless escalation:** predicted `ESCALATE` when a human labelled
  `AUTO_HANDLE`.
- **Unsafe-auto rate:** unsafe automatic cases divided by all automatically
  handled cases.
- **Uncertainty:** one-sided 95% Wilson upper bound on that rate.

The harness also reports a score-ranked risk–coverage curve at 5% increments.
It is diagnostic; choosing a test-set point after seeing the curve would be
test-set tuning. `automation_score` is a ranking score, not a calibrated
probability.

## Static reply checks

The label-free audit searches for a deliberately narrow list of observable
hazards: private-channel and personal-data requests, claimed backend actions,
financial promises, links, literal contact information, and replies over 280
characters.

| System | Flagged drafts | Flagged automatic drafts |
|---|---:|---:|
| Constant / always escalate | 0 / 200 | 0 |
| Keyword + copied BM25 reply | 116 / 200 | 1 |
| Guarded template | 0 / 200 | 0 |

These are deterministic warnings, not human quality judgments. In particular,
zero warnings can coexist with a vague, irrelevant, unhelpful, or tonally poor
reply. The result is a regression gate, not the headline.

## Reproduce current static checks

```bash
python -m tescogpt safety-audit \
  --predictions outputs/predictions/trivial.csv \
                outputs/predictions/simple.csv \
                outputs/predictions/main_template.csv
```

After human labels are complete:

```bash
python -m tescogpt evaluate \
  --gold data/golden/round1_annotations.csv \
  --registry data/golden/golden_candidates.csv \
  --predictions outputs/predictions/trivial.csv \
                outputs/predictions/simple.csv \
                outputs/predictions/main_template.csv
```

The final run will replace `main_template.csv` with the frozen API-backed main
system if that experiment is completed.
