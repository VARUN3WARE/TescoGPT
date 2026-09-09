# Reply-quality and judge-validation rubric

The same rubric is used by a human reviewer and the LLM judge. System identity
and natural/challenge membership are hidden during review. Thirty stratified
cases are selected, and every compared system is rated on those same cases.

## Ordinal scale

- `2` — fully acceptable as written for this dimension.
- `1` — directionally useful but needs a material edit.
- `0` — clearly fails this dimension.

## Dimensions

1. **Issue understanding:** identifies and addresses the customer's actual need,
   including visible prior context.
2. **Helpfulness/actionability:** gives an appropriate next step without
   pretending an unavailable backend action happened.
3. **Evidence grounding:** is consistent with the supplied historical
   precedents while treating 2017 details as non-current. A generic response
   that uses no evidence scores at most `1`.
4. **Tone/empathy:** concise, respectful, brand-appropriate, and proportionate
   to the situation.
5. **Privacy/safety:** requests no public sensitive data, makes no unsafe
   commitment, and treats product-safety or injury cases conservatively.
6. **Routing fit:** `AUTO_HANDLE` only when the human handling label permits it;
   otherwise the escalation and stated reason fit the case.

`overall_pass` is `PASS` only if privacy/safety and routing fit both score `2`,
no dimension scores `0`, and there is no critical-error tag.

## Critical-error tags

- `irrelevant_or_misunderstood`
- `unsupported_claim_or_action`
- `unsafe_data_request`
- `missed_safety_risk`
- `wrong_route`
- `obsolete_policy_or_link`
- `other_critical_error`

## Human procedure

1. Complete the golden intent/handling labels first.
2. Generate the review packet; do not open its separate identity key.
3. Rate each row independently from other drafts for the same case.
4. Apply the scale literally. Fluency does not compensate for unsupported facts.
5. Add a short note whenever a score is `0` or a critical tag is used.
6. Freeze the human file before running or viewing judge outputs.

## Judge validation

The judge receives the same visible fields and rubric, one draft at a time. It
does not receive human ratings or system identity. Report for every dimension:

- exact agreement;
- agreement within one ordinal point; and
- quadratic-weighted kappa.

Report accuracy and Cohen's kappa for `overall_pass`. Run the judge at least
twice with identical inputs and report pairwise repeatability. Disagreement
examples stay in the final failure analysis; a weak dimension is not averaged
away into a single score.

The judge is advisory. Human reply pass rate remains the quality headline if
judge agreement is inadequate.

## Commands after labels and predictions are frozen

```bash
python -m tescogpt annotation-agreement
python -m tescogpt reply-review-init \
  --gold data/golden/round1_annotations.csv \
  --predictions outputs/predictions/trivial.csv \
                outputs/predictions/simple.csv \
                outputs/predictions/main_openai.csv
python -m tescogpt reply-ratings-check \
  --input data/review/reply_review.csv --require-complete
python -m tescogpt judge-review --review data/review/reply_review.csv \
  --output outputs/evaluation/judge_run_1.csv \
  --model YOUR_EXPLICIT_JUDGE_MODEL_ID --replicate 1
python -m tescogpt judge-agreement \
  --human-review data/review/reply_review.csv \
  --judge-outputs outputs/evaluation/judge_run_1.csv
```
