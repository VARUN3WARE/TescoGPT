# Failure-analysis protocol

The final report must contain five failure modes with real examples and
hypotheses. The evidence ledger is generated before anyone chooses those five,
so selection cannot rely only on memorable anecdotes.

## Pre-registered candidate families

1. **Intent collision or composition.** A first-match rule or model focuses on a
   polite or transactional phrase and misses the operational issue. Evidence:
   gold/predicted intent mismatch, especially at high reported confidence.
2. **Unsafe automation.** The agent chooses `AUTO_HANDLE` where a human requires
   escalation. Hypothesis: confidence or lexical similarity overwhelms a safety,
   account, money, media, live-information, or judgment boundary.
3. **Needless escalation.** The agent escalates a safely answerable case.
   Hypothesis: conservative rules protect safety but sacrifice useful coverage,
   especially for acknowledgements and stable public guidance.
4. **Wrong operational reason.** The route is right but its stated reason is
   wrong or too generic. Hypothesis: binary routing hides weak handoff semantics.
5. **Irrelevant retrieval evidence.** A top-ranked precedent receives relevance
   grade `0`. Hypothesis: word overlap or a weak outcome proxy does not imply the
   same issue and compatible action.
6. **Unsafe, obsolete, or unhelpful draft.** Static checks warn or the blinded
   human reviewer fails a reply. Hypothesis: historical 2017 responses inherit
   private-data requests and stale policy, while safe templates can be vague.
7. **Judge blind spot.** Human and judge pass labels differ or an ordinal
   dimension differs by two points. Hypothesis: fluency or rubric interpretation
   causes the judge to miss grounding, privacy, or routing errors.

These are hypotheses, not findings. A family that produces no observed failures
will be reported as absent rather than illustrated with a cherry-picked near
miss.

## Evidence and selection rule

`failure-analysis` writes one row per observable event with the real case ID,
slice, message, system draft, severity, observed evidence, and source artifact.
It separately records these denominators:

- 200 gold cases per system for intent, route, reason, and static checks;
- 30 reviewed cases per system for human reply quality;
- 25 reviewed queries per retriever for retrieval relevance; and
- each genuine compared reply row for judge validity.

The final five are selected first by potential customer harm and trust impact,
then by frequency within the relevant denominator, and finally by whether the
failure reveals a distinct mechanism. Raw counts from different samples are
never ranked against each other. Each reported mode must include at least one
ledger row, an explicit denominator, a causal hypothesis, and a concrete next
test or mitigation.

## Command

```bash
python -m tescogpt failure-analysis \
  --gold data/golden/final_annotations.csv \
  --registry data/golden/golden_candidates.csv \
  --predictions outputs/predictions/trivial.csv \
                outputs/predictions/simple.csv \
                outputs/predictions/main_openai.csv \
  --reply-review data/review/reply_review.csv \
  --reply-key data/review/reply_review_key.csv \
  --retrieval-review data/review/retrieval_relevance.csv \
  --retrieval-key data/review/retrieval_relevance_key.csv \
  --judge-outputs outputs/evaluation/judge_run_1.csv \
                  outputs/evaluation/judge_run_2.csv
```
