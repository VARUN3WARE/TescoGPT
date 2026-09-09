# TescoGPT — evaluation-first support automation

**Report status:** experiment design and pre-label diagnostics are frozen. Human
gold labels, API-backed main predictions, reply ratings, and judge agreement are
pending. This document is intentionally not yet submission-ready.

## 1. Problem framing

TescoGPT handles public Tesco support messages from the 2017 Customer Support on
Twitter dataset. It returns one operational intent, a public reply draft,
`AUTO_HANDLE` or `ESCALATE`, a reason, and the historical precedents used.

The deployment question is not “How often is the model correct?” It is: **How
much traffic can be handled automatically while keeping harmful automatic sends
within a measured limit?** An always-escalate system is safe in a narrow sense
but creates no operational value. A fluent reply that requests personal data,
copies an obsolete policy, or invents a refund is unacceptable even when its
intent label is correct.

“Good” therefore means useful macro-F1 across the ten Tesco-specific intents,
relevant historical evidence, a helpful public draft, and useful automation
coverage reported together with unsafe-auto risk and uncertainty. The natural
traffic slice estimates performance on the sampled period; the challenge slice
only stress-tests difficult behavior.

This project does not connect to Tesco accounts, orders, payments, inventory, or
refund systems. It cannot complete those actions and must not imply that it did.
It also does not treat a 2017 reply as current Tesco policy or attempt to build a
general chatbot.

## 2. Data and evaluation set

The pipeline reconstructs complete reply trees rather than treating tweets as
independent rows. The full scan found 38,573 Tesco-authored messages, 16,722
Tesco-containing conversations, and 73,159 messages in those conversations.
About 54% of Tesco messages contain numbered multi-part fragments, so naïve
single-row pairing would truncate many historical responses. All figures and
source/output hashes are generated in the [data audit](docs/TESCO_DATA_AUDIT.md).

Complete conversations are split chronologically 70/15/15. The frozen test
sample contains one evaluated case from each of 200 conversations: 150 random
natural cases and 50 separately reported challenge cases. Challenge selection
targets safety, injury/allergy, money, accounts/orders, repeated failure,
distress, missing media/context, short messages, and punctuation-heavy cases.

The [codebook](docs/ANNOTATION_GUIDE.md) defines ten intents, tie-break rules,
handling decisions, twelve reason codes, and risk tags. One person labels
all 200 cases. A different person independently labels a stratified 60-case
overlap. Neither sees later Tesco responses, slice membership, challenge flags,
model output, or the other person's labels. The repository will report raw
agreement and Cohen's kappa before any adjudication.

## 3. System

Two baselines and the final system share one validated output schema.

- **Constant baseline:** always predicts `other_or_unclear`, returns one generic
  acknowledgement, and escalates everything.
- **Simple baseline:** ordered keyword intent rules, explicit routing rules, and
  the closest training-period Tesco reply from BM25.
- **TescoGPT:** BM25 retrieves 50 candidates, an outcome/safety reranker selects
  three precedents, a structured drafter writes a response, and a deterministic
  policy gate makes the final automation decision.

Historical outcomes are weak evidence. A later “thanks” is a positive phrase
proxy, not proof of resolution; no follow-up is not success. The reranker keeps
these categories explicit, penalizes unresolved follow-ups and unsafe historical
reply patterns, and preserves all evidence for inspection.

The API drafter uses an explicit model ID, strict JSON-schema output,
`store=False`, sanitized text, and a request-hash cache. It can propose an intent
and draft but cannot authorize automatic sending. Deterministic rules force
review for safety/injury, backend lookup, money, personal data, unavailable
media, current facts, repeated failure/distress, and discretionary judgment. A
draft that requests data, claims an unavailable action, promises money, embeds a
link/contact detail, or exceeds 280 characters is discarded and replaced with a
safe escalation draft.

## 4. Evaluation protocol

Intent evaluation reports accuracy, macro-F1 over observed intents and the full
taxonomy, every per-intent error count, and a 2,000-resample interval. Routing
reports coverage, unsafe automatic handling, needless escalation, and a
one-sided 95% Wilson upper bound on unsafe-auto risk. The entire score-ranked
risk–coverage curve is retained; selecting a test-set point after seeing that
curve is prohibited.

Reply quality uses a six-dimension, system-blinded human rubric: issue
understanding, helpfulness/actionability, evidence grounding, tone/empathy,
privacy/safety, and routing fit. The same rubric is applied by an LLM judge.
Agreement is reported per dimension using exact agreement, within-one agreement,
and quadratic-weighted kappa; overall pass uses accuracy and Cohen's kappa.
Repeated judge runs expose instability. If agreement is weak, the human pass
rate remains the headline.

## 5. Results

Human-dependent results are not available yet. The final table must compare both
baselines and the API-backed TescoGPT system on natural and challenge slices.
The offline reproduction command will generate it from committed per-example
files; numbers will not be typed into this report manually.

One pre-label regression check is available:

| System | Static warnings | Automatic cases | Warned automatic cases |
|---|---:|---:|---:|
| Constant / always escalate | 0/200 | 0 | 0 |
| Keyword + copied BM25 reply | 116/200 | 2 | 1 |
| Guarded offline template | 0/200 | 9 | 0 |

The copied replies trigger 98 private-channel, 78 personal-data, 53 excessive-
length, and one unsupported-financial-promise warning; rows can trigger multiple
warnings. These counts validate deterministic checks, not overall reply quality.

## 6. Preliminary failure analysis

These are traceable failure candidates identified before reading gold labels.
Final top-five frequencies and examples will be selected from frozen human
errors rather than from these hypotheses alone.

1. **Polite words can dominate intent rules.** In `tesco-2855864`, a question
   about salmon weight and a vertebra is predicted as feedback because it ends
   with “Thanks.” The safety rule still escalates it, but the intent is likely
   wrong. Hypothesis: a structured LLM classifier will handle compositional
   meaning better; the safety gate must remain independent.
2. **Missing media makes lexical retrieval meaningless.** `tesco-2950175` is
   only `@Tesco <media_or_link>`, yet BM25 retrieves and copies an unrelated
   promotion explanation. Hypothesis: reject evidence below a query-information
   threshold rather than always returning `k` items.
3. **Historical style imports privacy risk.** For `tesco-1726133`, the copied
   precedent asks for an order number, full name, and address and discusses a
   refund for a different problem. Hypothesis: retrieval evidence may guide a
   human handoff, but raw historical text must never be the sendable draft.
4. **A positive phrase is not a resolution.** The precedent `tesco-2495531`
   receives “Thanks” after Tesco says it does not know the answer and will ask a
   support team. It is tagged by the positive-follow-up proxy even though no
   resolution is visible. Hypothesis: manually audit outcome tiers and report
   proxy precision before calling the reranker outcome-aware.
5. **Conservative rules can waste safe automation.** `tesco-2899807` thanks
   Tesco for cat treats and includes a photo; the guarded agent escalates solely
   because media is present. A generic acknowledgement may be safe without
   interpreting the image. Hypothesis: distinguish “media is present” from
   “the requested answer depends on media,” then validate the distinction on
   human handling labels.

## 7. What is misleading about my headline number?

No single number establishes trust. Macro-F1 can hide failure on rare safety
intents. Routing accuracy rewards always escalating. Coverage without the number
and severity of unsafe sends rewards recklessness. Zero observed unsafe sends
does not imply zero risk; with a small auto-handled denominator, the confidence
bound can remain large. The 50-case challenge slice deliberately changes
prevalence and cannot estimate production traffic. The natural slice represents
one late-2017 Twitter window, not current Tesco customers or policies.

The current “0 static warnings” for the guarded template is especially easy to
misread. Regexes detect a narrow list of visible hazards; a vague, irrelevant,
unempathetic, or unsupported reply can pass every check. Likewise, 61.5% of
rank-one reranked precedents have a positive phrase in a follow-up, but “thanks”
often signals politeness rather than resolution. Final claims must place human
reply pass rate, judge–human agreement, coverage, unsafe-auto counts, confidence
bounds, and slice composition beside the headline.

## 8. With one more week

I would add a genuinely labelled development set for threshold calibration,
manually verify a sample of retrieval outcomes, test recent/current Tesco policy
sources with timestamped provenance, expand adversarial and multilingual cases,
measure calibration drift, and run a prospective shadow-mode trial with support
agents. Backend integrations would use scoped tools and confirmation receipts so
the system could distinguish drafting advice from actually completing an action.
