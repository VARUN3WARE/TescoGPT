# Agent architecture and trust boundaries

```mermaid
flowchart LR
    A[Incoming message + prior context] --> B[Training-only BM25 retrieval]
    B --> C[Outcome and safety reranker]
    C --> D[Structured draft provider]
    D --> E[Deterministic policy gate]
    E --> F[Intent, draft, route, reason, evidence]
    E -->|unsafe draft| G[Safe fallback + forced escalation]
```

The design keeps three questions separate: what the customer needs, what
historical behavior is relevant, and whether a public reply is safe to send.
Every compared system ultimately emits the same validated schema.

## Draft providers

`guarded_template_v1` is the offline fallback. It uses the deterministic intent
heuristic, outcome-aware evidence pack, intent-specific acknowledgement, and
policy gate. It is deliberately conservative and is not presented as the final
LLM result.

`OpenAIDrafter` is the structured generation path. It:

- requires an explicit model ID so “latest” cannot silently change;
- sends only sanitized text and three training-period precedents;
- treats customer and retrieved text as untrusted data;
- uses the Responses API with a strict JSON schema and `store=False`;
- limits the public draft to 280 characters;
- verifies that every cited evidence ID was actually retrieved; and
- hashes model, instructions, schema, and input into a resumable local cache.

The integration follows the official
[Responses API reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create).
No API-backed artifact is committed yet because this environment has no API key.
That absence is recorded rather than replaced with invented model output.

## Non-bypassable policy

The model may suggest an intent and draft, but it cannot grant itself permission
to auto-handle. Deterministic rules force review for food safety/injury, account
or order lookup, money or commitments, exposed personal information, missing
media/context, live facts, repeated failure/distress, and cases requiring human
judgment.

The generated reply is independently scanned for personal-data requests,
private-channel requests, claimed backend actions, financial promises, links,
literal contact data, and excessive length. A flagged draft is discarded,
replaced with a safe acknowledgement, and escalated; the triggering flags remain
in the prediction artifact.

`automation_score` is a ranking score, not a calibrated probability. Its policy
thresholds are frozen engineering priors until development labels exist. They
must not be tuned on the final golden set.

## Current non-human sanity check

Across the frozen 200 cases, the copied-reply baseline triggers at least one
static warning on 116 drafts. The guarded template triggers none and auto-handles
9 cases. This does not establish reply correctness, appropriate empathy, or safe
automation; those claims require the pending human labels and reply ratings.

## Commands

Offline guarded agent:

```bash
python -m tescogpt predict --system main-template \
  --input data/golden/golden_candidates.csv \
  --corpus data/processed/tesco_cases.csv \
  --output outputs/predictions/main_template.csv
```

Optional API generation after installing `.[llm]` and setting
`OPENAI_API_KEY`:

```bash
python -m tescogpt predict --system main-openai \
  --model YOUR_EXPLICIT_MODEL_ID \
  --input data/golden/golden_candidates.csv \
  --corpus data/processed/tesco_cases.csv \
  --output outputs/predictions/main_openai.csv
```

The API run is resumable through `artifacts/cache/openai_drafts`; final
predictions and their manifest are the experiment artifact.
