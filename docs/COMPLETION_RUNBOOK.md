# Completion runbook

`python -m tescogpt status` is the source of truth for progress. It writes
`outputs/evaluation/project_status.json`, lists every unmet final gate, and gives
the next concrete action. `--require-final-ready` makes the command usable as a
preflight gate.

Do not use an LLM to populate a human label or rating column. Do not inspect the
identity-key files while rating blinded reply or retrieval rows.

## 1. Complete the independent annotation rounds

Annotator one completes all 200 rows in
`outputs/annotation_workbook/round1_annotation.xlsx`. A different person
independently completes all 60 rows in `round2_annotation.xlsx`. Every row
needs the categorical labels, `must_include`, `must_avoid`, and the same stable
annotator ID for that round. Import each saved workbook with the commands below;
the importer verifies frozen IDs and context before atomically replacing its CSV.

```bash
python -m tescogpt labels-import-workbook \
  --workbook outputs/annotation_workbook/round1_annotation.xlsx \
  --output data/golden/round1_annotations.csv --require-complete
python -m tescogpt labels-import-workbook \
  --workbook outputs/annotation_workbook/round2_annotation.xlsx \
  --output data/golden/round2_annotations.csv --require-complete
python -m tescogpt labels-freeze
```

## 2. Measure agreement, then adjudicate

Agreement must be generated before any resolution. The independent source files
are never overwritten.

```bash
python -m tescogpt annotation-agreement
python -m tescogpt adjudication-init
```

One human fills every blank `final_*` value, one stable `adjudicator_id`, and a
short rationale for every disagreement in `data/golden/adjudication.csv`, then
runs:

```bash
python -m tescogpt adjudication-check --require-complete
python -m tescogpt adjudication-finalize
```

Set `gold_file` in `config/experiment.json` to
`data/golden/final_annotations.csv`. Leave `round_one_annotation_file` pointing
at round one so agreement remains independent.

## 3. Complete the blinded retrieval review

One reviewer fills `relevance_grade`, one stable `reviewer_id`, and a reason for
every borderline (`1`) judgment in the review workbook and does not inspect the
identity key. Importing checks that the blinded queries and candidates have not
changed before replacing the CSV.

```bash
python -m tescogpt retrieval-review-import-workbook --require-complete
python -m tescogpt retrieval-review-freeze \
  --review data/review/retrieval_relevance.csv \
  --identity-key data/review/retrieval_relevance_key.csv \
  --manifest data/review/retrieval_relevance.manifest.json
```

## 4. Freeze the API-backed main system

Install the optional dependency, set `OPENAI_API_KEY` outside the repository,
choose one explicit model ID, and smoke-test one frozen case before running all
200. The smoke-test response enters the same resumable cache as the batch, so a
successful call is reused rather than purchased twice.

```bash
python -m pip install -e ".[llm]"
python -m tescogpt api-smoke \
  --model YOUR_EXPLICIT_MODEL_ID \
  --input data/golden/golden_candidates.csv \
  --corpus data/processed/tesco_cases.csv
python -m tescogpt predict --system main-openai \
  --model YOUR_EXPLICIT_MODEL_ID \
  --input data/golden/golden_candidates.csv \
  --corpus data/processed/tesco_cases.csv \
  --output outputs/predictions/main_openai.csv
```

Replace `main_template.csv` in `prediction_files` with `main_openai.csv`, and set
`headline_system` to the exact system ID recorded in its adjacent manifest.

## 5. Rate replies before running the judge

Create the system-blinded packet from the final gold file and all three frozen
systems. One human rates every row with a stable reviewer ID using
`docs/JUDGE_RUBRIC.md` and adds a note for every zero or critical-error tag;
freeze those ratings before any judge call.

```bash
python -m tescogpt reply-review-init \
  --gold data/golden/final_annotations.csv \
  --predictions outputs/predictions/trivial.csv \
                outputs/predictions/simple.csv \
                outputs/predictions/main_openai.csv
python -m tescogpt reply-ratings-check \
  --input data/review/reply_review.csv --require-complete
python -m tescogpt reply-review-freeze \
  --review data/review/reply_review.csv \
  --identity-key data/review/reply_review_key.csv \
  --manifest data/review/reply_review.manifest.json
```

Add the three reply-review paths and the exact main system ID as
`reply_reference_system` in `config/experiment.json`.

## 6. Run two judge replicates

Use the same explicit judge model for both runs. Replicate is part of the cache
key, so the second run is an independent call rather than a reused first result.

```bash
python -m tescogpt judge-review \
  --review data/review/reply_review.csv \
  --review-manifest data/review/reply_review.manifest.json \
  --output outputs/evaluation/judge_run_1.csv \
  --model YOUR_EXPLICIT_JUDGE_MODEL_ID --replicate 1
python -m tescogpt judge-review \
  --review data/review/reply_review.csv \
  --review-manifest data/review/reply_review.manifest.json \
  --output outputs/evaluation/judge_run_2.csv \
  --model YOUR_EXPLICIT_JUDGE_MODEL_ID --replicate 2
```

Add both judge output paths to `judge_output_files` in the config.

## 7. Finalize evidence and reproduce

Replace every pending result in `REPORT.md` from generated artifacts, select the
five real failure modes from the failure ledger, and set config `status` to
`FINAL`. Never type an aggregate into the report without a generated source.
Only after that review, change the report's single
`SUBMISSION_STATUS: PENDING` marker to `SUBMISSION_STATUS: READY`. The preflight
also enforces all assignment sections and a conservative 2,400-word guard for
the six-page limit. A `READY` report is rejected if it retains placeholder or
future-result language, if Results does not name both baselines and cite
`outputs/evaluation/evidence_summary.md`, or if its first five numbered failure
modes lack a hypothesis and an example found in the frozen golden registry. It
separately verifies that the repository still contains
the reproducible README, 10–15 consecutively numbered decisions, citation and
AI-assistance disclosure, and the configured 150–250-example golden registry.

Use `outputs/evaluation/evidence_summary.md` as the cross-artifact fact sheet
when replacing pending prose. Verify any copied claim against its named source
JSON/CSV; never edit the generated summary itself.

```bash
python -m tescogpt status --require-final-ready
python -m tescogpt reproduce
python -m pytest
```

The final reproduction must report `COMPLETE`, pass every artifact hash, and
finish in under 15 minutes before the repository is tagged or submitted.
Its console JSON includes the measured `elapsed_seconds`; the tracked
`reproduction.json` intentionally stores only the 900-second limit and boolean
result so repeated successful runs are byte-stable.
The synthetic FINAL-path regression in `tests/test_reproduce_final.py` must also
remain green. It validates orchestration only; never cite its synthetic scores
as project results.

Do not change the `expected_*` counts in `config/experiment.json` to work around
missing ratings. They are the preregistered denominators; a changed protocol
requires an explicit decision-log entry and corresponding limitation in the
report.
