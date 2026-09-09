# Sources, attribution, and assistance disclosure

This file distinguishes source material and borrowed methods from work created
for TescoGPT. The project-specific Python implementation, prompts, schemas,
tests, and prose were written for this repository; no third-party implementation
code was intentionally copied.

An AI coding assistant was used extensively for planning, implementation,
testing, research, and editing. The submitter is responsible for understanding,
verifying, and modifying the resulting work. AI output must not be used to fill
the two independent annotation rounds, adjudication fields, retrieval relevance
labels, or human reply ratings: those artifacts are deliberately still marked
incomplete until people perform the reviews.

## Data and product context

- Thought Vector and collaborators, [Customer Support on
  Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).
  The message text and conversation graph come from this dataset. The raw file
  is not redistributed, and its separate Kaggle license and terms apply.
- Hiver, [The AI trust gap in customer
  support](https://hiverhq.com/blog/ai-trust-gap-in-support). This informed the
  product framing only; it is not evaluation evidence.
- The optional Banking77 dataset was not used. This avoids importing a banking
  taxonomy into a supermarket-support problem.

## Methods

- Robertson, S. and Zaragoza, H. (2009), [The Probabilistic Relevance
  Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019),
  *Foundations and Trends in Information Retrieval*, 3(4), 333–389. The BM25
  ranking concept and formula are borrowed; the small deterministic
  implementation in `src/tescogpt/retrieval/bm25.py` was written for this repo.
- Järvelin, K. and Kekäläinen, J. (2002), [Cumulated Gain-Based Evaluation of
  IR Techniques](https://doi.org/10.1145/582415.582418), *ACM Transactions on
  Information Systems*, 20(4), 422–446. This is the basis for the reported
  nDCG@3 retrieval metric.
- Efron, B. (1979), [Bootstrap Methods: Another Look at the
  Jackknife](https://doi.org/10.1214/aos/1176344552), *The Annals of
  Statistics*, 7(1), 1–26. Bootstrap resampling is borrowed; TescoGPT freezes
  its own paired, intent-stratified sampling and seeds in the evaluation
  protocol.
- Wilson, E. B. (1927), [Probable Inference, the Law of Succession, and
  Statistical Inference](https://doi.org/10.1080/01621459.1927.10502953),
  *Journal of the American Statistical Association*, 22(158), 209–212. This is
  the basis for binomial intervals and the one-sided upper safety bound.
- Cohen, J. (1960), [A Coefficient of Agreement for Nominal
  Scales](https://doi.org/10.1177/001316446002000104), *Educational and
  Psychological Measurement*, 20(1), 37–46. This is the basis for chance-
  corrected agreement on categorical labels and overall reply pass/fail.
- Cohen, J. (1968), [Weighted Kappa: Nominal Scale Agreement with Provision for
  Scaled Disagreement or Partial Credit](https://doi.org/10.1037/h0026256),
  *Psychological Bulletin*, 70(4), 213–220. This is the basis for quadratic-
  weighted agreement on ordinal reply scores.
- Xin, J., Tang, R., Yu, Y. and Lin, J. (2021), [The Art of Abstention:
  Selective Prediction and Error Regularization for Natural Language
  Processing](https://aclanthology.org/2021.acl-long.84/), ACL-IJCNLP. This
  paper's abstention framing influenced the decision to report risk against
  automation coverage instead of accuracy alone.
- Shi, L. et al. (2025), [Judging the Judges: A Systematic Study of
  Position Bias in LLM-as-a-Judge](https://aclanthology.org/2025.ijcnlp-long.18/),
  IJCNLP-AACL. This informed the decision to blind system identity, add controls
  and unsafe decoys, repeat judge runs, and keep the judge advisory unless it
  agrees sufficiently with humans.

These citations acknowledge the underlying ideas. TescoGPT's exact operational
definitions, thresholds, resampling strata, tie handling, degenerate-case
behavior, and trust gates are pre-registered in `docs/EVALUATION.md` and
`docs/JUDGE_RUBRIC.md` and tested in this repository.

## APIs and software

- The optional structured drafter and reply judge follow OpenAI's [Responses
  API reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
  and [Structured Outputs
  guide](https://developers.openai.com/api/docs/guides/structured-outputs), and
  use the pinned official Python package. No model response is represented as
  a human label, and no API-generated result is currently claimed.
- Runtime dependencies are NumPy and pandas; development uses pytest and Ruff.
  Exact compatible ranges are declared in `pyproject.toml`. Their respective
  licenses apply to those packages; TescoGPT's own source is MIT licensed.

## Dataset-derived material in the repository

Small frozen candidate, evidence, and review artifacts contain anonymized text
derived from Customer Support on Twitter. Customer identifiers remain
anonymized, no re-identification is attempted, and examples are included only
to make the evaluation auditable. Historical Tesco replies are precedents, not
current policy or proof that an issue was resolved.
