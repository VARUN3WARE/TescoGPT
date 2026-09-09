# Decision log

This is a living record. Decisions are added when they are made, not reconstructed
after the results are known.

1. **Use Tesco as the single brand.** It has sufficient volume and a useful mix
   of routine and safety-sensitive support, so selective escalation is meaningful.
2. **Define auto-handle as permission to send a public draft.** The project has
   no backend tools and must not imply that it completed refunds, order changes,
   or account actions.
3. **Treat historical messages as precedents, not automatically as resolutions.**
   A later customer message or manual review is required before a case receives a
   `verified_success` outcome label.
4. **Use complete conversations as the unit of splitting.** Individual tweet
   splits can place adjacent turns in training and evaluation and inflate both
   retrieval and generation quality.
5. **Keep a natural slice and a challenge slice separate.** A balanced stress set
   is useful for finding failures but cannot estimate real traffic performance.
6. **Make risk at a stated coverage the routing headline.** Escalation accuracy is
   misleading because an always-escalate system can score well while automating
   nothing.
7. **Freeze human labels before threshold tuning.** Development thresholds will
   use a separate development set; the golden set is for final evaluation and
   regression checks.
8. **Prefer deterministic safety checks around probabilistic models.** Public PII
   requests, unsupported commitments, and critical safety phrases should not
   depend only on an LLM confidence value.
9. **Do not use Banking77.** Its banking taxonomy does not match Tesco operations;
   adding it would increase apparent sophistication without improving validity.
10. **Prioritize an offline reproduction path over a hosted demo.** Reviewers must
    be able to verify the evidence quickly even if an external model API changes.

