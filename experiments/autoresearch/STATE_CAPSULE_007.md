# State Capsule 007 - AR-005 reviewed

Captured: 2026-08-16 after the AR-005 reviewer gate.

## Decision

Stage 4 remains the frozen competitive root and only promoted policy. Packed
data remains opt-in infrastructure. AR-005 closes exact-root warm-start gating,
identity mismatch rejection, and reset behavior, but is **rework** because the
production behavioral integration evidence is incomplete.

## Evidence

- Commit: `0afaa49`.
- Focused locked suite: 41 passed in 0.62 s.
- Review: `experiments/autoresearch/AR-005/review.md`.
- Semantic baseline suite: 4 passed, 2 failed for missing
  `structured_weight_decay` and missing real smoke corpus.
- Reviewer confirmed runtime `main()` calls the new resume helpers, but the
  committed tests do not exercise the actual packed temporal loader/training
  iterator or a positive optimizer+scheduler continuation.

## Open gates

AR-006 must add a valid-identity resume continuation test, exercise the actual
packed temporal loader with dedup and sequential TBPTT, cover canonical,
relative, absolute, copy, hardlink, and symlink root cases, and correct the
seed wording/no-op test residue. No packed self-play or RL run is allowed until
these gates are reviewed.

## Next control point

After AR-006 review, inspect the existing self-play/RL implementation and run a
small on-policy trajectory/log-probability probe from the frozen Stage 4 root,
using default Parquet unless packed is explicitly promoted.
