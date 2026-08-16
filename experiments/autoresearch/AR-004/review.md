# AR-004 reviewer report

## Decision

**REWORK.** No P0 was found. Runtime packed validation is mostly sound, but
the evidence does not yet justify promoting the backend or using it for
self-play.

## Findings

- **P1:** resume safety is not tested through the production trainer path. The
  unit test directly enables the legacy warm-start flag and does not prove the
  CLI/state gating, optimizer reset, or scheduler reset behavior.
- **P2:** the Stage 4 exception is basename-based (`stage4_root.pkl`) rather
  than bound to the frozen artifact path and SHA-256.
- **P2:** the claimed inverted-source-order test does not validate a correct
  `[A, B]` manifest against `[B, A]` input.
- **P2:** the seed-variant test does not vary seeds and does not test resume
  seed mismatch.
- **P2:** dedup coverage validates metadata only; no packed fixture exercises
  `opt_group` relabeling with a sequential TBPTT batch.
- **P2:** the additional progress-test collection still fails because the
  preserved baseline lacks `_standard_microbatch_count`; this is outside the
  AR-004 diff but keeps the repository validation state non-green.

## Confirmed

The production path opens all required trainer and order columns. Boundary,
split membership, row-order digests, ordered source digest, and no-fallback
packed/TBPTT checks are wired and the focused suite passes 19 tests.

## Required next work

Add a real trainer resume integration fixture covering the frozen Stage 4 root
and non-root legacy checkpoints, reset versus resume optimizer/scheduler
states, partial identity, all identity mismatches, and actual state-reset
assertions. Bind the exception to the approved root artifact hash. Correct the
source-order and seed tests, and add packed `opt_group` plus sequential TBPTT
behavioral coverage. Keep the backend opt-in until these pass.
