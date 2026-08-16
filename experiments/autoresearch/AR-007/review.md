# AR-007 reviewer report

## Decision

**KEEP AS OPT-IN, REWORK BEFORE PROMOTION.** No P0 was found. The refactor
preserves the optimizer semantics and the packed helpers are wired into the
production call sites, but coverage is not sufficient to promote packed or use
it as the default self-play data plane.

## Findings

- **P1:** the real optimizer micro-test mutates parameters and optimizer state,
  but the continuation case still uses a recording optimizer and starts real
  updates at phase zero. A real nonzero-phase continuation update is missing.
- **P1/P2:** packed iterator coverage remains helper-level. No committed test
  covers empty plans, auxiliary targets, `zero_wouldko`, validation memory
  continuity, or exact fetched-row values. No test invokes `main()` directly;
  production wiring is verified by call-site inspection.
- **P2:** the wrong-cwd relative-root rejection test is valid and passes.

## Confirmed

The extracted optimizer step retains normalization, clipping, LR scheduling,
optimizer update, and MLX evaluation order. The one-worker lookahead structure
and cache row-offset handling match the prior nested implementation. The
focused suite reproduces 51 passed and `git diff --check` passes.

## Decision for the research loop

Keep AR-007 and the packed backend opt-in. Do not promote packed or use it for
self-play yet. This does not block a Stage 4 plus default-Parquet self-play
probe; the packed promotion work is a separate infrastructure rework track.

## Deferred promotion work

Add a real nonzero-phase optimizer continuation test and packed coverage for
empty plans, auxiliary targets, `zero_wouldko`, validation memory continuity,
and exact fetched-row values before making packed the default or using it for
on-policy data collection.
