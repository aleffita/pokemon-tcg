# AR-005 reviewer report

## Decision

**REWORK.** No P0 was found. The implementation is largely sound, but the
behavioral integration evidence is narrower than the report claims.

## Findings

- **P1:** the tests call resume helpers directly; packed TBPTT tests manually
  call relabel and plan builders rather than exercising `_load_temporal_batch`
  or the actual training iterator.
- **P1:** there is no committed positive resume test with a matching identity
  that continues both optimizer and scheduler state. Existing positive
  assertions cover only the identity helper or reset behavior.
- **P2:** Stage 4 tests cover canonical and hardlink cases but not the complete
  regular-copy, symlink, absolute, and repository-relative matrix.
- **P2:** the seed-variant split wording is overstated because that test uses
  identical seedless arguments; seed mismatch is covered separately in resume
  identity tests.
- **P2:** `tests/test_packed_data.py` contains a dangling no-op expression.
- **Baseline validation:** focused AR-005 tests pass 41; the semantic suite
  remains 4 passed and 2 baseline failures for missing `structured_weight_decay`
  and missing real smoke corpus.

## Confirmed strengths

Exact Stage 4 path and SHA gating is fail-closed for canonical, copied,
hardlinked, and symlinked alternate paths. Optimizer reset starts from zero,
scheduler reset returns phase zero, identity mismatches fail closed, and the
reported focused logs reproduce.

## Required next work

Add a positive valid-identity resume integration test, exercise the real packed
temporal loader/training path with dedup and sequential TBPTT, parameterize the
Stage 4 path cases, correct the seed wording/test, and remove the dangling
expression. Keep the packed backend opt-in until these checks pass.
