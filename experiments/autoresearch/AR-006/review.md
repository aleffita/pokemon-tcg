# AR-006 reviewer report

## Decision

**REWORK.** No P0 was found. Packed remains opt-in and is not sufficient for
self-play yet.

## Findings

- **P1:** the valid-identity resume test uses a recording optimizer and scalar
  phase assertions but never executes the production optimizer update or
  scheduler advancement. Fresh-run behavior with no resume path is also not
  covered.
- **P1:** the packed fixture calls the extracted loader and loss directly,
  manually maintaining memory. It does not exercise the production packed
  train prefetch/iterator or validation path, although call-site inspection
  confirms both paths use the extracted loader.
- **P2:** wrong-cwd rejection for a project-relative Stage 4 path is not a
  committed test. The implementation is safe and a read-only probe confirmed
  rejection from another cwd.

## Confirmed

The 47-test locked suite and `git diff --check` reproduce. Seed identity
coverage is correct, source/order/dedup/recurrent assertions are meaningful,
and production call sites use explicit cache dependencies for both packed train
prefetch and validation.

## Required next work

Add a bounded production-compatible micro update test covering optimizer state
mutation, scheduler advancement, and fresh/reset behavior. Exercise the actual
packed train prefetch and validation entry paths with a tiny fixture, and add
the wrong-cwd relative-root rejection case. Keep this a micro probe only, not a
long run or tournament.
