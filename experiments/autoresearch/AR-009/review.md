# AR-009 reviewer report

## Decision

**REWORK BEFORE TOURNAMENT.** No P0 was found. The PPO micro-update and its
artifacts are mathematically coherent, but the candidate path is not yet safe
to execute competitively.

## Findings

- **P1:** the opt-in candidate loads during module import but fails on its first
  real `choose()` call with `KeyError: 'prospective_planner'`. The loader
  currently strips that stale planner field while the selected public agent
  accesses it directly.
- **P1:** candidate provenance (`root_sha256` and sample-manifest hash) is
  recorded in the payload but not enforced by the opt-in agent. A stale or
  detached candidate could therefore be loaded.
- **P2:** the sample manifest digests masks and memory but not all model-input
  tensors; input-feature tampering would not be detected.
- **P2:** bundle provenance is logged but not linked into the candidate
  payload, and the generated candidate/bundle/manifest artifacts remain
  untracked working-tree artifacts rather than commit contents.

## Confirmed

Root hash is checked before strict loading, both metadata date fields are
validated, bundle ordering/masks/memory/rows pass, the candidate is strict
loadable with 94 FP32 tensors, and the PPO objective has coherent terminal
returns, normalized advantages, clipped ratios, value loss, detached recurrent
boundaries, parameter mutation, and root KL diagnostics.

## Required next work

Fix the first-decision opt-in path, enforce candidate root/sample/bundle
provenance, include complete input digests, and only then run a named smoke
tournament. Keep the root and default agent behavior unchanged.
