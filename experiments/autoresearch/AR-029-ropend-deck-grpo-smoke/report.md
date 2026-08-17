# AR-029 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T20:15:22.222847+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Result

The collector created multiple exact recurrent sibling bases per matchup.
Each base selected its own effective K from the legal action set, each matchup
was normalized independently, and groups were combined in one FP32 policy-only
optimizer step when relative signal existed. If every group was homogeneous,
the update emitted a root-equivalent no-op candidate and preserved the
zero-variance evidence. The frozen root remains the fallback pending tournament.

| Metric | Result |
| --- | ---: |
| Groups / fibers | 2 / 4 |
| Effective K per base | `[2, 2]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 284 / 303 |
| Collection seconds / decisions/s | 1.885961 / 150.5863555357566 |
| Grouped optimizer steps | 0 |
| Update seconds | 0.0 |
| Loss / gradient norm | 0.0 / 0.0 |
| Candidate parameter L2 delta | 0.0 |

## Contracts checked

- Every sibling group has one exact simulator snapshot, distinct legal branch
  actions, common branch provenance, and independent recurrent lanes.
- Effective K is dynamic per base: `min(K_max, legal branch actions)`.
- Deck and matchup strata normalize returns independently; no group is centered
  against another matchup's terminal distribution.
- The candidate uses one optimizer step over all signal-bearing groups, while
  each group's sibling-relative credit remains separate; an all-zero-variance
  matrix is explicitly fail-closed as a no-op.
- All rollouts run to terminal completion and continuation credit uses discount
  `0.97` without duplicating conditional substeps.
- Candidate preflight passed: `True`.

## Limitations and next gate

This is a bounded grouped prospective update, not a strength estimate. The
collection remains serial and the recurrent learner boundary is detached. Run
the controlled same-deck candidate-vs-root gate and the multi-opponent panel
before interpreting or promoting the candidate.

## Provenance

- Code commit at execution: `380fabb795e04e3223b44b661ea0be4dbb4ecc5c`
- Candidate SHA-256: `403687836083d2c848cee34d2e14df73df6025df797730395eba66a4327e095b`
- Sample manifest SHA-256: `ed1bd0e4a95dce470e9f18e97e0822c3b17c13b6a737428c00e2f535cedc9350`
- Trajectory bundle SHA-256: `3de3922814905047283eef9ec37ccb9487bbc96d855741a403dc8d299a073ff3`
- Tournament gate: pending
