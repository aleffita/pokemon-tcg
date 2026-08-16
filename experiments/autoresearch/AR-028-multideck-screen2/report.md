# AR-028-multideck-screen2 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T19:43:54.850069+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Result

The collector created multiple exact recurrent sibling bases per matchup.
Each base selected its own effective K from the legal action set, each matchup
was normalized independently, and groups were combined in one FP32 policy-only
optimizer step when relative signal existed. If every group was homogeneous,
the update emitted a root-equivalent no-op candidate and preserved the
zero-variance evidence. The frozen root remains the fallback pending tournament.

| Metric | Result |
| --- | ---: |
| Groups / fibers | 8 / 21 |
| Effective K per base | `[2, 4, 2, 2, 2, 4, 3, 2]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 1285 / 1348 |
| Collection seconds / decisions/s | 11.668447 / 110.12605426982502 |
| Grouped optimizer steps | 1 |
| Update seconds | 4.707561375107616 |
| Loss / gradient norm | -0.05488505691447586 / 1.2298381328582764 |
| Candidate parameter L2 delta | 0.009211690045174248 |

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

- Code commit at execution: `9e2274f67be4bf8be4bb8281776439546b2be788`
- Candidate SHA-256: `bc07eb8507b86bdadebba1608681335d8dfc48cea5462fc547f725ad1f236300`
- Sample manifest SHA-256: `8f645f7d8ff7858d8bceb6d2e8555b9d78e5900b5b9fe2eee10412e09a1f899f`
- Trajectory bundle SHA-256: `c867ca38f61402a91f403edd59d16ad16e07f8a39a27ae0a1ece1c0296850c4e`
- Tournament gate: pending
