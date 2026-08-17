# AR-028-lucario-targeted - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T19:54:59.714265+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Effective K per base | `[2, 2, 4, 3, 2, 2, 4, 2]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 1322 / 1389 |
| Collection seconds / decisions/s | 12.972843 / 101.90518434839396 |
| Grouped optimizer steps | 1 |
| Update seconds | 7.34693295811303 |
| Loss / gradient norm | -0.0035466413137689974 / 0.43632370233535767 |
| Candidate parameter L2 delta | 0.009207524589517763 |

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

- Code commit at execution: `1284066a3bbd78de285e99613593867a81a83f8f`
- Candidate SHA-256: `c197aad9937dd0f8500943631c3ceb02fa696a969a197eb7e37ebbe4f2ad1274`
- Sample manifest SHA-256: `eca83d3b3bf92277bb21c1d6de5ca18bdeafc04aba5a37a4ae62a445215dcc91`
- Trajectory bundle SHA-256: `e57a03a37299de27c67b762be277ae45deeaab0ed040e5951c3848a3d3c399dc`
- Tournament gate: pending
