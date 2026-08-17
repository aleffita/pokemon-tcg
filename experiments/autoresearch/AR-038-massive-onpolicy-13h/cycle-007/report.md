# AR-038-C007 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-17T01:21:04.268170+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Result

The collector created multiple exact recurrent sibling bases per matchup.
Each base selected its own effective K from the legal action set, launched all K
continuations concurrently, and combined sibling-relative credit with paired
inter-deck credit across equal opponent/group seeds. The frozen behavior data
is reused for multiple FP32 policy-only epochs when relative signal exists. If
every sibling and deck cohort was homogeneous,
the update emitted a root-equivalent no-op candidate and preserved the
zero-variance evidence. The frozen root remains the fallback pending tournament.

| Metric | Result |
| --- | ---: |
| Groups / fibers | 52 / 127 |
| Effective K per base | `[2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 3, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 3, 2, 2, 2, 2, 2, 4, 2, 2, 2]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 7314 / 7724 |
| Collection seconds / decisions/s | 63.51755 / 115.1492773324805 |
| Grouped optimizer steps | 3 |
| Update seconds | 600.3802286658902 |
| Loss / gradient norm | 0.8731357558061448 / 2.7249300479888916 |
| Candidate parameter L2 delta | 0.0006044452777167687 |

## Contracts checked

- Every sibling group has one exact simulator snapshot, distinct legal branch
  actions, common branch provenance, and independent recurrent lanes.
- Effective K is dynamic per base: `min(K_max, legal branch actions)`.
- Deck and matchup strata normalize returns independently; no group is centered
  against another matchup's terminal distribution.
- The candidate uses `3` requested optimizer
  epochs over all signal-bearing groups, while each group's sibling-relative
  credit remains separate; an all-zero-signal matrix is explicitly fail-closed.
- All K sibling futures execute simultaneously after the recurrent branch base
  is fixed; no polling or scheduler participates in process completion.
- All rollouts run to terminal completion and continuation credit uses discount
  `0.97` without duplicating conditional substeps.
- Candidate preflight passed: `True`.

## Limitations and next gate

This is a bounded grouped prospective update, not a strength estimate. Groups
are collected sequentially while sibling games within each group are parallel;
the recurrent learner boundary is detached. Run
the controlled same-deck candidate-vs-root gate and the multi-opponent panel
before interpreting or promoting the candidate.

## Provenance

- Code commit at execution: `74b8befab00c8514699a6054cc434e10527c9690`
- Candidate SHA-256: `dc972b61cfd355e0312363e2364cd0990d5641600770b1e836ea9b3c698da197`
- Sample manifest SHA-256: `b8f09a64a5bf880d0ef9cc8ea754e73948aad37d8a18a9a2e853b6145bb7ef91`
- Trajectory bundle SHA-256: `be55fb2c814ae7a4fe05ace1993bb20fea27ee663119c7639b606c69f0ad18ce`
- Tournament gate: pending
