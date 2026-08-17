# AR-038-C001 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-17T00:10:51.060148+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Groups / fibers | 52 / 160 |
| Effective K per base | `[4, 2, 4, 4, 2, 2, 3, 2, 4, 3, 4, 3, 3, 2, 3, 4, 2, 4, 4, 4, 4, 4, 3, 2, 4, 2, 4, 4, 2, 2, 4, 3, 2, 4, 2, 2, 4, 4, 2, 4, 4, 4, 2, 4, 4, 4, 3, 2, 2, 2, 2, 2]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 9745 / 10342 |
| Collection seconds / decisions/s | 54.015493 / 180.41120081943302 |
| Grouped optimizer steps | 3 |
| Update seconds | 617.3764402091037 |
| Loss / gradient norm | 0.8923792303003291 / 2.5308525562286377 |
| Candidate parameter L2 delta | 0.0006047407859426242 |

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
- Candidate SHA-256: `a19cbc81b3d97611fc3c7a02297ba4f30bfe7bc058c4af74e187effd9c5fcafb`
- Sample manifest SHA-256: `2e881241f1a8269219f8c0c1cf90f0fbec21a38a43ff5d45124ffc98d3f7a908`
- Trajectory bundle SHA-256: `db616bc1cf5d9f369a2ef0f7bace260bdba56682396c0c6c530bddc1ff5b0c84`
- Tournament gate: pending
