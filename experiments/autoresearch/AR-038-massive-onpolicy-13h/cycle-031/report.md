# AR-038-C031 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-17T06:03:05.054750+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Groups / fibers | 52 / 123 |
| Effective K per base | `[2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 4, 3, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 3]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 7819 / 8309 |
| Collection seconds / decisions/s | 51.478686 / 151.88810239787102 |
| Grouped optimizer steps | 3 |
| Update seconds | 570.5586287498008 |
| Loss / gradient norm | 0.8747420353882377 / 1.965256929397583 |
| Candidate parameter L2 delta | 0.0006043559044243805 |

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
- Candidate SHA-256: `ee9202c9a1ca56c593d8f05918d53266f8a9a9745d5b03baa8715805c84fcf42`
- Sample manifest SHA-256: `24e76803e73f53c61bdcd120fae3e3b17a4c4d8bf765d65fa2d6b56f56ba39d2`
- Trajectory bundle SHA-256: `04f92b3e8ba6f8730b024b25e45b332e896fafbbe2e7f546dd80800c8997d0e7`
- Tournament gate: pending
