# AR-038-C022 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-17T04:20:04.696344+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Groups / fibers | 52 / 115 |
| Effective K per base | `[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 4]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 7149 / 7578 |
| Collection seconds / decisions/s | 51.891797 / 137.7674403109751 |
| Grouped optimizer steps | 3 |
| Update seconds | 545.8848356672097 |
| Loss / gradient norm | 0.8715684713688863 / 2.4990649223327637 |
| Candidate parameter L2 delta | 0.000604515225318246 |

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
- Candidate SHA-256: `a49350a1890d20138e40c89ffe675111842219123206d286b9439a0e431ebf06`
- Sample manifest SHA-256: `3b90d74973376e15c702b6132f2e5f4a4f7852b59538727fe1877e28c48f7ee2`
- Trajectory bundle SHA-256: `37c1500cd0cd3de8bf9f5b87b9ab02677308b9949385ea9e69d4a2592e966549`
- Tournament gate: pending
