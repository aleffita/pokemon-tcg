# AR-030 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T20:26:30.491276+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Groups / fibers | 4 / 8 |
| Effective K per base | `[2, 2, 2, 2]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 571 / 605 |
| Collection seconds / decisions/s | 3.932843 / 145.1875738708965 |
| Grouped optimizer steps | 1 |
| Update seconds | 5.779233708977699 |
| Loss / gradient norm | 0.8995583093166352 / 3.749363660812378 |
| Candidate parameter L2 delta | 0.0029680325250657975 |

## Contracts checked

- Every sibling group has one exact simulator snapshot, distinct legal branch
  actions, common branch provenance, and independent recurrent lanes.
- Effective K is dynamic per base: `min(K_max, legal branch actions)`.
- Deck and matchup strata normalize returns independently; no group is centered
  against another matchup's terminal distribution.
- The candidate uses `1` requested optimizer
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

- Code commit at execution: `376845b84cfdbe99c8a9b1141493be205d010a24`
- Candidate SHA-256: `4f34b531ea6024f557ab69cfc5eb11ac069700973714ab7e6228753d81553c02`
- Sample manifest SHA-256: `1671d28b39fd8c2075fbb4d2a6e1df317d2cad511889f489f054fa580923effe`
- Trajectory bundle SHA-256: `6201437db57ee29f6141d84211d57ad92c71b0a5b1d1762718717df5ae7f0a30`
- Tournament gate: pending
