# AR-037 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T23:45:56.582793+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Groups / fibers | 52 / 144 |
| Effective K per base | `[4, 2, 4, 3, 4, 2, 2, 3, 2, 2, 4, 2, 4, 2, 4, 2, 2, 2, 4, 3, 2, 2, 3, 4, 2, 2, 2, 2, 4, 2, 3, 4, 2, 2, 3, 4, 4, 2, 2, 2, 2, 2, 4, 4, 2, 2, 2, 3, 4, 2, 3, 4]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 8382 / 8906 |
| Collection seconds / decisions/s | 55.495979 / 151.0379694427033 |
| Grouped optimizer steps | 7 |
| Update seconds | 1364.7855845841113 |
| Loss / gradient norm | 0.8554481943765974 / 2.713228464126587 |
| Candidate parameter L2 delta | 0.0021103103700253795 |

## Contracts checked

- Every sibling group has one exact simulator snapshot, distinct legal branch
  actions, common branch provenance, and independent recurrent lanes.
- Effective K is dynamic per base: `min(K_max, legal branch actions)`.
- Deck and matchup strata normalize returns independently; no group is centered
  against another matchup's terminal distribution.
- The candidate uses `20` requested optimizer
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

- Code commit at execution: `98a9ac150a864217a909d1b4ba7e7f76001ebbfa`
- Candidate SHA-256: `3f757f006ee0537915d2894f8c13851c013e21625e3039be228b06c69c17ec02`
- Sample manifest SHA-256: `0d4f96c9733372a5e65fd8676fcd0e532817bde11161a61229596f1ebb9ebe0e`
- Trajectory bundle SHA-256: `227cc95003adcebe0ddc13fa2db764c0a1c8c8d6e37ad6b1347107eefafa0301`
- Tournament gate: pending
