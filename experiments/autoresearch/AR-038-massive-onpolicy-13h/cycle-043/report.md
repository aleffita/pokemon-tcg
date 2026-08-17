# AR-038-C043 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-17T08:13:16.897508+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Groups / fibers | 52 / 135 |
| Effective K per base | `[4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 3, 2, 2, 4, 3, 2, 2, 4, 4, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 3, 4, 2, 2, 4, 4, 2, 2, 4]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 8013 / 8501 |
| Collection seconds / decisions/s | 49.980833 / 160.32145667298252 |
| Grouped optimizer steps | 3 |
| Update seconds | 561.3817308750004 |
| Loss / gradient norm | 0.8463385661524285 / 1.5099130868911743 |
| Candidate parameter L2 delta | 0.0006042051100746132 |

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
- Candidate SHA-256: `f9cad3da15a169114cc7dd90a2d8731713d9751dfa6d4029520bcfbfdf6730ef`
- Sample manifest SHA-256: `83dbf32bce80e6271b95bac13d0703ce55771dbdac4a2e767b1c5cbb20a58bd2`
- Trajectory bundle SHA-256: `b4bd8916f7060723429b24e6a9f25f06819fdfc6834d982b7b9044d946291730`
- Tournament gate: pending
