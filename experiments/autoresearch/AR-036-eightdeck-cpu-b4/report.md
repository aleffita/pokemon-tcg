# AR-036 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T23:07:08.665600+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Groups / fibers | 32 / 88 |
| Effective K per base | `[4, 4, 4, 2, 2, 2, 4, 2, 4, 3, 2, 2, 4, 3, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 4, 4, 2, 2, 2, 2, 2]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 5173 / 5461 |
| Collection seconds / decisions/s | 36.53998 / 141.57095823029078 |
| Grouped optimizer steps | 10 |
| Update seconds | 1223.1172902078833 |
| Loss / gradient norm | 0.8854190839186705 / 2.6759033203125 |
| Candidate parameter L2 delta | 0.0030126902571863813 |

## Contracts checked

- Every sibling group has one exact simulator snapshot, distinct legal branch
  actions, common branch provenance, and independent recurrent lanes.
- Effective K is dynamic per base: `min(K_max, legal branch actions)`.
- Deck and matchup strata normalize returns independently; no group is centered
  against another matchup's terminal distribution.
- The candidate uses `12` requested optimizer
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

- Code commit at execution: `545bbe3bda78f04a582998af882ccab14056d4e4`
- Candidate SHA-256: `ae1681e0d5bc87a4f76926b19292d7553523fceb194dca59f38a40815f6c466c`
- Sample manifest SHA-256: `ecb4a1bebe41c1a6a160d388e2255386468c350abfbf3b7f7f025f8a7da89fac`
- Trajectory bundle SHA-256: `ffeb32d6af13d1dc5eaee6196c155d1aaba1a6504b58790e65d4809bfe0004eb`
- Tournament gate: pending
