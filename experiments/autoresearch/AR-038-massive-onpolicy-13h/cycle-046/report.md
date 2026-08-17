# AR-038-C046 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-17T08:46:32.395016+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Groups / fibers | 52 / 131 |
| Effective K per base | `[4, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 3, 2, 2, 2, 4, 3, 2, 2, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4, 4, 2, 2, 4, 3, 2, 2]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 7670 / 8175 |
| Collection seconds / decisions/s | 49.218061 / 155.83710319778464 |
| Grouped optimizer steps | 3 |
| Update seconds | 536.1700510829687 |
| Loss / gradient norm | 0.8458258560028635 / 1.5143389701843262 |
| Candidate parameter L2 delta | 0.0006041729093548587 |

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
- Candidate SHA-256: `61cf766fb79090ab487416b3f909fb0c37b8cd98f9cad2c4ba66e9eecfcc0bcd`
- Sample manifest SHA-256: `54184cf7f44826cd66092a354d0c5518d882840be3f5e12700924f07c42c852c`
- Trajectory bundle SHA-256: `ba2069df63e0a2da7170b79fd4b10d3380a49c4773ae3b99ed401a92f6f9e758`
- Tournament gate: pending
