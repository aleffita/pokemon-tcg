# AR-029 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T20:20:30.975109+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Groups / fibers | 24 / 72 |
| Effective K per base | `[4, 2, 4, 2, 2, 3, 4, 3, 4, 2, 2, 2, 2, 4, 4, 4, 2, 2, 4, 2, 4, 4, 2, 4]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 4915 / 5212 |
| Collection seconds / decisions/s | 28.020018 / 175.41030896531655 |
| Grouped optimizer steps | 8 |
| Update seconds | 177.4268408329226 |
| Loss / gradient norm | 0.06782883864182693 / 0.3231911361217499 |
| Candidate parameter L2 delta | 0.028932774796952897 |

## Contracts checked

- Every sibling group has one exact simulator snapshot, distinct legal branch
  actions, common branch provenance, and independent recurrent lanes.
- Effective K is dynamic per base: `min(K_max, legal branch actions)`.
- Deck and matchup strata normalize returns independently; no group is centered
  against another matchup's terminal distribution.
- The candidate uses `8` requested optimizer
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

- Code commit at execution: `884274321d5d77545430e35d77352e2f9596f84c`
- Candidate SHA-256: `68dd7d95d5e9918ffdd3535b96305c00785980715a5c1aaa325471ee2eed0c99`
- Sample manifest SHA-256: `eadff64c41f58ad0b03a0d86a38cc5eba46fac9d01d78f977d1e6e296590198a`
- Trajectory bundle SHA-256: `34fd2356603fa9b921c0accd710cfb18b0b9cda1dbded3e43432f081644e414c`
- Tournament gate: pending
