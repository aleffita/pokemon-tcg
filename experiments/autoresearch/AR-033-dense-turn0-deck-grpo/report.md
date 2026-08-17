# AR-033 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T21:47:05.141146+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Groups / fibers | 60 / 139 |
| Effective K per base | `[2, 2, 4, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 4, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 8266 / 8778 |
| Collection seconds / decisions/s | 58.223389 / 141.97043684604324 |
| Grouped optimizer steps | 1 |
| Update seconds | 1731.8672183749732 |
| Loss / gradient norm | 0.8985509578448623 / 2.902475595474243 |
| Candidate parameter L2 delta | 0.0030222521422162765 |

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

- Code commit at execution: `3893479b43bb3439418af23fafa3334a75e13be5`
- Candidate SHA-256: `65e4cc904d34a3a48356428ec7fdaa36fc2f8c920f07a4f4e873b1c52b4e8cdf`
- Sample manifest SHA-256: `d45c72e2abb2fde20d21fe580a3eca827b0760857635f7e4cf62615600f1c485`
- Trajectory bundle SHA-256: `ef87cf41bd0d6f7c7cbbb431af631184c4e727a1ad7977ccb6ff674b68fad14e`
- Tournament gate: pending
