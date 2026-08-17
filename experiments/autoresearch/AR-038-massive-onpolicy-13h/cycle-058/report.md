# AR-038-C058 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-17T11:01:02.478156+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Groups / fibers | 56 / 146 |
| Effective K per base | `[2, 3, 4, 2, 4, 2, 3, 2, 2, 3, 3, 2, 2, 2, 2, 2, 4, 4, 2, 2, 3, 4, 2, 2, 2, 4, 3, 2, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4, 2, 4, 2, 4, 2, 2, 2, 3, 3, 2, 2, 2, 2, 4, 2]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 8286 / 8798 |
| Collection seconds / decisions/s | 56.128813 / 147.62471547104127 |
| Grouped optimizer steps | 3 |
| Update seconds | 549.201406000182 |
| Loss / gradient norm | 0.8051239304046238 / 1.4649325609207153 |
| Candidate parameter L2 delta | 0.0006042988957798189 |

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
- Candidate SHA-256: `40a4143cdf02b2f4dd4b653bb40681e809f1894361582a89bc5d11055f7b2270`
- Sample manifest SHA-256: `1898b05d21b04d6813ecbabf07bc8efc38e8a1180689c1a86ce3b66244bc9a7b`
- Trajectory bundle SHA-256: `ba07a5d3fd8bc168e9fa10fbc6af88afd9a3eb464a8a9141e711f078df91f4ec`
- Tournament gate: pending
