# AR-030-I - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T20:49:54.691001+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Logical decisions / substeps | 637 / 672 |
| Collection seconds / decisions/s | 4.345669 / 146.58270998381093 |
| Grouped optimizer steps | 2 |
| Update seconds | 13.572289374889806 |
| Loss / gradient norm | 0.8683550536632538 / 2.353379487991333 |
| Candidate parameter L2 delta | 0.00594031438784034 |

## Contracts checked

- Every sibling group has one exact simulator snapshot, distinct legal branch
  actions, common branch provenance, and independent recurrent lanes.
- Effective K is dynamic per base: `min(K_max, legal branch actions)`.
- Deck and matchup strata normalize returns independently; no group is centered
  against another matchup's terminal distribution.
- The candidate uses `2` requested optimizer
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

- Code commit at execution: `e1be522360e4160eeb38a5b60bfd77d8eed52f72`
- Candidate SHA-256: `595527f6adc76c121749afd04244f0041e3931c4d43be87216bad671fceef26d`
- Sample manifest SHA-256: `ae6d1e6c464a3c5e8cd074df1a71387d43c2690bc1471ae8e46d07203a454bcb`
- Trajectory bundle SHA-256: `dd10e38083b1d3d2c18a0ea25385e955b7bfcbbc770917a35914884a51f19cd7`
- Tournament gate: pending
