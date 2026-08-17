# AR-032 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T21:05:30.822794+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Groups / fibers | 40 / 85 |
| Effective K per base | `[3, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 5100 / 5373 |
| Collection seconds / decisions/s | 36.036445 / 141.52339536960096 |
| Grouped optimizer steps | 4 |
| Update seconds | 735.0039874590002 |
| Loss / gradient norm | 0.8395435106754304 / 3.308471441268921 |
| Candidate parameter L2 delta | 0.011920823066721242 |

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

- Code commit at execution: `76201a1e4e8714f6a39c26fa4e99736236e6bf0d`
- Candidate SHA-256: `c1a9dda365216752024e983500688b8a552dbeb8c5ad8cf96638a5f23eb6760b`
- Sample manifest SHA-256: `fb5bbaf0b3c966d4804e6158a838efc51b77ae5006332329e15e5f9d091f138e`
- Trajectory bundle SHA-256: `aaf292e91eaa428f8a66a4ff0d401479861829a494146d56ed40a7a53559a478`
- Tournament gate: pending
