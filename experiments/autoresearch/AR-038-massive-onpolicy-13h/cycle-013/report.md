# AR-038-C013 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-17T02:40:39.969203+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Groups / fibers | 52 / 140 |
| Effective K per base | `[3, 2, 2, 2, 3, 3, 2, 4, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 4, 3, 2, 4, 4, 2, 2, 4, 2, 4, 2, 4, 4, 4, 2, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 2, 4, 4, 2, 3]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 8201 / 8711 |
| Collection seconds / decisions/s | 68.645423 / 119.46899868676599 |
| Grouped optimizer steps | 3 |
| Update seconds | 665.6389077079948 |
| Loss / gradient norm | 0.867550181711639 / 2.497986078262329 |
| Candidate parameter L2 delta | 0.0006044805740616241 |

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
- Candidate SHA-256: `0ec57af31faf72ba9333c6253b305f4ca872715a058ebc671c7d7b9f3aae9ec6`
- Sample manifest SHA-256: `73a131284c0551b4a5f0eff928c4a933faa673b09e2d3e350bb4f49a27bccd81`
- Trajectory bundle SHA-256: `3d8738ce1c22655cdb5fea4bf9086515a157a2fdc3e04ef1de2f923311bdc6cf`
- Tournament gate: pending
