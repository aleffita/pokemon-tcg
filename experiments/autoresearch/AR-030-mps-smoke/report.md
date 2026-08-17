# AR-030-MPS - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T20:28:51.778825+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.0 |
| Logical decisions / substeps | 451 / 484 |
| Collection seconds / decisions/s | 3.443439 / 130.97371552785157 |
| Grouped optimizer steps | 1 |
| Update seconds | 12.793072083033621 |
| Loss / gradient norm | 0.8925953477621079 / 3.10003662109375 |
| Candidate parameter L2 delta | 0.0029694303841592312 |

## Contracts checked

- Every sibling group has one exact simulator snapshot, distinct legal branch
  actions, common branch provenance, and independent recurrent lanes.
- Effective K is dynamic per base: `min(K_max, legal branch actions)`.
- Deck and matchup strata normalize returns independently; no group is centered
  against another matchup's terminal distribution.
- The candidate uses `1` requested optimizer
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

- Code commit at execution: `bbc964ca50149c66443484ac6411612bd8180e66`
- Candidate SHA-256: `159a750ae667cd126c2b7a38585d9cabea38ad980e98596631e516d016def9b6`
- Sample manifest SHA-256: `576fa28d701aca46aef053f06a23a49ac0776ccf00a25917d05e4f2c5e2dd4ad`
- Trajectory bundle SHA-256: `ecdd55ae79d582e238037cb4251edd5234d8c0c4c3d900ae3289d0da32052cff`
- Tournament gate: pending
