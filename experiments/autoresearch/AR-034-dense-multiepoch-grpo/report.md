# AR-034 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T22:18:28.420900+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Groups / fibers | 20 / 46 |
| Effective K per base | `[2, 2, 2, 4, 2, 2, 2, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 2601 / 2753 |
| Collection seconds / decisions/s | 18.002045 / 144.48358104751583 |
| Grouped optimizer steps | 5 |
| Update seconds | 1724.1197738749906 |
| Loss / gradient norm | 0.8545255827557314 / 3.116802215576172 |
| Candidate parameter L2 delta | 0.005021008527245035 |

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

- Code commit at execution: `3893479b43bb3439418af23fafa3334a75e13be5`
- Candidate SHA-256: `c5768125554f5038a60c3945c63ced949e2fa96ff0908561e2f8ee7ec88ec1cf`
- Sample manifest SHA-256: `89e8c9f978e34b01531e12394d8cfed45d42f3a975addcd16a382dd54927ab89`
- Trajectory bundle SHA-256: `312240a75cdd539d587812ee84ce13fc8f06ed0d47b022a5503edd169d8b7ed6`
- Tournament gate: pending
