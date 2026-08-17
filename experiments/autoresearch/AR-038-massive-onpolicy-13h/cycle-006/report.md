# AR-038-C006 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-17T01:08:42.213143+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Groups / fibers | 52 / 127 |
| Effective K per base | `[2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 3, 2, 4, 2, 4, 2, 4, 2, 2, 2, 2, 2, 4, 2, 4, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 4, 2, 3, 2, 2, 2, 2, 2, 4]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 7177 / 7635 |
| Collection seconds / decisions/s | 62.586682 / 114.67295808966898 |
| Grouped optimizer steps | 3 |
| Update seconds | 575.2165405410342 |
| Loss / gradient norm | 0.8691146819427855 / 2.3242244720458984 |
| Candidate parameter L2 delta | 0.0006043353230399892 |

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
- Candidate SHA-256: `4b2a1232d7f362b214d9fb055a3893bf63b243995c4e4feca3c4e15aef027970`
- Sample manifest SHA-256: `ad5cdefb8d73de7c7de74f1bc4a4de34d439faa1e250f0d226f38fc95e618628`
- Trajectory bundle SHA-256: `26d38431ca3349b9bc7afce04de6928f54c0fd9e1233804e174e2327047ff258`
- Tournament gate: pending
