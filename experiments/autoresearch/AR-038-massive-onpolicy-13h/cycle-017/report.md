# AR-038-C017 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-17T03:25:44.629417+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Groups / fibers | 52 / 104 |
| Effective K per base | `[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 6274 / 6675 |
| Collection seconds / decisions/s | 48.029281 / 130.6286483677962 |
| Grouped optimizer steps | 3 |
| Update seconds | 478.33253849996254 |
| Loss / gradient norm | 0.8944525424863897 / 2.7391409873962402 |
| Candidate parameter L2 delta | 0.0006045006356399151 |

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
- Candidate SHA-256: `cc64760c5a393bef817e8774a3f4e318a0f05ed90d4983fccf94ede412bc6bb3`
- Sample manifest SHA-256: `b203f1ed79c2017eb343286619e984c0d6ec08929a4e5e8e55db2c6069e0853e`
- Trajectory bundle SHA-256: `e39f184805c878232407d0777815cef7e0edf35ac7a2b31fb1a7473e84cd5184`
- Tournament gate: pending
