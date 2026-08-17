# AR-038-C054 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-17T10:15:46.722120+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

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
| Groups / fibers | 56 / 134 |
| Effective K per base | `[2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 3, 2, 2, 4, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 3, 2, 2, 2, 4]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 7523 / 8012 |
| Collection seconds / decisions/s | 47.681614 / 157.77569945860373 |
| Grouped optimizer steps | 3 |
| Update seconds | 509.4457824998535 |
| Loss / gradient norm | 0.8252198741084551 / 1.3844183683395386 |
| Candidate parameter L2 delta | 0.0006043010671068488 |

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
- Candidate SHA-256: `15bed68ab5b43496e274f9ab19bf440e04897666599afb79bd461e563447447d`
- Sample manifest SHA-256: `9962bc566493d71e19dbb1ecd3a6b824c6cd1d6cdadb454d73c5700f02b054ba`
- Trajectory bundle SHA-256: `c8f62fbdc7414840d4c65917eeda20ceafa6d8b08d73c3b86615668ba7f7164f`
- Tournament gate: pending
