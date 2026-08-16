# AR-025 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T18:35:57.057264+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Result

The collector created multiple exact recurrent sibling bases per matchup.
Each base selected its own effective K from the legal action set, each matchup
was normalized independently, and all groups were combined in one FP32
policy-only optimizer step with terminal credit through discounted future
logical decisions. The frozen root remains the fallback after the tournament
gate.

| Metric | Result |
| --- | ---: |
| Groups / fibers | 16 / 43 |
| Effective K per base | `[4, 2, 4, 2, 2, 2, 2, 2, 2, 2, 3, 4, 2, 2, 4, 4]` |
| Logical decisions / substeps | 2914 / 3061 |
| Collection seconds / decisions/s | 27.275887 / 106.83428855216941 |
| One grouped optimizer step | 1 |
| Update seconds | 66.6324357080739 |
| Loss / gradient norm | -0.0009624201920814812 / 0.7253366112709045 |
| Candidate parameter L2 delta | 0.009227449781494418 |

## Contracts checked

- Every sibling group has one exact simulator snapshot, distinct legal branch
  actions, common branch provenance, and independent recurrent lanes.
- Effective K is dynamic per base: `min(K_max, legal branch actions)`.
- Deck and matchup strata normalize returns independently; no group is centered
  against another matchup's terminal distribution.
- The candidate uses one optimizer step over all groups, while each group's
  sibling-relative credit remains separate.
- All rollouts run to terminal completion and continuation credit uses discount
  `0.97` without duplicating conditional substeps.
- Candidate preflight passed: `True`.

## Limitations and next gate

This is a bounded grouped prospective update, not a strength estimate. The
collection remains serial and the recurrent learner boundary is detached. The
scaled candidate lost both the direct root gate and the six-opponent panel,
so it is rejected for promotion and retained as negative evidence for this
scale.

## Tournament gate

| Surface | Candidate | Frozen root | Decision |
| --- | ---: | ---: | --- |
| Same-deck vs root, n=30 | `13-17-0` (43.3%) | n/a | candidate loses |
| Six-opponent panel, n=60 | `10-50-0` (16.7%) | `12-48-0` (20.0%) | reject promotion |

Candidate panel detail: `lb1009` `0-10`, `lb945` `0-10`, `lb826` `1-9`,
`lb814` `3-7`, `random` `3-7`, `first` `3-7`. Increasing group count did
not improve absolute field strength and eleven of sixteen groups were
zero-variance.

## Provenance

- Code commit at execution: `c051b8cd234a7791c366cf58e55e107d90a51745`
- Candidate SHA-256: `93b46cb113c917d4ea12cb25eb0bdcc7ca6ce31fbdd2ad71e6c5e2f31455bb52`
- Sample manifest SHA-256: `d6305fb4123336728ec19df6338f949a4c9db6a0edcf0acb2f1017176466d693`
- Trajectory bundle SHA-256: `42a62071f219842704f8cb18218ad3253b650c50928a9ee2039316cff5a7b333`
- Tournament candidate-vs-root report: `experiments/autoresearch/AR-025/tournament_candidate_vs_root_same_deck_30.json` (SHA-256 `e1f456169857d0903661583baf6d59a899f427656eda72258671fb2bd1f6e4ac`)
- Tournament candidate-panel report: `experiments/autoresearch/AR-025/tournament_candidate_panel_10.json` (SHA-256 `e6d277a66fa3b1cfb794ffac90e2e862f61735956fe73e781d8b35e6fea2e5c3`)
- Tournament root-panel report: `experiments/autoresearch/AR-025/tournament_root_panel_10.json` (SHA-256 `816c701a72ee415cabb702f6791947b696b72bbd2b3a801f4e45b2b5a5dba740`)
- Decision: reject candidate for promotion; frozen Stage 4 root remains fallback.
