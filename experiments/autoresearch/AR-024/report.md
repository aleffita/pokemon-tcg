# AR-024 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T18:28:32.444817+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Result

The collector created multiple exact recurrent sibling bases per matchup.
Each base selected its own effective K from the legal action set, each matchup
was normalized independently, and all groups were combined in one FP32
policy-only optimizer step with terminal credit through discounted future
logical decisions. The frozen root remains the fallback after the tournament
gate.

| Metric | Result |
| --- | ---: |
| Groups / fibers | 4 / 8 |
| Effective K per base | `[2, 2, 2, 2]` |
| Logical decisions / substeps | 591 / 611 |
| Collection seconds / decisions/s | 5.295967 / 111.59434831059819 |
| One grouped optimizer step | 1 |
| Update seconds | 2.664541875012219 |
| Loss / gradient norm | 0.02643195353448391 / 0.7295721769332886 |
| Candidate parameter L2 delta | 0.009215549014408107 |

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
candidate won the controlled same-deck root gate and improved the matching
panel by one win, but absolute field strength remains low, so it is rejected
for promotion and retained as experimental evidence.

## Tournament gate

| Surface | Candidate | Frozen root | Decision |
| --- | ---: | ---: | --- |
| Same-deck vs root, n=30 | `19-11-0` (63.3%) | n/a | candidate wins direct gate |
| Six-opponent panel, n=60 | `8-52-0` (13.3%) | `7-53-0` (11.7%) | marginal gain, reject promotion |

Candidate panel detail: `lb1009` `0-10`, `lb945` `0-10`, `lb826` `3-7`,
`lb814` `2-8`, `random` `3-7`, `first` `0-10`. The one-win aggregate
improvement is not a supremacy signal and two training strata had zero
variance.

## Provenance

- Code commit at execution: `0e16b76455e48c82762a9ed981c2e3b29d956f01`
- Candidate SHA-256: `abade9c813286b0480e7fb265cfa659412492dd95bb1aafa21337a839816dcd3`
- Sample manifest SHA-256: `bee9061a85fea25a27ba917af8a0bac7af34947908bee36dab9149d40356ff3c`
- Trajectory bundle SHA-256: `51e836c7a95ddef254cbb127d61a9cab61cfa1501f5eed02aa82c12a8c2113cf`
- Tournament candidate-vs-root report: `experiments/autoresearch/AR-024/tournament_candidate_vs_root_same_deck_30.json` (SHA-256 `d6d802bbbee1d282efb5a78c8d2871171346b92f3b7254e23ab9e4aca10f6a41`)
- Tournament candidate-panel report: `experiments/autoresearch/AR-024/tournament_candidate_panel_10.json` (SHA-256 `9605b84bcdce169335d73bb8288da4524bf2f3ebe726854fa4d8329e3b3caaa0`)
- Tournament root-panel report: `experiments/autoresearch/AR-024/tournament_root_panel_10.json` (SHA-256 `2f20385eee72653aa6bfae6fa5cdd4656c35e57dd9fef3a7d3a362bcba6d321a`)
- Decision: reject candidate for promotion; frozen Stage 4 root remains fallback.
