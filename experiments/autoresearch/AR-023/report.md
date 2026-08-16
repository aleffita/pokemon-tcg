# AR-023 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T18:22:08.167214+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Result

The collector created multiple exact recurrent sibling bases per matchup.
Each base selected its own effective K from the legal action set, each matchup
was normalized independently, and all groups were combined in one FP32
policy-only optimizer step with terminal credit through discounted future
logical decisions. The frozen root remains the fallback after the tournament
gate.

| Metric | Result |
| --- | ---: |
| Groups / fibers | 4 / 12 |
| Effective K per base | `[2, 4, 4, 2]` |
| Logical decisions / substeps | 1005 / 1040 |
| Collection seconds / decisions/s | 9.187725 / 109.38507827590689 |
| One grouped optimizer step | 1 |
| Update seconds | 6.222183666890487 |
| Loss / gradient norm | -0.011875701136887074 / 0.5540380477905273 |
| Candidate parameter L2 delta | 0.009200024094590262 |

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
candidate won the controlled same-deck root gate but lost the external panel
relative to the frozen root, so it is rejected for promotion and retained as
experimental evidence.

## Tournament gate

| Surface | Candidate | Frozen root | Decision |
| --- | ---: | ---: | --- |
| Same-deck vs root, n=30 | `20-10-0` (66.7%) | n/a | candidate wins direct gate |
| External panel, n=30 | `7-23-0` (23.3%) | `8-22-0` (26.7%) | reject promotion |

External panel detail: candidate vs `lb826_alakazam_seok` `2-8`, `random`
`3-7`, `first` `2-8`; frozen root vs the same panel `1-9`, `3-7`, `4-6`.
The candidate is not the current best because its absolute field strength is
lower than the frozen root on this panel.

## Provenance

- Code commit at execution: `eed78b6fc41ced26877f4121e7385db28709cfe9`
- Candidate SHA-256: `6c064668e3201deb73bb32be415dc73204e9414b5c2b7c4b50ebdec65e579e4a`
- Sample manifest SHA-256: `9bcc89436c1a1f267ea903574f3b48ebe7298da3b21c0d52f6e460dcf6b53439`
- Trajectory bundle SHA-256: `7bfcc30a81509239fc9cbcce9f95c50b13d3dfb4c9bcb5f1358cd3c378348221`
- Tournament candidate-vs-root report: `experiments/autoresearch/AR-023/tournament_candidate_vs_root_same_deck_30.json` (SHA-256 `e9becd02602e5befaae5ad266bfcbeb6642db780bb2b345a99ee15ec24ad5188`)
- Tournament candidate-panel report: `experiments/autoresearch/AR-023/tournament_candidate_panel_10.json` (SHA-256 `871605969e0b3116b87f7b66c99a3867aa4b0520b4e80fd8fd07fc59bf5fb710`)
- Tournament root-panel report: `experiments/autoresearch/AR-023/tournament_root_panel_10.json` (SHA-256 `95d181e17939a28935d7605c63b1b6dd7859aaeb874b7e7eb3c3c367cd7e30af`)
- Decision: reject candidate for promotion; frozen Stage 4 root remains fallback.
