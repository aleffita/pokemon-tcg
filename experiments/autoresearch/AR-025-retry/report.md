# AR-025-retry - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T18:36:04.454499+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Result

The collector created multiple exact recurrent sibling bases per matchup.
Each base selected its own effective K from the legal action set, each matchup
was normalized independently, and all groups were combined in one FP32
policy-only optimizer step with terminal credit through discounted future
logical decisions. The frozen root remains the fallback after the tournament
gate.

| Metric | Result |
| --- | ---: |
| Groups / fibers | 8 / 17 |
| Effective K per base | `[2, 2, 2, 2, 2, 2, 2, 3]` |
| Logical decisions / substeps | 1200 / 1245 |
| Collection seconds / decisions/s | 12.10168 / 99.15978642394391 |
| One grouped optimizer step | 1 |
| Update seconds | 6.603137208148837 |
| Loss / gradient norm | -0.040290653705596924 / 0.6505328416824341 |
| Candidate parameter L2 delta | 0.00922129716001427 |

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

This is a bounded retry of the AR-025 scale hypothesis, not a strength
estimate. The candidate won the direct root gate but remained weak on the
multi-opponent panel, so it is rejected for promotion and retained as a
retry/scale diagnostic.

## Tournament gate

| Surface | Candidate | Frozen root | Decision |
| --- | ---: | ---: | --- |
| Same-deck vs root, n=30 | `16-14-0` (53.3%) | n/a | candidate wins direct gate |
| Six-opponent panel, n=60 | `7-53-0` (11.7%) | `12-48-0` (20.0%) | reject promotion |

The lower-scale retry did not recover field strength; no promotion is
allowed from the direct gate alone.

## Provenance

- Code commit at execution: `c051b8cd234a7791c366cf58e55e107d90a51745`
- Candidate SHA-256: `2bd20e999284877a75ca7cdfe3f6be7a53af1269deaff7ce81c8d75e7111700b`
- Sample manifest SHA-256: `7190b2ea5019302c4bb3d5f5e19b07d45ee48ee3502bdbfc6db1ae06a9524e5f`
- Trajectory bundle SHA-256: `35be19a2f59dba4c78240b6035c6e92b42d25642ce035de133ca447cc8c139d3`
- Tournament candidate-vs-root report: `experiments/autoresearch/AR-025-retry/tournament_candidate_vs_root_same_deck_30.json` (SHA-256 `840818709bc5193e89b0ad1da4a8b59baeb611076864cd02f7fad867863e05d0`)
- Tournament candidate-panel report: `experiments/autoresearch/AR-025-retry/tournament_candidate_panel_10.json` (SHA-256 `c7d099c0d9823ffd3b82c65489c1c45c2ac2b75b498bc6570d8e4c00ff13a930`)
- Frozen-root panel reference: `experiments/autoresearch/AR-025/tournament_root_panel_10.json` (SHA-256 `816c701a72ee415cabb702f6791947b696b72bbd2b3a801f4e45b2b5a5dba740`)
- Decision: reject retry candidate for promotion; frozen Stage 4 root remains fallback.
