# AR-022 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T18:11:52.163720+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Result

The collector created multiple exact recurrent sibling bases per matchup.
Each base selected its own effective K from the legal action set, each matchup
was normalized independently, and all groups were combined in one FP32
policy-only optimizer step with terminal credit through discounted future
logical decisions. The frozen root remains the fallback pending tournament.

| Metric | Result |
| --- | ---: |
| Groups / fibers | 4 / 8 |
| Effective K per base | `[2, 2, 2, 2]` |
| Logical decisions / substeps | 685 / 728 |
| Collection seconds / decisions/s | 9.909958 / 69.12239241669995 |
| One grouped optimizer step | 1 |
| Update seconds | 3.4491552088875324 |
| Loss / gradient norm | -0.0068812002427875996 / 0.8533870577812195 |
| Candidate parameter L2 delta | 0.009192485631226077 |

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

## Tournament gate

| Surface | W-L-D | Win rate | Report |
| --- | ---: | ---: | --- |
| Candidate vs frozen Stage 4 root, same `agent/deck.csv`, n=30 | 13-17-0 | 43.3% | `tournament_candidate_vs_root_same_deck_30.json` |
| Candidate panel: lb826, random, first, n=10 each | 7-23-0 | 23.3% | `tournament_candidate_panel_10.json` |
| Frozen-root panel: lb826, random, first, n=10 each | 8-22-0 | 26.7% | `tournament_root_panel_10.json` |

The external-opponent deck-stratified update did not improve either gate: it
lost 13-17 to the frozen root and trailed the root panel 23.3% to 26.7%.
Three of four training groups had zero variance and therefore contributed no
gradient, leaving only 220 credited logical actions. Reject AR-022 and keep
the frozen Stage 4 root fallback.

## Limitations and next gate

This is a bounded grouped prospective update, not a strength estimate. The
collection remains serial and the recurrent learner boundary is detached. The
external deck strata need a better opponent-policy or richer branch sampler
before another scale-up; no promotion is justified by this run.

## Provenance

- Code commit at execution: `50c1b6d67f52123a7332cd97c402c7c4b892222c`
- Candidate SHA-256: `0fb2fed2282298cb2e1e2f9cf14ca28b101735c5e839f303abba6f9d49da0c1a`
- Sample manifest SHA-256: `5f0a6935e5ba3712b6601835efe4c7cb0f3ee161cd0a20142a7e5a62c790464e`
- Trajectory bundle SHA-256: `066785e77850fbd6f8e61f7d942b4db3da02cccab17cac1133e413d26824c201`
- Candidate-vs-root report SHA-256: `59701f089c9288ee80ca46cac84a876c90282857f491e0e5f7004645c22fc70b`
- Candidate-panel report SHA-256: `9dab68518ef151df63b1876eb376595edc9881a69e5e86853af33a26261400cb`
- Frozen-root panel report SHA-256: `b8e0100da4c0377d30c5e181c22f347ec3a102227b5560483c0e19f5f871df8f`
- Tournament gate: rejected for promotion, same-deck `13-17-0`
