# AR-021 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T18:04:32.326997+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Result

The collector created multiple exact recurrent sibling bases per matchup.
Each base selected its own effective K from the legal action set, each matchup
was normalized independently, and all groups were combined in one FP32
policy-only optimizer step with terminal credit through discounted future
logical decisions. The frozen root remains the fallback pending tournament.

| Metric | Result |
| --- | ---: |
| Groups / fibers | 4 / 14 |
| Effective K per base | `[4, 4, 4, 2]` |
| Logical decisions / substeps | 1079 / 1176 |
| Collection seconds / decisions/s | 15.582338 / 69.24506432020944 |
| One grouped optimizer step | 1 |
| Update seconds | 8.520291083026677 |
| Loss / gradient norm | 0.017396682873368263 / 0.5238350033760071 |
| Candidate parameter L2 delta | 0.009213928003272408 |

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
| Candidate vs frozen Stage 4 root, same `agent/deck.csv`, n=30 | 22-8-0 | 73.3% | `tournament_candidate_vs_root_same_deck_30.json` |
| Candidate panel: lb826, random, first, n=10 each | 8-22-0 | 26.7% | `tournament_candidate_panel_10.json` |
| Frozen-root panel: lb826, random, first, n=10 each | 7-23-0 | 23.3% | `tournament_root_panel_10.json` |

AR-021 strongly beats the frozen root on the controlled same-deck gate, but
remains weak against the external panel: 2-8 against lb826, 4-6 against
random, and 2-8 against first. The root panel is 3-7, 4-6, and 0-10. This is
not yet supremacy against the field, so the root remains the operational
fallback and the next experiment should train with external-opponent deck
strata rather than promote this candidate.

## Limitations and next gate

This is a bounded grouped prospective update, not a strength estimate. The
collection remains serial and the recurrent learner boundary is detached. Use
the external-panel result to define the next bounded hypothesis; do not promote
AR-021 from the root-relative win alone.

## Provenance

- Code commit at execution: `a53614237a3ec057a0cfc86e945f9b93ffa71651`
- Candidate SHA-256: `52702295763ecee036e4f6bfaac6660df6ca5ec1cfca66efab5146ae8b292718`
- Sample manifest SHA-256: `cef52027903fa1cd97fe44d48cd6609e0e237e8d7aed7f80f7bcceb05e2de017`
- Trajectory bundle SHA-256: `2939d84c371de69b950e21d3504cfde8fad7c75e52edb822e139a9d67576f296`
- Candidate-vs-root report SHA-256: `116b113523affb671f31aa046bbbe934a8f47953b8a95865127d88af1768944c`
- Candidate-panel report SHA-256: `de6acc463afa441e788c7fb5b0dcf85bed1247918e4ace8b6448e29cba8a51ff`
- Frozen-root panel report SHA-256: `5e9c3fdfc4e1558b6349c0463980750d0c0cd9e5c36b891d5b42d0d76c92d508`
- Tournament gate: candidate keep experimental, root fallback
