# AR-026 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T18:48:29.906613+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Result

The collector created multiple exact recurrent sibling bases per matchup.
Each base selected its own effective K from the legal action set, each matchup
was normalized independently, and all groups were combined in one FP32
policy-only optimizer step with terminal credit through discounted future
logical decisions. Branch sampling mixed policy mass with uniform legal-action
mass at `0.5`. The frozen root remains the fallback after the gate.

| Metric | Result |
| --- | ---: |
| Groups / fibers | 8 / 20 |
| Effective K per base | `[2, 4, 2, 2, 4, 2, 2, 2]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / `0.5` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 1434 / 1509 |
| Collection seconds / decisions/s | 13.648123 / 105.06939177131154 |
| One grouped optimizer step | 1 |
| Update seconds | 8.569577292073518 |
| Loss / gradient norm | 0.03191724792122841 / 0.9121775031089783 |
| Candidate parameter L2 delta | 0.00923267879010395 |

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
candidate won the direct root gate and improved the panel by three wins over
the frozen root, but absolute field strength remains low; retain it as the
current experimental direction and scale it before any promotion decision.

## Tournament gate

| Surface | Candidate | Frozen root | Decision |
| --- | ---: | ---: | --- |
| Same-deck vs root, n=30 | `21-9-0` (70.0%) | n/a | candidate wins direct gate |
| Six-opponent panel, n=60 | `15-45-0` (25.0%) | `12-48-0` (20.0%) | positive signal, scale next |

Candidate panel detail: `lb1009` `0-10`, `lb945` `1-9`, `lb826` `3-7`,
`lb814` `1-9`, `random` `6-4`, `first` `4-6`.

## Provenance

- Code commit at execution: `d27dda708c52c0c631dfed52727441f34dcbf80f`
- Candidate SHA-256: `af13a7ece3bca7c42760091b86478458b2e028c9130a6c9776010ede377347c2`
- Sample manifest SHA-256: `8cb8b3d992b5f637a159ad7273acdf1eb1ebfb044973e2464b9ed7f161655e9a`
- Trajectory bundle SHA-256: `d935410dfc9bb546619bb126e7ec6af47e9d9caa567922e2e82262c4bc0d3b64`
- Tournament candidate-vs-root report: `experiments/autoresearch/AR-026/tournament_candidate_vs_root_same_deck_30.json` (SHA-256 `bcc1041c9f1e978745c50b30ec5d6f678eb64a3075616fdb94749d135cb8463a`)
- Tournament candidate-panel report: `experiments/autoresearch/AR-026/tournament_candidate_panel_10.json` (SHA-256 `b4d6b1735df149182f1919266fe3cc345f6d02e35f9ac35eb95945887ca7fa7f`)
- Tournament root-panel report: `experiments/autoresearch/AR-026/tournament_root_panel_10.json` (SHA-256 `49024a62a7d53f64cc2ec05731d5e1b11de89a62f0a1d1d550579efba57c2bda`)
- Decision: keep experimental; scale the same branch-diversity hypothesis; frozen Stage 4 remains fallback.
