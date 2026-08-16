# AR-027-retry - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T19:04:12.424837+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Result

The collector created multiple exact recurrent sibling bases per matchup.
Each base selected its own effective K from the legal action set, each matchup
was normalized independently, and all groups were combined in one FP32
policy-only optimizer step with terminal credit through discounted future
logical decisions. The frozen root remains the fallback pending tournament.

| Metric | Result |
| --- | ---: |
| Groups / fibers | 16 / 41 |
| Effective K per base | `[2, 2, 2, 2, 3, 4, 4, 2, 4, 2, 4, 2, 2, 2, 2, 2]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 3031 / 3185 |
| Collection seconds / decisions/s | 30.201511 / 100.35921560688664 |
| One grouped optimizer step | 1 |
| Update seconds | 16.868608332937583 |
| Loss / gradient norm | -0.018700913622063504 / 0.4703052341938019 |
| Candidate parameter L2 delta | 0.009225403082207767 |

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
collection remains serial and the recurrent learner boundary is detached. Run
the controlled same-deck candidate-vs-root gate and the multi-opponent panel
before interpreting or promoting the candidate.

## Provenance

- Code commit at execution: `509b948bee7fb1b43503a900257268d9bd14f848`
- Candidate SHA-256: `6db9eb3496ca00f9a70cbb1f8e24027cacb456219b74e355e18c95783981d24c`
- Sample manifest SHA-256: `5f28b4e730222d4807421737ca032e36d7c5bf7e204f009cfc05f7fa6210cf19`
- Trajectory bundle SHA-256: `fc9d77cb3b09562b5248031d6c2372f476e9deb279df6439eebe88f9e824bad3`
- Tournament gate: rejected for promotion. Candidate vs frozen root was
  `13-17-0` in 30; candidate panel was `9-51-0` in 60; frozen-root panel was
  `12-48-0` in the same 60-game surface. Keep the frozen Stage 4 root.

## Tournament evidence

| Opponent | Candidate W-L-D | Root W-L-D |
| --- | ---: | ---: |
| random | 2-8-0 | 5-5-0 |
| first | 4-6-0 | 3-7-0 |
| lb1009_mega_lucario_ex_islet | 0-10-0 | 0-10-0 |
| lb945_multiply_ivan | 0-10-0 | 1-9-0 |
| lb826_alakazam_seok | 1-9-0 | 1-9-0 |
| lb814_crustle_emre | 2-8-0 | 2-8-0 |

The candidate's external panel win rate was 15.0%, below the frozen root's
20.0%. The direct same-deck gate also failed at 43.3%. This candidate is
retained as diagnostic evidence only.
