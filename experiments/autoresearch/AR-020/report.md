# AR-020 - sibling-fiber GRPO micro-update

Captured on 2026-08-16T17:47:00.052564+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Result

    The collector created the maximum available distinct legal sibling fibers
    (requested K=4) from one exact in-process recurrent base, with common
    post-branch randomness and independent recurrent lanes. One FP32
    policy-only update applied terminal group-relative credit to the branch and
    discounted future logical decisions. The frozen root remains the fallback.
    The candidate was evaluated on the candidate-vs-root and multi-opponent
    tournament surfaces and was not promoted.

| Metric | Result |
| --- | ---: |
| Effective K per decision base / branch actions | `[4, 3]` / `[[2, 0, 1, 3], [0, 2, 1]]` |
| Group returns | `[[-1.0, -1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]]` |
| Return mean / population std | `[-0.5, -0.3333333432674408]` / `[0.8660253882408142, 0.9428090453147888]` |
| Logical decisions / substeps | 569 / 658 |
| Collection seconds / decisions/s | 8.839274 / 64.3718009671044 |
| Update seconds | 4.025555834174156 |
| Loss / gradient norm | 0.006037665065377951 / 0.8922091722488403 |
| Branch ratio mean / min / max | 1.000110149383545 / 0.8930689692497253 / 1.0740165710449219 |
| Candidate parameter L2 delta | 0.013461024582650779 |
| Candidate bytes | 5695568 |

## Contracts checked

- All fibers share the same first-state action-mask, model-input, recurrent
  memory, agent side, and random seed, while their first actions are distinct
  legal actions from the frozen root distribution.
- Behavior data uses true recurrent current-vs-current self-play with an
  independent mirror lane. Complete logical-action behavior logprobs remain
  recorded for every continuation.
- Sibling credit targets the branch logical action once per fiber and, in this
  run, propagates discounted terminal credit through future logical decisions.
  It is never duplicated across conditional substeps.
- Group returns are terminal values in `{-1, 0, +1}`; zero variance fails
  closed to an optimizer no-op. This run had
  `zero_variance_group=False`.
- Candidate is strict FP32 and linked to the adjacent bounded provenance
  bundle. Candidate preflight passed: `True`.

## Tournament gate

| Surface | W-L-D | Win rate | Report |
| --- | ---: | ---: | --- |
| Candidate vs frozen Stage 4 root, n=10 | 2-8-0 | 20.0% | `tournament_candidate_vs_root_10.json` |
| Candidate panel: lb826, random, first, n=10 each | 8-22-0 | 26.7% | `tournament_candidate_panel_10.json` |
| Frozen-root panel: lb826, random, first, n=10 each | 3-27-0 | 10.0% | `tournament_root_panel_10.json` |

The direct gate rejects promotion: AR-020 lost 2-8 to the frozen root. The
candidate panel was 8-22, with 1-9 against lb826, 6-4 against random, and 1-9
against first. The root panel was 0-10, 2-8, and 1-9 respectively. These
panel runs used each packaged agent's default deck, and the candidate and
frozen-root packages have different deck contents, so the panel delta is
directional evidence rather than a controlled same-deck estimate. The root
remains the operational fallback.

## Limitations and next gate

This is a bounded dynamic-K prospective micro-update, not a strength estimate.
Collection is serial and the recurrent learner boundary is detached. Continue
with the next explicitly bounded hypothesis only after recording this negative
promotion gate; keep the frozen Stage 4 root available as the fallback.

## Provenance

- Code commit at execution: `a171dc88524c80c6e32b766a41297e1ed55c00c9`
- Candidate SHA-256: `89a70d4eddb3c856d7c4a4e1ad520e2d23bc7230c76b4c10904c45970eeb8637`
- Sample manifest SHA-256: `d1996dae8ee82ac253a5834516f37da344192b5f146d1ac9c320eda9d2bf9616`
- Trajectory bundle SHA-256: `2c7d868791c38879780f92d26d71aa5eddf5e58f06263f002cdf61d8e2d45363`
- Candidate-vs-root report SHA-256: `949531b753976663239033aa2e962ec128129c43fed2534a7944b5738a70692e`
- Candidate-panel report SHA-256: `3aec5157f60143b035d69422b69844989a5d5bb60ce02a803426768e82f59948`
- Frozen-root panel report SHA-256: `ccaf3c505666472bdb633d7aadb3c7cb50df705bcfca86337455f425c43b6c96`
- Candidate-vs-root gate: rejected for promotion, `2-8-0`
