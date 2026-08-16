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
| Candidate vs frozen Stage 4 root, same `agent/deck.csv`, n=30 | 11-19-0 | 36.7% | `tournament_candidate_vs_root_same_deck_30.json` |
| Candidate artifact panel: lb826, random, first, n=10 each | 9-21-0 | 30.0% | `tournament_candidate_artifact_deck_panel_10.json` |
| Frozen-root artifact panel: lb826, random, first, n=10 each | 6-24-0 | 20.0% | `tournament_root_artifact_deck_panel_10.json` |

The controlled direct gate rejects promotion: AR-020 lost 11-19 to the frozen
root across 30 alternating-side games with the same `agent/deck.csv`. The
same-deck candidate panel was 2-8 against lb826, 3-7 against random, and 4-6
against first; the corresponding frozen-root panel was 2-8, 4-6, and 0-10.
An independent 10-game harness run with the same artifact deck was 6-4, but
the larger controlled run is the gate and supersedes that small result.
An earlier diagnostic route using the public `latest-submission-300elo` deck
reported 2-8 against root and 8-22 on its panel; it is retained as evidence
but is not the AR-020 artifact-deck gate. The root remains the operational
fallback.

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
- Controlled same-deck candidate-vs-root report SHA-256: `b01030a7d4002c2f9169e9e42be7bd2921f5b2c5dfaf89b84cc5d8e11cda0cf9`
- Small same-deck harness candidate-vs-root report SHA-256: `9346f79bffbbd47c216fc3fc5cfb55dfbf478980f0b74099967ff73c0b1df976`
- Controlled same-deck candidate-panel report SHA-256: `730ab71e630b7a0a6507462145f87a3f965cf11020ce8af784f369ce9080104a`
- Controlled same-deck frozen-root panel report SHA-256: `6c4431c444d6b9eb16af7cd6c9aa32442239405efc736b0e7f220835c62ef131`
- Diagnostic public-deck candidate-vs-root report SHA-256: `949531b753976663239033aa2e962ec128129c43fed2534a7944b5738a70692e`
- Candidate-vs-root gate: rejected for promotion, same-deck `11-19-0`
