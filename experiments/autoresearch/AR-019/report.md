# AR-019 - trajectory-group GRPO micro-update

Captured on 2026-08-16T17:02:10.629259+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Result

The corrected AR-018 current-vs-current true recurrent collector produced a
single in-memory group of K=4 complete agent trajectories. One FP32,
policy-only trajectory-group GRPO update was applied to a copy of the root.
The frozen root remains the fallback. The candidate then ran a 10-game direct
gate against the frozen root and a 30-game opponent panel. No promotion was
made.

| Metric | Result |
| --- | ---: |
| Group returns | `[1.0, -1.0, 1.0, 1.0]` |
| Return mean / population std | `0.5` / `0.8660253882408142` |
| Logical decisions / substeps | 278 / 311 |
| Collection seconds / decisions/s | 6.267561 / 44.355374405573826 |
| Update seconds | 1.775413541821763 |
| Loss / gradient norm | 0.1619904488325119 / 2.5191781520843506 |
| Ratio mean / min / max | 1.0 / 0.9999923706054688 / 1.0000028610229492 |
| Candidate parameter L2 delta | 0.009198722439288157 |
| Candidate bytes | 5694416 |

## Tournament gate

| Surface | W-L-D | Win rate |
| --- | ---: | ---: |
| Candidate vs frozen Stage 4 root, n=10 | 6-4-0 | 60.0% |
| Candidate panel: lb826, random, first, n=30 | 9-21-0 | 30.0% |
| Frozen root panel: lb826, random, first, n=30 | 9-21-0 | 30.0% |

The direct candidate-vs-root result is positive, while the small panel ties
the frozen root. The candidate remains experimental and the root remains the
fallback; this is not evidence for unconditional promotion.

## Contracts checked

- Behavior data came from the frozen root and true recurrent current-vs-current
  self-play. The mirror retained its independent Stage 4 memory lane.
- Every retained agent sample has a detached recurrent input, a real legal
  mask, an action, and a finite behavior substep logprob.
- Each logical-action behavior logprob is the sum of its conditional substep
  logprobs. Learner ratios use that logical sum once per decision.
- Group credit is assigned once per logical decision and shared across all
  substeps of that decision. No separate substep-relative credit is used.
- Zero-variance groups normalize to zero advantages and perform zero optimizer
  steps. This run had `zero_variance_group=False`.
- Candidate checkpoint is strict FP32 inference format and is linked to the
  adjacent compact bounded provenance bundle. No unbounded rollout buffer was
  written.
- Existing candidate provenance preflight passed for the root, manifest, and
  trajectory bundle: `True`.

## Limitations

This is a K=4 micro-update, not a strength estimate. Collection is serial,
the recurrent learner boundary is detached, and value loss is intentionally
zero. Tournament samples are small and the panel did not improve aggregate
win rate over the frozen root. The candidate is experimental only; Stage 4
root remains the fallback.

## Provenance

- Code commit at execution: `5dd6cfc935379c0ded7a2982110e0c3ccf805121`
- Root SHA-256: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`
- Candidate SHA-256: `3e23d7c3c191fa301baabc559dd9add82d6ffbec05ae8b8e6fd3327cdc17d183`
- Candidate path: `experiments/autoresearch/AR-019/candidate.pt`
- Sample manifest: `experiments/autoresearch/AR-019/sample.manifest.json`
- Sample manifest file SHA-256: `c757a496457983bacf851bb31815f399d2e64829cc57a7613c304789e971a30a`
- Sample manifest content SHA-256: `f4b02aec85d4c1bad46194316920646d6f264f90ff205d6cad5d61232d600e08`
- Trajectory bundle: `experiments/autoresearch/AR-019/trajectory_bundle.pt.gz`
- Trajectory bundle SHA-256: `e86ede4ecd2dac75585452f08d52d43eaf64f6bd90848786618657566c3a18f3`
- Candidate preflight: `{'passed': True, 'approved_root_sha256': 'b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b', 'artifacts': {'sample_manifest': 'experiments/autoresearch/AR-019/sample.manifest.json', 'trajectory_bundle': 'experiments/autoresearch/AR-019/trajectory_bundle.pt.gz'}}`
- Candidate-vs-root tournament: `experiments/autoresearch/AR-019/tournament_candidate_vs_root_10.json`
- Candidate panel tournament: `experiments/autoresearch/AR-019/tournament_candidate_panel_10.json`
- Frozen root panel tournament: `experiments/autoresearch/AR-019/tournament_root_panel_10.json`
- Rollout persistence: `compact bounded provenance bundle persisted; no unbounded rollout buffer`
