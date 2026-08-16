# AR-019 - trajectory-group GRPO micro-update

Captured on 2026-08-16T16:42:34.464748+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Result

The corrected AR-018 current-vs-current true recurrent collector produced a
single in-memory group of K=4 complete agent trajectories. One FP32,
policy-only trajectory-group GRPO update was applied to a copy of the root.
The frozen root remains the fallback. No tournament, package, submission, or
promotion was run.

| Metric | Result |
| --- | ---: |
| Group returns | `[1.0, 1.0, -1.0, -1.0]` |
| Return mean / population std | `0.0` / `1.0` |
| Logical decisions / substeps | 340 / 380 |
| Collection seconds / decisions/s | 7.347098 / 46.27677354690585 |
| Update seconds | 2.152347584022209 |
| Loss / gradient norm | 0.08823533356189728 / 2.373613119125366 |
| Ratio mean / min / max | 0.9999998807907104 / 0.9999895095825195 / 1.0000028610229492 |
| Candidate parameter L2 delta | 0.009198900828294767 |
| Candidate bytes | 5693904 |

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
- Candidate checkpoint is strict FP32 inference format and is independent of
  any persisted rollout bundle. No large rollout artifact was written.

## Limitations

This is a K=4 micro-update, not a strength estimate. Collection is serial,
the recurrent learner boundary is detached, value loss is intentionally zero,
and no tournament has been run. The candidate is experimental only; Stage 4
root remains the fallback. The group return variance is small-sample and the
candidate has not been promoted.

## Provenance

- Code commit at execution: `8b0c16677cb4affadf6137b7510a14a86ce19991`
- Root SHA-256: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`
- Candidate SHA-256: `2e592340522c697405676811f4d68c53c93129fd89b8f14b07427d4624ed77e7`
- Candidate path: `experiments/autoresearch/AR-019/candidate.pt`
- Rollout persistence: `none; compact tensors retained in memory only`
