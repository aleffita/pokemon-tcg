# State Capsule 018 - corrected true recurrent self-play gate

Captured 2026-08-16 after the AR-018 reset-retry repair and final four-game
smoke.

## Current policy and root

- Stage 4 remains the only promoted/fallback policy.
- Frozen root checkpoint SHA-256:
  `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- No weights, root artifacts, package, submission, tournament, GRPO update,
  or RoPE-ND change was made by AR-018.

## AR-018 state

- Foundation commit: `3867171`.
- Initial probe commit: `434d3f6`.
- Reset-retry repair commit: `28c2b96`.
- Mode: current-vs-current true recurrent self-play.
- Metadata date: `2026-08-12`.
- Smoke: 4 games, seed `18000`, both agent sides covered `[1, 0, 0, 1]`.
- Decisions: agent 353, mirror 319, total 672.
- Substep records: 759.
- Throughput: 118.369 records/s and 104.801 decisions/s.
- Agent terminal returns: `[-1.0, +1.0, -1.0, +1.0]`.
- Mirror terminal returns: `[+1.0, -1.0, +1.0, -1.0]`.
- Both lane continuity checks: true in every game.
- Parquet and packed hot paths: false.

The reset hook executes before every battle-start retry, preventing a
discarded opening from leaking mirror memory or events into the accepted
episode. Mirror terminal events carry the opponent-perspective reward. A
copied learner snapshot recomputes complete logical-action logprobs and
matches the behavior snapshot at ratio one.

## Evidence

- `experiments/autoresearch/AR-018/report.md`
- `experiments/autoresearch/AR-018/review.md`
- `experiments/autoresearch/AR-018/manifest.json`
- `experiments/autoresearch/AR-018/logs/selfplay.jsonl`
- `experiments/autoresearch/AR-018/logs/tests.log`

Focused validation: 32 tests passed; `py_compile` and `git diff --check`
passed. The corrected AR-018 gate is kept as the foundation for
trajectory-group GRPO. The next control point is not another BC or
Parquet/packed-data task.
