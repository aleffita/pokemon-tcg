# State Capsule 018 - true recurrent self-play gate

Captured 2026-08-16 after AR-018 implementation and one real smoke game.

## Current policy and root

- Stage 4 remains the only promoted/fallback policy.
- Frozen root checkpoint SHA-256:
  `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- No weights, root artifacts, package, submission, tournament, GRPO update,
  or RoPE-ND change was made by AR-018.

## AR-018 state

- Foundation code was present in commit `3867171`.
- Probe, tests, and executable wrapper were committed in `434d3f6`.
- Mode: current-vs-current true recurrent self-play.
- Metadata date: `2026-08-12`.
- Games: 1.
- Decisions: agent 85, mirror 73, total 158.
- Substep records: 168.
- Throughput: 52.532 records/s and 49.405 decisions/s.
- Terminal returns: agent `-1.0`, opponent `+1.0`.
- Both lane continuity checks: true.
- Parquet and packed hot paths: false.

The learner and mirror begin from the same learned-init digest but maintain
separate recurrent tensors. Each logical action keeps conditional substep
logprobs and receives their complete sum as `logical_action_logprob` and
`decision_logprob`. Legal actions, sides, memory input/output digests, and
committed last-substep digests are recorded in the compact self-play log.

## Evidence

- `experiments/autoresearch/AR-018/report.md`
- `experiments/autoresearch/AR-018/review.md`
- `experiments/autoresearch/AR-018/manifest.json`
- `experiments/autoresearch/AR-018/logs/selfplay.jsonl`
- `experiments/autoresearch/AR-018/logs/tests.log`

Focused validation: 30 tests passed; `py_compile` and `git diff --check`
passed. The next control point is trajectory-group GRPO, not another BC or
Parquet/packed-data task.
