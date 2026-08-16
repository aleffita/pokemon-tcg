# State Capsule 008 - AR-006 reviewed

Captured: 2026-08-16 after the AR-006 reviewer gate.

## Decision

Stage 4 remains the frozen competitive root and only promoted policy. Packed
data remains opt-in infrastructure. AR-006 validates exact-root gating, valid
identity setup, and the extracted packed loader, but is **rework** because the
tests do not yet perform a real optimizer/scheduler micro-update or traverse
the production packed train/validation loops.

## Evidence

- Commit: `9de4516`.
- Focused locked suite: 47 passed in 0.64 s.
- Review: `experiments/autoresearch/AR-006/review.md`.
- Stage 4 root hash remains `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Open gates

AR-007 must add a bounded production-compatible optimizer/scheduler micro
update, cover fresh/reset behavior, exercise packed train prefetch and
validation paths with a tiny fixture, and test wrong-cwd relative-root
rejection. No packed self-play or RL run is allowed until this gate is
reviewed.

## Next control point

After AR-007 review, use the frozen Stage 4 root with the default Parquet
backend to inspect and probe the real self-play/RL trajectory and log-probability
boundary. Packed may become an infrastructure option only after the micro-loop
evidence is accepted.
