# State Capsule 010 - AR-008 reviewed

Captured: 2026-08-16 after the AR-008 reviewer gate.

## Decision

Stage 4 remains the frozen competitive root and only promoted policy. AR-008
establishes a real FP32 trajectory smoke contract but is **not update-ready**:
the committed rows cannot recompute current-policy logprobs, and the root hash
was recorded after load rather than enforced before use.

## Evidence

- Commit: `1126f72`.
- Probe: 233 rows, random 114, mirror_no_memory 119, one terminal reward per
  episode, 64.45 rows/s.
- Focused tests: 6 passed.
- Review: `experiments/autoresearch/AR-008/review.md`.
- Stage 4 root SHA observed:
  `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Open gates

AR-009 must enforce the root hash before loading, reject both date-field
conflicts, and make the trajectory update-ready by retaining observations and
real masks in-process or in a provenance-linked artifact. Only after those
checks may a small PPO/group-relative alternative update be attempted.

## Next control point

Produce one small candidate from current-policy sampling, package its exact
checkpoint/deck/provenance, then use a named smoke tournament against random,
first, and one fixed public opponent before deciding keep/revert.
