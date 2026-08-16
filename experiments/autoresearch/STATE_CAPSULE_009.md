# State Capsule 009 - AR-007 reviewed

Captured: 2026-08-16 after the AR-007 reviewer gate.

## Decision

Stage 4 remains the frozen competitive root and only promoted policy. AR-007
keeps the packed pipeline as opt-in infrastructure, but packed is not promoted
and is not the self-play data plane. The research loop now advances through the
default Parquet fallback while packed promotion remains a deferred rework
track.

## Evidence

- Commit: `ded9f50`.
- Focused locked suite: 51 passed in 0.64 s.
- Review: `experiments/autoresearch/AR-007/review.md`.
- Stage 4 root hash remains
  `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- The reviewer confirmed optimizer semantics and packed train/validation call
  sites, with remaining coverage gaps limited to promotion-grade edge cases.

## Open infrastructure work

Before packed can become default or feed on-policy collection, add a real
nonzero-phase optimizer continuation update and tests for empty plans,
auxiliary targets, `zero_wouldko`, validation memory continuity, and exact
fetched rows.

## Next control point

Inspect the existing self-play/RL implementation and establish the smallest
valid on-policy trajectory plus complete autoregressive log-probability probe
from the frozen Stage 4 root, using default Parquet and tournament only after a
candidate policy is produced.
