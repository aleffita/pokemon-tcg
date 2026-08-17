# State Capsule 007 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T01:21:04.268170+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `127` fibers with effective K
  `[2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 3, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 3, 2, 2, 2, 2, 2, 4, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `dc972b61cfd355e0312363e2364cd0990d5641600770b1e836ea9b3c698da197`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C007/report.md`
- `experiments/autoresearch/AR-038-C007/manifest.json`
- `experiments/autoresearch/AR-038-C007/metrics.json`
- `experiments/autoresearch/AR-038-C007/sample.manifest.json`
- `experiments/autoresearch/AR-038-C007/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C007/candidate.pt`

## Metrics

- Collection: `63.51755` s,
  `115.1492773324805` decisions/s.
- Update: `600.3802286658902` s; `3` optimizer steps.
- Credited logical actions: `7314`.
- Parameter L2 delta: `0.0006044452777167687`;
  gradient norm `2.7249300479888916`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
