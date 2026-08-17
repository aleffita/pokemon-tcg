# State Capsule 002 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T00:22:15.099259+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `142` fibers with effective K
  `[4, 3, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 4, 4, 3, 2, 2, 4, 4, 2, 4, 2, 4, 2, 2, 2, 4, 2, 4, 4, 2, 2, 2, 2, 2, 2, 3, 3, 4, 2, 4, 4, 4, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `3d5dfef73cc6d2c861d96b72e922a7fe48d8baea2490c7c2d1fa077a5a2125d9`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C002/report.md`
- `experiments/autoresearch/AR-038-C002/manifest.json`
- `experiments/autoresearch/AR-038-C002/metrics.json`
- `experiments/autoresearch/AR-038-C002/sample.manifest.json`
- `experiments/autoresearch/AR-038-C002/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C002/candidate.pt`

## Metrics

- Collection: `48.560617` s,
  `163.3010548269664` decisions/s.
- Update: `544.7986504591536` s; `3` optimizer steps.
- Credited logical actions: `7930`.
- Parameter L2 delta: `0.0006046269219020212`;
  gradient norm `2.774322748184204`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
