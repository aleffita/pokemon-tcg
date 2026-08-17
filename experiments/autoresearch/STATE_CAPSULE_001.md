# State Capsule 001 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T00:10:51.060148+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `160` fibers with effective K
  `[4, 2, 4, 4, 2, 2, 3, 2, 4, 3, 4, 3, 3, 2, 3, 4, 2, 4, 4, 4, 4, 4, 3, 2, 4, 2, 4, 4, 2, 2, 4, 3, 2, 4, 2, 2, 4, 4, 2, 4, 4, 4, 2, 4, 4, 4, 3, 2, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `a19cbc81b3d97611fc3c7a02297ba4f30bfe7bc058c4af74e187effd9c5fcafb`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C001/report.md`
- `experiments/autoresearch/AR-038-C001/manifest.json`
- `experiments/autoresearch/AR-038-C001/metrics.json`
- `experiments/autoresearch/AR-038-C001/sample.manifest.json`
- `experiments/autoresearch/AR-038-C001/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C001/candidate.pt`

## Metrics

- Collection: `54.015493` s,
  `180.41120081943302` decisions/s.
- Update: `617.3764402091037` s; `3` optimizer steps.
- Credited logical actions: `9745`.
- Parameter L2 delta: `0.0006047407859426242`;
  gradient norm `2.5308525562286377`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
