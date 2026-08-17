# State Capsule 036 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T06:58:54.860173+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `118` fibers with effective K
  `[2, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `c5c1fc0a4acb445a1ebedbb3bc4056a1c2aa9a9f944071940cd3d08bec5ddde6`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C036/report.md`
- `experiments/autoresearch/AR-038-C036/manifest.json`
- `experiments/autoresearch/AR-038-C036/metrics.json`
- `experiments/autoresearch/AR-038-C036/sample.manifest.json`
- `experiments/autoresearch/AR-038-C036/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C036/candidate.pt`

## Metrics

- Collection: `45.574174` s,
  `153.72741742592663` decisions/s.
- Update: `499.33205462503247` s; `3` optimizer steps.
- Credited logical actions: `7006`.
- Parameter L2 delta: `0.0006043931234888822`;
  gradient norm `1.9915179014205933`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
