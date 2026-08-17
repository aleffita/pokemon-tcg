# State Capsule 004 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T00:44:39.289034+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `118` fibers with effective K
  `[3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 3, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `7f6a99d4b3a861f9b05c0cc971ca5f23c05d21fd83c9f22d9b9c4b119b8328af`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C004/report.md`
- `experiments/autoresearch/AR-038-C004/manifest.json`
- `experiments/autoresearch/AR-038-C004/metrics.json`
- `experiments/autoresearch/AR-038-C004/sample.manifest.json`
- `experiments/autoresearch/AR-038-C004/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C004/candidate.pt`

## Metrics

- Collection: `49.150486` s,
  `142.13491185185492` decisions/s.
- Update: `531.759851250099` s; `3` optimizer steps.
- Credited logical actions: `6986`.
- Parameter L2 delta: `0.0006046513530247692`;
  gradient norm `2.7837629318237305`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
