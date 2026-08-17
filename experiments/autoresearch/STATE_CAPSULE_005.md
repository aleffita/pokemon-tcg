# State Capsule 005 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T00:56:48.249615+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `120` fibers with effective K
  `[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `524b09ebb19879a997e579ba82f7a551babd151e62345f47c8720cc62a055608`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C005/report.md`
- `experiments/autoresearch/AR-038-C005/manifest.json`
- `experiments/autoresearch/AR-038-C005/metrics.json`
- `experiments/autoresearch/AR-038-C005/sample.manifest.json`
- `experiments/autoresearch/AR-038-C005/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C005/candidate.pt`

## Metrics

- Collection: `61.609956` s,
  `114.42955706523021` decisions/s.
- Update: `591.805879249936` s; `3` optimizer steps.
- Credited logical actions: `7050`.
- Parameter L2 delta: `0.0006048711457376437`;
  gradient norm `2.7690248489379883`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
