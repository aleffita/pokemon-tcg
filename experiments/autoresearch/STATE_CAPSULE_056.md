# State Capsule 056 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T10:38:10.134845+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `56` exact recurrent sibling groups
  and `141` fibers with effective K
  `[2, 2, 4, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 4, 3, 2, 2, 4, 2, 3, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 4, 2, 2, 2, 4, 3, 4, 2, 4, 2, 4, 2, 4, 4, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `b19cd3830c1fbc37962148c4d961df4daa7b9d07d1cadbefac2b4005e4e17ad5`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C056/report.md`
- `experiments/autoresearch/AR-038-C056/manifest.json`
- `experiments/autoresearch/AR-038-C056/metrics.json`
- `experiments/autoresearch/AR-038-C056/sample.manifest.json`
- `experiments/autoresearch/AR-038-C056/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C056/candidate.pt`

## Metrics

- Collection: `51.074161` s,
  `157.2419364966774` decisions/s.
- Update: `541.6239716671407` s; `3` optimizer steps.
- Credited logical actions: `8031`.
- Parameter L2 delta: `0.0006041687299345953`;
  gradient norm `1.2891379594802856`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
