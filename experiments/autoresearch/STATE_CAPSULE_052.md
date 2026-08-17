# State Capsule 052 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T09:54:30.731022+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `144` fibers with effective K
  `[4, 4, 4, 2, 2, 3, 3, 2, 3, 2, 2, 2, 2, 2, 3, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 3, 2, 4, 4, 4, 2, 4, 4, 3, 2, 2, 2, 4, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `f39d35d5d10a40704993e2c2bfb795aad71075a1f110ca490429a2c93af26428`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C052/report.md`
- `experiments/autoresearch/AR-038-C052/manifest.json`
- `experiments/autoresearch/AR-038-C052/metrics.json`
- `experiments/autoresearch/AR-038-C052/sample.manifest.json`
- `experiments/autoresearch/AR-038-C052/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C052/candidate.pt`

## Metrics

- Collection: `52.575463` s,
  `155.2435211203507` decisions/s.
- Update: `556.5458114170469` s; `3` optimizer steps.
- Credited logical actions: `8162`.
- Parameter L2 delta: `0.0006042923164468379`;
  gradient norm `1.3874380588531494`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
