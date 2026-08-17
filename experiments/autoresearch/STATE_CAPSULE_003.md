# State Capsule 003 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T00:33:41.563769+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `133` fibers with effective K
  `[4, 3, 2, 2, 4, 3, 2, 2, 4, 2, 2, 2, 2, 3, 2, 2, 3, 4, 2, 2, 2, 3, 2, 2, 3, 4, 2, 2, 2, 4, 2, 2, 4, 3, 2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `3905136672c67646bd3534f03ff60b017ba5ec96a064d389de510f2e8ed0f61c`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C003/report.md`
- `experiments/autoresearch/AR-038-C003/manifest.json`
- `experiments/autoresearch/AR-038-C003/metrics.json`
- `experiments/autoresearch/AR-038-C003/sample.manifest.json`
- `experiments/autoresearch/AR-038-C003/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C003/candidate.pt`

## Metrics

- Collection: `53.86885` s,
  `144.40627595533542` decisions/s.
- Update: `549.7594751249999` s; `3` optimizer steps.
- Credited logical actions: `7779`.
- Parameter L2 delta: `0.0006045556713381601`;
  gradient norm `2.654484748840332`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
