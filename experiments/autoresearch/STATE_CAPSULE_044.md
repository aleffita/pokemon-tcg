# State Capsule 044 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T08:23:33.263724+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `126` fibers with effective K
  `[2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 3, 3, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 4, 2, 2, 4, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `69812d09b1c87944d77bc50b439e25b27abc601eaafe956b1c2fab75d77edbd6`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C044/report.md`
- `experiments/autoresearch/AR-038-C044/manifest.json`
- `experiments/autoresearch/AR-038-C044/metrics.json`
- `experiments/autoresearch/AR-038-C044/sample.manifest.json`
- `experiments/autoresearch/AR-038-C044/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C044/candidate.pt`

## Metrics

- Collection: `45.228474` s,
  `156.80387549718` decisions/s.
- Update: `492.16416004206985` s; `3` optimizer steps.
- Credited logical actions: `7092`.
- Parameter L2 delta: `0.0006042723670950864`;
  gradient norm `1.628007411956787`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
