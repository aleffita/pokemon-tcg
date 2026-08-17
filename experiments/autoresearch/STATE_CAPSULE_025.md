# State Capsule 025 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T04:54:58.244937+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `133` fibers with effective K
  `[2, 2, 4, 2, 3, 2, 2, 2, 2, 2, 4, 2, 3, 2, 3, 2, 3, 2, 4, 2, 4, 2, 4, 2, 3, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 4, 2, 4, 2, 2, 2, 2, 2, 4, 2, 4, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `3c175b8e0caff4bb789e7869aa3a33ac301a3797db74c36880f95617d4cf3b22`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C025/report.md`
- `experiments/autoresearch/AR-038-C025/manifest.json`
- `experiments/autoresearch/AR-038-C025/metrics.json`
- `experiments/autoresearch/AR-038-C025/sample.manifest.json`
- `experiments/autoresearch/AR-038-C025/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C025/candidate.pt`

## Metrics

- Collection: `52.526018` s,
  `149.12609677832765` decisions/s.
- Update: `594.5954044579994` s; `3` optimizer steps.
- Credited logical actions: `7833`.
- Parameter L2 delta: `0.0006043574417210513`;
  gradient norm `2.3878841400146484`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
