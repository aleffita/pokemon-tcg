# State Capsule 014 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T02:54:35.293713+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `127` fibers with effective K
  `[4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 4, 2, 4, 2, 2, 2, 4, 2, 4, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 3, 2, 2, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `3cc35e1da644f42be95034b9957ecd8d96d3b859eab940b929c51d67c8037baa`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C014/report.md`
- `experiments/autoresearch/AR-038-C014/manifest.json`
- `experiments/autoresearch/AR-038-C014/metrics.json`
- `experiments/autoresearch/AR-038-C014/sample.manifest.json`
- `experiments/autoresearch/AR-038-C014/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C014/candidate.pt`

## Metrics

- Collection: `69.377905` s,
  `114.76276159019147` decisions/s.
- Update: `679.4898196668364` s; `3` optimizer steps.
- Credited logical actions: `7962`.
- Parameter L2 delta: `0.0006046282976080098`;
  gradient norm `2.5600907802581787`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
