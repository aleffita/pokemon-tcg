# State Capsule 065 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T12:15:13.847235+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `56` exact recurrent sibling groups
  and `121` fibers with effective K
  `[2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `297c7d06924d1b345bd82f03fbc6e99f98d15845b26da40030dcd2dc851b523e`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C065/report.md`
- `experiments/autoresearch/AR-038-C065/manifest.json`
- `experiments/autoresearch/AR-038-C065/metrics.json`
- `experiments/autoresearch/AR-038-C065/sample.manifest.json`
- `experiments/autoresearch/AR-038-C065/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C065/candidate.pt`

## Metrics

- Collection: `45.32382` s,
  `151.39941924049393` decisions/s.
- Update: `491.2695100829005` s; `3` optimizer steps.
- Credited logical actions: `6862`.
- Parameter L2 delta: `0.0006040745134193798`;
  gradient norm `1.18485689163208`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
