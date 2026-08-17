# State Capsule 061 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T11:33:38.117499+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `56` exact recurrent sibling groups
  and `121` fibers with effective K
  `[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `9b50c7ea28fa6ba97069377800a231d801a2bd164e4dc1e3e96e132ab340982b`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C061/report.md`
- `experiments/autoresearch/AR-038-C061/manifest.json`
- `experiments/autoresearch/AR-038-C061/metrics.json`
- `experiments/autoresearch/AR-038-C061/sample.manifest.json`
- `experiments/autoresearch/AR-038-C061/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C061/candidate.pt`

## Metrics

- Collection: `48.050645` s,
  `147.40697144691936` decisions/s.
- Update: `501.03012670809403` s; `3` optimizer steps.
- Credited logical actions: `7083`.
- Parameter L2 delta: `0.0006042770730415621`;
  gradient norm `1.2836205959320068`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
