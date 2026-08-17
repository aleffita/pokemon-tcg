# State Capsule ted - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-16T19:54:59.714265+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `8` exact recurrent sibling groups
  and `21` fibers with effective K
  `[2, 2, 4, 3, 2, 2, 4, 2]`.
- The grouped FP32 policy-only path applied independent group-relative terminal
  credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `c197aad9937dd0f8500943631c3ceb02fa696a969a197eb7e37ebbe4f2ad1274`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-028-lucario-targeted/report.md`
- `experiments/autoresearch/AR-028-lucario-targeted/manifest.json`
- `experiments/autoresearch/AR-028-lucario-targeted/metrics.json`
- `experiments/autoresearch/AR-028-lucario-targeted/sample.manifest.json`
- `experiments/autoresearch/AR-028-lucario-targeted/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-028-lucario-targeted/candidate.pt`

## Metrics

- Collection: `12.972843` s,
  `101.90518434839396` decisions/s.
- Update: `7.34693295811303` s; one optimizer step.
- Credited logical actions: `503`.
- Parameter L2 delta: `0.009207524589517763`;
  gradient norm `0.43632370233535767`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
