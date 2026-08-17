# State Capsule 031 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T06:03:05.054750+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `123` fibers with effective K
  `[2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 4, 3, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 3]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `ee9202c9a1ca56c593d8f05918d53266f8a9a9745d5b03baa8715805c84fcf42`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C031/report.md`
- `experiments/autoresearch/AR-038-C031/manifest.json`
- `experiments/autoresearch/AR-038-C031/metrics.json`
- `experiments/autoresearch/AR-038-C031/sample.manifest.json`
- `experiments/autoresearch/AR-038-C031/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C031/candidate.pt`

## Metrics

- Collection: `51.478686` s,
  `151.88810239787102` decisions/s.
- Update: `570.5586287498008` s; `3` optimizer steps.
- Credited logical actions: `7819`.
- Parameter L2 delta: `0.0006043559044243805`;
  gradient norm `1.965256929397583`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
