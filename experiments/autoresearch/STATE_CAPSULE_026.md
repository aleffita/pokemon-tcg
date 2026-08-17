# State Capsule 026 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T05:06:49.744022+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `138` fibers with effective K
  `[2, 4, 2, 4, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 4, 2, 3, 2, 3, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 3, 2, 3, 2, 2, 2, 4, 2, 4, 2, 3, 2, 4, 2, 2, 2, 4, 2, 4, 2, 4]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `0297dfc5992671d1fb25d8a8614d02ee745361230a5b3c1bba9c546e43160f75`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C026/report.md`
- `experiments/autoresearch/AR-038-C026/manifest.json`
- `experiments/autoresearch/AR-038-C026/metrics.json`
- `experiments/autoresearch/AR-038-C026/sample.manifest.json`
- `experiments/autoresearch/AR-038-C026/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C026/candidate.pt`

## Metrics

- Collection: `50.027157` s,
  `153.4966302354223` decisions/s.
- Update: `579.5441676250193` s; `3` optimizer steps.
- Credited logical actions: `7679`.
- Parameter L2 delta: `0.0006043588394478249`;
  gradient norm `2.355170965194702`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
