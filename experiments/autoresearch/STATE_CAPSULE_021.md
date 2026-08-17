# State Capsule 021 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T04:08:50.601083+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `120` fibers with effective K
  `[4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `218aaaa42aaa7594da8cccf8e97d922cbe3878b6900fb3922789d4384b2f104a`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C021/report.md`
- `experiments/autoresearch/AR-038-C021/manifest.json`
- `experiments/autoresearch/AR-038-C021/metrics.json`
- `experiments/autoresearch/AR-038-C021/sample.manifest.json`
- `experiments/autoresearch/AR-038-C021/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C021/candidate.pt`

## Metrics

- Collection: `51.412744` s,
  `140.0625499267185` decisions/s.
- Update: `542.6486431248486` s; `3` optimizer steps.
- Credited logical actions: `7201`.
- Parameter L2 delta: `0.0006043751007229699`;
  gradient norm `2.1033995151519775`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
