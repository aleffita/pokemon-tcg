# State Capsule 009 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T01:46:41.042212+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `127` fibers with effective K
  `[2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 4, 2, 2, 2, 4, 2, 4, 2, 3, 2, 2, 2, 4, 2, 4, 2, 3, 2, 4, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `eadc99ece4e0f6774bc80920830f2b9be562d73c12b0a0c15419a87eadc315d1`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C009/report.md`
- `experiments/autoresearch/AR-038-C009/manifest.json`
- `experiments/autoresearch/AR-038-C009/metrics.json`
- `experiments/autoresearch/AR-038-C009/sample.manifest.json`
- `experiments/autoresearch/AR-038-C009/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C009/candidate.pt`

## Metrics

- Collection: `64.404034` s,
  `116.90261559575158` decisions/s.
- Update: `621.5541926671285` s; `3` optimizer steps.
- Credited logical actions: `7529`.
- Parameter L2 delta: `0.0006044952309985285`;
  gradient norm `2.6580748558044434`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
