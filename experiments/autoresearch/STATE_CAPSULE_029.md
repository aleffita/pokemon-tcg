# State Capsule 029 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T05:39:31.273539+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `115` fibers with effective K
  `[2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `477212fd5c432d4e0e82d4176bc279bfa3ecd8c4d0bc3067c0c0920aea2915a8`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C029/report.md`
- `experiments/autoresearch/AR-038-C029/manifest.json`
- `experiments/autoresearch/AR-038-C029/metrics.json`
- `experiments/autoresearch/AR-038-C029/sample.manifest.json`
- `experiments/autoresearch/AR-038-C029/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C029/candidate.pt`

## Metrics

- Collection: `47.61435` s,
  `146.5314567434241` decisions/s.
- Update: `497.4033568338491` s; `3` optimizer steps.
- Credited logical actions: `6977`.
- Parameter L2 delta: `0.0006043102424933163`;
  gradient norm `2.2608845233917236`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
