# State Capsule 049 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T09:17:48.924838+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `123` fibers with effective K
  `[2, 2, 3, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 4, 2, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 3]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `641a050d8233250a2b54fd6ddf2f6170cd4a8124df2983a7206167fabc30a378`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C049/report.md`
- `experiments/autoresearch/AR-038-C049/manifest.json`
- `experiments/autoresearch/AR-038-C049/metrics.json`
- `experiments/autoresearch/AR-038-C049/sample.manifest.json`
- `experiments/autoresearch/AR-038-C049/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C049/candidate.pt`

## Metrics

- Collection: `48.087278` s,
  `154.96822110064568` decisions/s.
- Update: `517.2864891670179` s; `3` optimizer steps.
- Credited logical actions: `7452`.
- Parameter L2 delta: `0.0006044072346785085`;
  gradient norm `1.4616565704345703`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
