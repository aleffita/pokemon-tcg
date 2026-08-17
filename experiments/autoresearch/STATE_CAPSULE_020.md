# State Capsule 020 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T03:57:41.537715+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `117` fibers with effective K
  `[2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `3d0f91da9507f8b3276f704cceca1e814add784dce244a89f10efe6ce253a641`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C020/report.md`
- `experiments/autoresearch/AR-038-C020/manifest.json`
- `experiments/autoresearch/AR-038-C020/metrics.json`
- `experiments/autoresearch/AR-038-C020/sample.manifest.json`
- `experiments/autoresearch/AR-038-C020/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C020/candidate.pt`

## Metrics

- Collection: `49.597698` s,
  `135.9538901086931` decisions/s.
- Update: `499.52563354186714` s; `3` optimizer steps.
- Credited logical actions: `6743`.
- Parameter L2 delta: `0.0006044595348442313`;
  gradient norm `2.38761830329895`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
