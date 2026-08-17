# State Capsule 010 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T02:00:14.173972+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `134` fibers with effective K
  `[2, 4, 2, 2, 2, 2, 2, 3, 2, 4, 2, 4, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 4, 2, 4, 2, 2, 2, 3, 2, 4, 2, 3, 2, 2, 2, 4, 2, 3, 2, 2, 2, 2, 2, 4, 2, 4]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `b81c52c9528857db1d274968812f95b43a0965d05fb74d5827d589e8f1fcfc1e`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C010/report.md`
- `experiments/autoresearch/AR-038-C010/manifest.json`
- `experiments/autoresearch/AR-038-C010/metrics.json`
- `experiments/autoresearch/AR-038-C010/sample.manifest.json`
- `experiments/autoresearch/AR-038-C010/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C010/candidate.pt`

## Metrics

- Collection: `69.084145` s,
  `115.74290998150707` decisions/s.
- Update: `659.9243497920688` s; `3` optimizer steps.
- Credited logical actions: `7996`.
- Parameter L2 delta: `0.0006046231788196982`;
  gradient norm `2.7018566131591797`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
