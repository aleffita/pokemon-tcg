# State Capsule 016 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T03:15:49.472615+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `115` fibers with effective K
  `[4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `cc9cd8bdc53b3dae312eb41681ee5d9fbbe0b1136d612d8abe270a2d836772a9`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C016/report.md`
- `experiments/autoresearch/AR-038-C016/manifest.json`
- `experiments/autoresearch/AR-038-C016/metrics.json`
- `experiments/autoresearch/AR-038-C016/sample.manifest.json`
- `experiments/autoresearch/AR-038-C016/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C016/candidate.pt`

## Metrics

- Collection: `48.583778` s,
  `140.1084942856966` decisions/s.
- Update: `517.4007994169369` s; `3` optimizer steps.
- Credited logical actions: `6807`.
- Parameter L2 delta: `0.0006044178605978324`;
  gradient norm `2.340405225753784`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
