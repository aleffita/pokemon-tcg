# State Capsule 064 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T12:05:02.968837+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `56` exact recurrent sibling groups
  and `123` fibers with effective K
  `[2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `9b1b2f3902e4feaf93cb6d08e4c960de1e1bbb414c1171a00d96cc3c9945503f`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C064/report.md`
- `experiments/autoresearch/AR-038-C064/manifest.json`
- `experiments/autoresearch/AR-038-C064/metrics.json`
- `experiments/autoresearch/AR-038-C064/sample.manifest.json`
- `experiments/autoresearch/AR-038-C064/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C064/candidate.pt`

## Metrics

- Collection: `50.283829` s,
  `142.47124995857268` decisions/s.
- Update: `508.2387280841358` s; `3` optimizer steps.
- Credited logical actions: `7164`.
- Parameter L2 delta: `0.0006042115229791181`;
  gradient norm `1.2595288753509521`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
