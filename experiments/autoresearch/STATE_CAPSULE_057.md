# State Capsule 057 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T10:49:30.127872+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `56` exact recurrent sibling groups
  and `158` fibers with effective K
  `[2, 2, 2, 2, 3, 2, 4, 3, 2, 3, 3, 2, 2, 2, 4, 2, 4, 3, 2, 4, 4, 2, 4, 4, 4, 2, 4, 2, 2, 2, 2, 4, 2, 3, 4, 4, 2, 4, 2, 3, 2, 4, 2, 4, 2, 2, 2, 3, 4, 4, 3, 4, 2, 2, 2, 3]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `e41f49d0dfccfddfe542db46adb9a3351574034f82b71f4e6492f7682e7aa450`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C057/report.md`
- `experiments/autoresearch/AR-038-C057/manifest.json`
- `experiments/autoresearch/AR-038-C057/metrics.json`
- `experiments/autoresearch/AR-038-C057/sample.manifest.json`
- `experiments/autoresearch/AR-038-C057/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C057/candidate.pt`

## Metrics

- Collection: `50.587381` s,
  `161.2655150897086` decisions/s.
- Update: `540.1285690830555` s; `3` optimizer steps.
- Credited logical actions: `8158`.
- Parameter L2 delta: `0.0006039196223171115`;
  gradient norm `0.9871091246604919`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
