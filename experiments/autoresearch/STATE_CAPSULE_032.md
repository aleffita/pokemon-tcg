# State Capsule 032 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T06:14:13.108073+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `130` fibers with effective K
  `[2, 4, 3, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 4, 2, 2, 2, 4, 2, 2, 4, 4, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 4, 2, 2, 2, 4, 4, 2, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `9337bf66f7102075e8df792ae1c99b8d66d484baac67344244bdf411554cf2dc`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C032/report.md`
- `experiments/autoresearch/AR-038-C032/manifest.json`
- `experiments/autoresearch/AR-038-C032/metrics.json`
- `experiments/autoresearch/AR-038-C032/sample.manifest.json`
- `experiments/autoresearch/AR-038-C032/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C032/candidate.pt`

## Metrics

- Collection: `49.077968` s,
  `152.81806197554297` decisions/s.
- Update: `538.3492569159716` s; `3` optimizer steps.
- Credited logical actions: `7500`.
- Parameter L2 delta: `0.0006044663856278215`;
  gradient norm `2.2496449947357178`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
