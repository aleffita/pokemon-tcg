# State Capsule 034 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T06:37:42.059240+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `129` fibers with effective K
  `[2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 3, 4, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 4, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 3]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `bf7a2f724d40ce62e15c2b15dd4eae77ac2ccb361b422b5c3e09954bf6760df0`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C034/report.md`
- `experiments/autoresearch/AR-038-C034/manifest.json`
- `experiments/autoresearch/AR-038-C034/metrics.json`
- `experiments/autoresearch/AR-038-C034/sample.manifest.json`
- `experiments/autoresearch/AR-038-C034/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C034/candidate.pt`

## Metrics

- Collection: `51.343096` s,
  `161.4822770631944` decisions/s.
- Update: `606.9407484161202` s; `3` optimizer steps.
- Credited logical actions: `8291`.
- Parameter L2 delta: `0.0006045249491257666`;
  gradient norm `2.2921218872070312`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
