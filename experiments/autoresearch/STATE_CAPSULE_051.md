# State Capsule 051 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T09:42:51.573284+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `155` fibers with effective K
  `[2, 3, 2, 4, 4, 2, 2, 2, 4, 3, 4, 4, 4, 2, 3, 4, 4, 4, 3, 2, 4, 4, 2, 4, 4, 2, 3, 2, 4, 3, 4, 2, 3, 4, 2, 4, 2, 3, 2, 2, 2, 2, 2, 4, 4, 3, 4, 2, 2, 2, 4, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `62fac94117039ed7290d41c5fcb0f1a05c14308ef37dfbd342d566d55d43a231`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C051/report.md`
- `experiments/autoresearch/AR-038-C051/manifest.json`
- `experiments/autoresearch/AR-038-C051/metrics.json`
- `experiments/autoresearch/AR-038-C051/sample.manifest.json`
- `experiments/autoresearch/AR-038-C051/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C051/candidate.pt`

## Metrics

- Collection: `54.129071` s,
  `166.5833139230633` decisions/s.
- Update: `615.8242473748978` s; `3` optimizer steps.
- Credited logical actions: `9017`.
- Parameter L2 delta: `0.0006041665105896504`;
  gradient norm `1.2597469091415405`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
