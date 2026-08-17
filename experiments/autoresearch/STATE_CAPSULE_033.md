# State Capsule 033 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T06:25:17.915917+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `123` fibers with effective K
  `[4, 4, 2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `40f0745657524f5defd632e7e32dd2c0224da003ed3e6daebc44c874c30db3d0`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C033/report.md`
- `experiments/autoresearch/AR-038-C033/manifest.json`
- `experiments/autoresearch/AR-038-C033/metrics.json`
- `experiments/autoresearch/AR-038-C033/sample.manifest.json`
- `experiments/autoresearch/AR-038-C033/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C033/candidate.pt`

## Metrics

- Collection: `48.935073` s,
  `152.7738596443724` decisions/s.
- Update: `535.3098367501516` s; `3` optimizer steps.
- Credited logical actions: `7476`.
- Parameter L2 delta: `0.0006043154498864785`;
  gradient norm `1.806110143661499`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
