# State Capsule 022 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T04:20:04.696344+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `115` fibers with effective K
  `[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 4]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `a49350a1890d20138e40c89ffe675111842219123206d286b9439a0e431ebf06`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C022/report.md`
- `experiments/autoresearch/AR-038-C022/manifest.json`
- `experiments/autoresearch/AR-038-C022/metrics.json`
- `experiments/autoresearch/AR-038-C022/sample.manifest.json`
- `experiments/autoresearch/AR-038-C022/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C022/candidate.pt`

## Metrics

- Collection: `51.891797` s,
  `137.7674403109751` decisions/s.
- Update: `545.8848356672097` s; `3` optimizer steps.
- Credited logical actions: `7149`.
- Parameter L2 delta: `0.000604515225318246`;
  gradient norm `2.4990649223327637`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
