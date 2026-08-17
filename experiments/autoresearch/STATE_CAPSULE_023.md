# State Capsule 023 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T04:30:36.285333+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `116` fibers with effective K
  `[2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 3, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `42c2315bd0a4a91e945e3b4f49832f3f0f9cc17af0c7292c85cbac4f1c130aa1`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C023/report.md`
- `experiments/autoresearch/AR-038-C023/manifest.json`
- `experiments/autoresearch/AR-038-C023/metrics.json`
- `experiments/autoresearch/AR-038-C023/sample.manifest.json`
- `experiments/autoresearch/AR-038-C023/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C023/candidate.pt`

## Metrics

- Collection: `50.979183` s,
  `133.42701003689632` decisions/s.
- Update: `507.35383712500334` s; `3` optimizer steps.
- Credited logical actions: `6802`.
- Parameter L2 delta: `0.0006046156085142514`;
  gradient norm `1.913076639175415`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
