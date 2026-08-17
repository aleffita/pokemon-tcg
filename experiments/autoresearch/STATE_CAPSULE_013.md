# State Capsule 013 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T02:40:39.969203+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `140` fibers with effective K
  `[3, 2, 2, 2, 3, 3, 2, 4, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 4, 3, 2, 4, 4, 2, 2, 4, 2, 4, 2, 4, 4, 4, 2, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 2, 4, 4, 2, 3]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `0ec57af31faf72ba9333c6253b305f4ca872715a058ebc671c7d7b9f3aae9ec6`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C013/report.md`
- `experiments/autoresearch/AR-038-C013/manifest.json`
- `experiments/autoresearch/AR-038-C013/metrics.json`
- `experiments/autoresearch/AR-038-C013/sample.manifest.json`
- `experiments/autoresearch/AR-038-C013/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C013/candidate.pt`

## Metrics

- Collection: `68.645423` s,
  `119.46899868676599` decisions/s.
- Update: `665.6389077079948` s; `3` optimizer steps.
- Credited logical actions: `8201`.
- Parameter L2 delta: `0.0006044805740616241`;
  gradient norm `2.497986078262329`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
