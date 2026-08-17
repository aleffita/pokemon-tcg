# State Capsule 030 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T05:51:19.341808+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `120` fibers with effective K
  `[2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `7749bb8c0029c1140ed8b1e32f63c9585bfc4de5adf6acf64b24e7fd666f11ab`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C030/report.md`
- `experiments/autoresearch/AR-038-C030/manifest.json`
- `experiments/autoresearch/AR-038-C030/metrics.json`
- `experiments/autoresearch/AR-038-C030/sample.manifest.json`
- `experiments/autoresearch/AR-038-C030/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C030/candidate.pt`

## Metrics

- Collection: `51.99876` s,
  `149.6766451727082` decisions/s.
- Update: `575.3703289579134` s; `3` optimizer steps.
- Credited logical actions: `7783`.
- Parameter L2 delta: `0.0006043467145244025`;
  gradient norm `2.153181314468384`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
