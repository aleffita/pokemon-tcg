# State Capsule 048 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T09:07:04.659810+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `117` fibers with effective K
  `[2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 4]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `0192df2b5dd2ba5c56532e0aeda2429850e6229898256b6b8ccfbedcb9505241`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C048/report.md`
- `experiments/autoresearch/AR-038-C048/manifest.json`
- `experiments/autoresearch/AR-038-C048/metrics.json`
- `experiments/autoresearch/AR-038-C048/sample.manifest.json`
- `experiments/autoresearch/AR-038-C048/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C048/candidate.pt`

## Metrics

- Collection: `47.957354` s,
  `154.26205647789533` decisions/s.
- Update: `509.360814207932` s; `3` optimizer steps.
- Credited logical actions: `7398`.
- Parameter L2 delta: `0.0006045005804452916`;
  gradient norm `1.6268922090530396`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
