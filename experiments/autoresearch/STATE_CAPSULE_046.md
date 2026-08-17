# State Capsule 046 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T08:46:32.395016+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `131` fibers with effective K
  `[4, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 3, 2, 2, 2, 4, 3, 2, 2, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4, 4, 2, 2, 4, 3, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `61cf766fb79090ab487416b3f909fb0c37b8cd98f9cad2c4ba66e9eecfcc0bcd`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C046/report.md`
- `experiments/autoresearch/AR-038-C046/manifest.json`
- `experiments/autoresearch/AR-038-C046/metrics.json`
- `experiments/autoresearch/AR-038-C046/sample.manifest.json`
- `experiments/autoresearch/AR-038-C046/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C046/candidate.pt`

## Metrics

- Collection: `49.218061` s,
  `155.83710319778464` decisions/s.
- Update: `536.1700510829687` s; `3` optimizer steps.
- Credited logical actions: `7670`.
- Parameter L2 delta: `0.0006041729093548587`;
  gradient norm `1.5143389701843262`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
