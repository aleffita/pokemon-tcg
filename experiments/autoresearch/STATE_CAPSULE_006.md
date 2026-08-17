# State Capsule 006 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T01:08:42.213143+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `127` fibers with effective K
  `[2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 3, 2, 4, 2, 4, 2, 4, 2, 2, 2, 2, 2, 4, 2, 4, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 4, 2, 3, 2, 2, 2, 2, 2, 4]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `4b2a1232d7f362b214d9fb055a3893bf63b243995c4e4feca3c4e15aef027970`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C006/report.md`
- `experiments/autoresearch/AR-038-C006/manifest.json`
- `experiments/autoresearch/AR-038-C006/metrics.json`
- `experiments/autoresearch/AR-038-C006/sample.manifest.json`
- `experiments/autoresearch/AR-038-C006/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C006/candidate.pt`

## Metrics

- Collection: `62.586682` s,
  `114.67295808966898` decisions/s.
- Update: `575.2165405410342` s; `3` optimizer steps.
- Credited logical actions: `7177`.
- Parameter L2 delta: `0.0006043353230399892`;
  gradient norm `2.3242244720458984`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
