# State Capsule 066 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T12:24:59.205673+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `56` exact recurrent sibling groups
  and `127` fibers with effective K
  `[2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `b7ab038c4ae7c4788e23a216eef79218385d5c14ed31d0ea7f0460126d289b6c`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C066/report.md`
- `experiments/autoresearch/AR-038-C066/manifest.json`
- `experiments/autoresearch/AR-038-C066/metrics.json`
- `experiments/autoresearch/AR-038-C066/sample.manifest.json`
- `experiments/autoresearch/AR-038-C066/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C066/candidate.pt`

## Metrics

- Collection: `45.944075` s,
  `145.22003198849131` decisions/s.
- Update: `466.9949409170076` s; `3` optimizer steps.
- Credited logical actions: `6672`.
- Parameter L2 delta: `0.0006039605352235374`;
  gradient norm `0.9941917061805725`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
