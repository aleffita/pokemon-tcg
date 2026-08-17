# State Capsule 028 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T05:29:12.008723+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `116` fibers with effective K
  `[2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `c1e49448438284aa05576b5f6bb0dfb1a738beb8de9f572769f660e87bea63e8`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C028/report.md`
- `experiments/autoresearch/AR-038-C028/manifest.json`
- `experiments/autoresearch/AR-038-C028/metrics.json`
- `experiments/autoresearch/AR-038-C028/sample.manifest.json`
- `experiments/autoresearch/AR-038-C028/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C028/candidate.pt`

## Metrics

- Collection: `46.965048` s,
  `149.19605878803392` decisions/s.
- Update: `514.1155757501256` s; `3` optimizer steps.
- Credited logical actions: `7007`.
- Parameter L2 delta: `0.0006043979700469671`;
  gradient norm `2.2538790702819824`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
