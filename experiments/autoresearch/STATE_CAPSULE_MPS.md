# State Capsule MPS - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-16T20:28:51.778825+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `4` exact recurrent sibling groups
  and `8` fibers with effective K
  `[2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `159a750ae667cd126c2b7a38585d9cabea38ad980e98596631e516d016def9b6`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-030-MPS/report.md`
- `experiments/autoresearch/AR-030-MPS/manifest.json`
- `experiments/autoresearch/AR-030-MPS/metrics.json`
- `experiments/autoresearch/AR-030-MPS/sample.manifest.json`
- `experiments/autoresearch/AR-030-MPS/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-030-MPS/candidate.pt`

## Metrics

- Collection: `3.443439` s,
  `130.97371552785157` decisions/s.
- Update: `12.793072083033621` s; `1` optimizer steps.
- Credited logical actions: `0`.
- Parameter L2 delta: `0.0029694303841592312`;
  gradient norm `3.10003662109375`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
