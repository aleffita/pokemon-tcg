# State Capsule 015 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T03:05:10.366470+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `116` fibers with effective K
  `[2, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `251538d074a6f4dd99a05a3dcdeda1731713281addeb90e7cdd807c8402be738`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C015/report.md`
- `experiments/autoresearch/AR-038-C015/manifest.json`
- `experiments/autoresearch/AR-038-C015/metrics.json`
- `experiments/autoresearch/AR-038-C015/sample.manifest.json`
- `experiments/autoresearch/AR-038-C015/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C015/candidate.pt`

## Metrics

- Collection: `61.897254` s,
  `111.15194146660315` decisions/s.
- Update: `495.4901914577931` s; `3` optimizer steps.
- Credited logical actions: `6880`.
- Parameter L2 delta: `0.0006044122562762378`;
  gradient norm `2.351273775100708`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
