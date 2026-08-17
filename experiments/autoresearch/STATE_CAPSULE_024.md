# State Capsule 024 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T04:42:47.156795+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `128` fibers with effective K
  `[2, 2, 2, 4, 2, 2, 2, 3, 2, 3, 2, 4, 2, 2, 2, 4, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 3, 2, 4, 2, 3, 2, 4, 2, 3, 2, 2, 2, 3, 2, 2, 2, 4, 2, 3, 2, 4, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `ad970b799172bc2fb1e35643f231a7436c8d461302a0ec6f5d57298e31ee6aaf`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C024/report.md`
- `experiments/autoresearch/AR-038-C024/manifest.json`
- `experiments/autoresearch/AR-038-C024/metrics.json`
- `experiments/autoresearch/AR-038-C024/sample.manifest.json`
- `experiments/autoresearch/AR-038-C024/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C024/candidate.pt`

## Metrics

- Collection: `54.496178` s,
  `148.52417780345314` decisions/s.
- Update: `594.385998666985` s; `3` optimizer steps.
- Credited logical actions: `8094`.
- Parameter L2 delta: `0.0006043477489085142`;
  gradient norm `2.196876049041748`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
