# State Capsule 039 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T07:30:00.724650+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `121` fibers with effective K
  `[2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `6bcb646d14e1c75c9a8f3d9386d8a1ebf7fec0bf4baa51a608dfdbfde2771478`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C039/report.md`
- `experiments/autoresearch/AR-038-C039/manifest.json`
- `experiments/autoresearch/AR-038-C039/metrics.json`
- `experiments/autoresearch/AR-038-C039/sample.manifest.json`
- `experiments/autoresearch/AR-038-C039/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C039/candidate.pt`

## Metrics

- Collection: `47.015385` s,
  `152.20549733843475` decisions/s.
- Update: `528.3339205409866` s; `3` optimizer steps.
- Credited logical actions: `7156`.
- Parameter L2 delta: `0.0006042946231816791`;
  gradient norm `1.9278353452682495`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
