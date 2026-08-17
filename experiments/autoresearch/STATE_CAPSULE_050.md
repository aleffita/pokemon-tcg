# State Capsule 050 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T09:30:06.637014+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `146` fibers with effective K
  `[2, 2, 2, 2, 2, 2, 3, 2, 2, 4, 4, 2, 2, 2, 4, 4, 2, 2, 4, 4, 2, 3, 3, 4, 2, 4, 4, 4, 2, 4, 4, 4, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 3, 4, 2, 4, 4, 4]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `6d87dcab28be481fa6c5a04ec072640c1e0b3b62849dedf97b413021451e29fb`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C050/report.md`
- `experiments/autoresearch/AR-038-C050/manifest.json`
- `experiments/autoresearch/AR-038-C050/metrics.json`
- `experiments/autoresearch/AR-038-C050/sample.manifest.json`
- `experiments/autoresearch/AR-038-C050/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C050/candidate.pt`

## Metrics

- Collection: `54.928862` s,
  `158.96925092601714` decisions/s.
- Update: `594.2071386249736` s; `3` optimizer steps.
- Credited logical actions: `8732`.
- Parameter L2 delta: `0.0006042747922165391`;
  gradient norm `1.396795630455017`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
