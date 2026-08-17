# State Capsule 043 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T08:13:16.897508+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `135` fibers with effective K
  `[4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 3, 2, 2, 4, 3, 2, 2, 4, 4, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 3, 4, 2, 2, 4, 4, 2, 2, 4]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `f9cad3da15a169114cc7dd90a2d8731713d9751dfa6d4029520bcfbfdf6730ef`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C043/report.md`
- `experiments/autoresearch/AR-038-C043/manifest.json`
- `experiments/autoresearch/AR-038-C043/metrics.json`
- `experiments/autoresearch/AR-038-C043/sample.manifest.json`
- `experiments/autoresearch/AR-038-C043/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C043/candidate.pt`

## Metrics

- Collection: `49.980833` s,
  `160.32145667298252` decisions/s.
- Update: `561.3817308750004` s; `3` optimizer steps.
- Credited logical actions: `8013`.
- Parameter L2 delta: `0.0006042051100746132`;
  gradient norm `1.5099130868911743`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
