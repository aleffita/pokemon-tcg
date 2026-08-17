# State Capsule 063 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T11:54:27.693672+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `56` exact recurrent sibling groups
  and `127` fibers with effective K
  `[3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `f9a0cd21901cd60a6e76d74c965a25375cd760a73e59701e2f95d18380a1a577`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C063/report.md`
- `experiments/autoresearch/AR-038-C063/manifest.json`
- `experiments/autoresearch/AR-038-C063/metrics.json`
- `experiments/autoresearch/AR-038-C063/sample.manifest.json`
- `experiments/autoresearch/AR-038-C063/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C063/candidate.pt`

## Metrics

- Collection: `49.316969` s,
  `143.29753277101113` decisions/s.
- Update: `501.5070090419613` s; `3` optimizer steps.
- Credited logical actions: `7067`.
- Parameter L2 delta: `0.0006041686338814523`;
  gradient norm `1.2720937728881836`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
