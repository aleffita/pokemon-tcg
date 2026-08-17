# State Capsule 054 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T10:15:46.722120+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `56` exact recurrent sibling groups
  and `134` fibers with effective K
  `[2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 3, 2, 2, 4, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 3, 2, 2, 2, 4]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `15bed68ab5b43496e274f9ab19bf440e04897666599afb79bd461e563447447d`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C054/report.md`
- `experiments/autoresearch/AR-038-C054/manifest.json`
- `experiments/autoresearch/AR-038-C054/metrics.json`
- `experiments/autoresearch/AR-038-C054/sample.manifest.json`
- `experiments/autoresearch/AR-038-C054/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C054/candidate.pt`

## Metrics

- Collection: `47.681614` s,
  `157.77569945860373` decisions/s.
- Update: `509.4457824998535` s; `3` optimizer steps.
- Credited logical actions: `7523`.
- Parameter L2 delta: `0.0006043010671068488`;
  gradient norm `1.3844183683395386`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
