# State Capsule 059 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T11:11:55.967531+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `56` exact recurrent sibling groups
  and `138` fibers with effective K
  `[4, 4, 2, 2, 2, 3, 2, 2, 3, 4, 2, 2, 4, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 4, 4, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `889db1571fc0ccacc3a491e130b2776630ed29b056ffde64d96ce8277872acde`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C059/report.md`
- `experiments/autoresearch/AR-038-C059/manifest.json`
- `experiments/autoresearch/AR-038-C059/metrics.json`
- `experiments/autoresearch/AR-038-C059/sample.manifest.json`
- `experiments/autoresearch/AR-038-C059/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C059/candidate.pt`

## Metrics

- Collection: `49.587573` s,
  `146.32698321988047` decisions/s.
- Update: `522.7526780411135` s; `3` optimizer steps.
- Credited logical actions: `7256`.
- Parameter L2 delta: `0.0006042075239127325`;
  gradient norm `1.0589689016342163`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
