# State Capsule 058 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T11:01:02.478156+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `56` exact recurrent sibling groups
  and `146` fibers with effective K
  `[2, 3, 4, 2, 4, 2, 3, 2, 2, 3, 3, 2, 2, 2, 2, 2, 4, 4, 2, 2, 3, 4, 2, 2, 2, 4, 3, 2, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4, 2, 4, 2, 4, 2, 2, 2, 3, 3, 2, 2, 2, 2, 4, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `40a4143cdf02b2f4dd4b653bb40681e809f1894361582a89bc5d11055f7b2270`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C058/report.md`
- `experiments/autoresearch/AR-038-C058/manifest.json`
- `experiments/autoresearch/AR-038-C058/metrics.json`
- `experiments/autoresearch/AR-038-C058/sample.manifest.json`
- `experiments/autoresearch/AR-038-C058/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C058/candidate.pt`

## Metrics

- Collection: `56.128813` s,
  `147.62471547104127` decisions/s.
- Update: `549.201406000182` s; `3` optimizer steps.
- Credited logical actions: `8286`.
- Parameter L2 delta: `0.0006042988957798189`;
  gradient norm `1.4649325609207153`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
