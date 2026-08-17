# State Capsule 055 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T10:26:51.359620+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `56` exact recurrent sibling groups
  and `135` fibers with effective K
  `[2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 3, 2, 2, 4, 3]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `fdadcb34a4d418e555c98749b7c35d3b46d42330ffccf2a813c1aaed69e06300`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C055/report.md`
- `experiments/autoresearch/AR-038-C055/manifest.json`
- `experiments/autoresearch/AR-038-C055/metrics.json`
- `experiments/autoresearch/AR-038-C055/sample.manifest.json`
- `experiments/autoresearch/AR-038-C055/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C055/candidate.pt`

## Metrics

- Collection: `50.767894` s,
  `154.05405666782798` decisions/s.
- Update: `531.7276074998081` s; `3` optimizer steps.
- Credited logical actions: `7821`.
- Parameter L2 delta: `0.000604080859133191`;
  gradient norm `1.2005813121795654`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
