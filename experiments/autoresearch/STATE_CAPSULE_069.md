# State Capsule 069 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T12:54:09.101560+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `56` exact recurrent sibling groups
  and `112` fibers with effective K
  `[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `6f197deaf69e81e4e8a210edb14a3abfed989d4d07fc8d564d960f5a5f09532f`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C069/report.md`
- `experiments/autoresearch/AR-038-C069/manifest.json`
- `experiments/autoresearch/AR-038-C069/metrics.json`
- `experiments/autoresearch/AR-038-C069/sample.manifest.json`
- `experiments/autoresearch/AR-038-C069/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C069/candidate.pt`

## Metrics

- Collection: `44.800234` s,
  `140.60194430021377` decisions/s.
- Update: `449.1285297500435` s; `3` optimizer steps.
- Credited logical actions: `6299`.
- Parameter L2 delta: `0.000604078675246876`;
  gradient norm `1.1569080352783203`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
