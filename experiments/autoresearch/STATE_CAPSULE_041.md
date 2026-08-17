# State Capsule 041 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T07:51:49.124166+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `134` fibers with effective K
  `[2, 2, 4, 2, 3, 2, 4, 2, 2, 2, 4, 2, 4, 2, 4, 2, 3, 2, 4, 2, 3, 2, 2, 2, 4, 2, 2, 2, 4, 2, 4, 2, 4, 2, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 4, 2, 4, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `93dd83094795515e0991d53f81594e84b6b91b1bbd7bf0ce7016b3a66ab0a312`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C041/report.md`
- `experiments/autoresearch/AR-038-C041/manifest.json`
- `experiments/autoresearch/AR-038-C041/metrics.json`
- `experiments/autoresearch/AR-038-C041/sample.manifest.json`
- `experiments/autoresearch/AR-038-C041/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C041/candidate.pt`

## Metrics

- Collection: `48.218358` s,
  `157.49188186754807` decisions/s.
- Update: `541.3315421249717` s; `3` optimizer steps.
- Credited logical actions: `7594`.
- Parameter L2 delta: `0.0006044815251039922`;
  gradient norm `1.830335021018982`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
