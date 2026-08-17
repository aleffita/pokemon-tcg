# State Capsule 040 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T07:40:39.412542+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `124` fibers with effective K
  `[2, 2, 2, 3, 2, 3, 2, 3, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 3, 2, 4, 2, 4, 2, 3, 2, 3, 2, 2, 2, 3, 2, 4, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `299c5c94a6499a3c13d598048a468ffd3b564b14bff5e94990a76c34de5baa5a`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C040/report.md`
- `experiments/autoresearch/AR-038-C040/manifest.json`
- `experiments/autoresearch/AR-038-C040/metrics.json`
- `experiments/autoresearch/AR-038-C040/sample.manifest.json`
- `experiments/autoresearch/AR-038-C040/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C040/candidate.pt`

## Metrics

- Collection: `47.52767` s,
  `151.44862010556156` decisions/s.
- Update: `514.4598009157926` s; `3` optimizer steps.
- Credited logical actions: `7198`.
- Parameter L2 delta: `0.0006044026858580858`;
  gradient norm `1.526237964630127`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
