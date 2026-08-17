# State Capsule 037 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T07:09:10.239681+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `115` fibers with effective K
  `[2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `eab363f7b1fa070214949fed83e93a42fffff77af44f62549cef53bfd9d03cf2`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C037/report.md`
- `experiments/autoresearch/AR-038-C037/manifest.json`
- `experiments/autoresearch/AR-038-C037/metrics.json`
- `experiments/autoresearch/AR-038-C037/sample.manifest.json`
- `experiments/autoresearch/AR-038-C037/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C037/candidate.pt`

## Metrics

- Collection: `46.228831` s,
  `153.21607460310648` decisions/s.
- Update: `493.89690629113466` s; `3` optimizer steps.
- Credited logical actions: `7083`.
- Parameter L2 delta: `0.0006044446637500953`;
  gradient norm `2.006361722946167`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
