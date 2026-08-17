# State Capsule 042 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T08:01:44.868173+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `116` fibers with effective K
  `[2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `f8833898a92be9f70f7a33b44d64de27859072e2e8cc3920bedaa82bb1deb399`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C042/report.md`
- `experiments/autoresearch/AR-038-C042/manifest.json`
- `experiments/autoresearch/AR-038-C042/metrics.json`
- `experiments/autoresearch/AR-038-C042/sample.manifest.json`
- `experiments/autoresearch/AR-038-C042/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C042/candidate.pt`

## Metrics

- Collection: `45.113874` s,
  `148.40224083151205` decisions/s.
- Update: `477.1642948749941` s; `3` optimizer steps.
- Credited logical actions: `6695`.
- Parameter L2 delta: `0.0006043581547166398`;
  gradient norm `2.073535680770874`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
