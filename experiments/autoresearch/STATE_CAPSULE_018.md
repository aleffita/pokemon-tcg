# State Capsule 018 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T03:36:27.753562+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `114` fibers with effective K
  `[2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `6e6859ed8fce617e73db9556783a0fd0eab3334f16b842c9dccd1edb769993de`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C018/report.md`
- `experiments/autoresearch/AR-038-C018/manifest.json`
- `experiments/autoresearch/AR-038-C018/metrics.json`
- `experiments/autoresearch/AR-038-C018/sample.manifest.json`
- `experiments/autoresearch/AR-038-C018/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C018/candidate.pt`

## Metrics

- Collection: `51.658064` s,
  `133.3770459911424` decisions/s.
- Update: `519.1056165830232` s; `3` optimizer steps.
- Credited logical actions: `6890`.
- Parameter L2 delta: `0.0006045886395897793`;
  gradient norm `2.64837908744812`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
