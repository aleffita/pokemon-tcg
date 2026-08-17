# State Capsule 047 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T08:56:29.716289+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `120` fibers with effective K
  `[4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `bbbdf768e9851ad7d69da679277c519f080996c3020f53a03fe4a7ac4781e465`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C047/report.md`
- `experiments/autoresearch/AR-038-C047/manifest.json`
- `experiments/autoresearch/AR-038-C047/metrics.json`
- `experiments/autoresearch/AR-038-C047/sample.manifest.json`
- `experiments/autoresearch/AR-038-C047/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C047/candidate.pt`

## Metrics

- Collection: `44.691721` s,
  `156.0915515254043` decisions/s.
- Update: `475.5303699169308` s; `3` optimizer steps.
- Credited logical actions: `6976`.
- Parameter L2 delta: `0.0006042774478460386`;
  gradient norm `1.6134233474731445`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
