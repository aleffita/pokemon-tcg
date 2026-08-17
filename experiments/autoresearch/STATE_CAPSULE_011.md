# State Capsule 011 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T02:13:52.539952+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `132` fibers with effective K
  `[2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 4, 2, 2, 2, 4, 2, 2, 3, 4, 2, 2, 3, 2, 2, 3, 2, 2, 2, 4, 4, 2, 2, 3, 4, 4, 2, 2, 2, 2, 2, 4, 2, 3, 2, 4, 2, 4, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `6cba06d47da3c6e51fe686d2cbd3333b172bab35fe26fc4a9564816affbca141`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C011/report.md`
- `experiments/autoresearch/AR-038-C011/manifest.json`
- `experiments/autoresearch/AR-038-C011/metrics.json`
- `experiments/autoresearch/AR-038-C011/sample.manifest.json`
- `experiments/autoresearch/AR-038-C011/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C011/candidate.pt`

## Metrics

- Collection: `67.119196` s,
  `119.26543361531121` decisions/s.
- Update: `663.5086267089937` s; `3` optimizer steps.
- Credited logical actions: `8005`.
- Parameter L2 delta: `0.0006044740900551635`;
  gradient norm `2.6808128356933594`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
