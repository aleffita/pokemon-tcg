# State Capsule 045 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T08:35:24.013958+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `131` fibers with effective K
  `[2, 4, 4, 2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 3, 3, 2, 2, 4, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 3, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `805d5d953cb5d249c647dde94c5e925635fcbb2b66808c89adb1d0a93af56d15`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C045/report.md`
- `experiments/autoresearch/AR-038-C045/manifest.json`
- `experiments/autoresearch/AR-038-C045/metrics.json`
- `experiments/autoresearch/AR-038-C045/sample.manifest.json`
- `experiments/autoresearch/AR-038-C045/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C045/candidate.pt`

## Metrics

- Collection: `51.241129` s,
  `159.75448041046369` decisions/s.
- Update: `575.0811674168799` s; `3` optimizer steps.
- Credited logical actions: `8186`.
- Parameter L2 delta: `0.0006043771141889598`;
  gradient norm `1.7686690092086792`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
