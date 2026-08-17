# State Capsule 019 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T03:47:20.143890+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `112` fibers with effective K
  `[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `900ffc050918e247907abaffa6206c25ca1a2be2486b0ae7f73ff670bd010628`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C019/report.md`
- `experiments/autoresearch/AR-038-C019/manifest.json`
- `experiments/autoresearch/AR-038-C019/metrics.json`
- `experiments/autoresearch/AR-038-C019/sample.manifest.json`
- `experiments/autoresearch/AR-038-C019/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C019/candidate.pt`

## Metrics

- Collection: `51.922391` s,
  `134.85511417539723` decisions/s.
- Update: `526.1599641670473` s; `3` optimizer steps.
- Credited logical actions: `7002`.
- Parameter L2 delta: `0.0006044137710405266`;
  gradient norm `2.6576902866363525`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
