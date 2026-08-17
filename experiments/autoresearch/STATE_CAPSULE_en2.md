# State Capsule en2 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-16T19:43:54.850069+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `8` exact recurrent sibling groups
  and `21` fibers with effective K
  `[2, 4, 2, 2, 2, 4, 3, 2]`.
- The grouped FP32 policy-only path applied independent group-relative terminal
  credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `bc07eb8507b86bdadebba1608681335d8dfc48cea5462fc547f725ad1f236300`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-028-multideck-screen2/report.md`
- `experiments/autoresearch/AR-028-multideck-screen2/manifest.json`
- `experiments/autoresearch/AR-028-multideck-screen2/metrics.json`
- `experiments/autoresearch/AR-028-multideck-screen2/sample.manifest.json`
- `experiments/autoresearch/AR-028-multideck-screen2/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-028-multideck-screen2/candidate.pt`

## Metrics

- Collection: `11.668447` s,
  `110.12605426982502` decisions/s.
- Update: `4.707561375107616` s; one optimizer step.
- Credited logical actions: `189`.
- Parameter L2 delta: `0.009211690045174248`;
  gradient norm `1.2298381328582764`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
