# State Capsule 0-I - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-16T20:49:54.691001+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `4` exact recurrent sibling groups
  and `8` fibers with effective K
  `[2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `595527f6adc76c121749afd04244f0041e3931c4d43be87216bad671fceef26d`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-030-I/report.md`
- `experiments/autoresearch/AR-030-I/manifest.json`
- `experiments/autoresearch/AR-030-I/metrics.json`
- `experiments/autoresearch/AR-030-I/sample.manifest.json`
- `experiments/autoresearch/AR-030-I/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-030-I/candidate.pt`

## Metrics

- Collection: `4.345669` s,
  `146.58270998381093` decisions/s.
- Update: `13.572289374889806` s; `2` optimizer steps.
- Credited logical actions: `0`.
- Parameter L2 delta: `0.00594031438784034`;
  gradient norm `2.353379487991333`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
