# State Capsule 019 - first trajectory-group GRPO micro-update

Captured 2026-08-16T16:42:34.464748+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-019 collected K=4 current-vs-current true recurrent trajectories in
  memory and applied one FP32 policy-only trajectory-group GRPO update.
- Candidate: `2e592340522c697405676811f4d68c53c93129fd89b8f14b07427d4624ed77e7` (5693904 bytes).
- Candidate parameter L2 delta: `0.009198900828294767`; changed parameters:
  `1293290`.
- No tournament, package, submission, promotion, MoE, RoPE-ND, or historical
  ETL/Parquet/packed-data work was run.

## Evidence

- `experiments/autoresearch/AR-019/report.md`
- `experiments/autoresearch/AR-019/manifest.json`
- `experiments/autoresearch/AR-019/metrics.json`
- `experiments/autoresearch/AR-019/logs/tests.log`
- `experiments/autoresearch/AR-019/candidate.pt`

## Metrics

- Returns: `[1.0, 1.0, -1.0, -1.0]`; population std `1.0`.
- Logical decisions/substeps: `340/380`.
- Ratios: mean `0.9999998807907104`, range
  `[0.9999895095825195, 1.0000028610229492]`.
- Loss/gradient: `0.08823533356189728` / `2.373613119125366`.

## Limitations and next control point

This micro-update is not a competitive result. It uses a detached recurrent
learner boundary, no value loss, serial collection, and K=4 returns. Do not
promote or tournament this candidate without a later explicit research gate.
