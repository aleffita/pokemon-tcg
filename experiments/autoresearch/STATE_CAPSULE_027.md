# State Capsule 027 - superseded grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-16T19:02:25.577757+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `16` exact recurrent sibling groups
  and `39` fibers with effective K
  `[4, 4, 2, 2, 2, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2]`.
- One grouped FP32 policy-only update applied independent group-relative
  terminal credit through future continuation with discount
  `0.97`.
- Candidate: `e098fa1dc6747d63e85e25f4cb908001128145fe6e4b05c9aed9fa73470a5fef`; preflight passed.
- This output is superseded as a diagnostic because its execution boundary was
  not cleanly attributable relative to the optimization transition. No
  promotion, RoPE-ND, MoE, or historical ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-027/report.md`
- `experiments/autoresearch/AR-027/manifest.json`
- `experiments/autoresearch/AR-027/metrics.json`
- `experiments/autoresearch/AR-027/sample.manifest.json`
- `experiments/autoresearch/AR-027/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-027/candidate.pt`

## Metrics

- Collection: `22.888142` s,
  `106.03744089183888` decisions/s.
- Update: `9.161688708001748` s; one optimizer step.
- Credited logical actions: `559`.
- Parameter L2 delta: `0.009215305122152694`;
  gradient norm `0.6900275349617004`.

## Next control point

Do not run or interpret a gate from this output directory. Use `AR-027-retry`
as the authoritative replacement and keep the root fallback.
