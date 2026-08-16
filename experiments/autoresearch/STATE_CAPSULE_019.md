# State Capsule 019 - first trajectory-group GRPO micro-update

Captured 2026-08-16T17:02:10.629259+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-019 collected K=4 current-vs-current true recurrent trajectories in
  memory and applied one FP32 policy-only trajectory-group GRPO update.
- Candidate: `3e23d7c3c191fa301baabc559dd9add82d6ffbec05ae8b8e6fd3327cdc17d183` (5694416 bytes).
- Candidate parameter L2 delta: `0.009198722439288157`; changed parameters:
  `1293304`.
- Compact provenance manifest: `experiments/autoresearch/AR-019/sample.manifest.json`;
  SHA-256 `c757a496457983bacf851bb31815f399d2e64829cc57a7613c304789e971a30a`.
- Compact trajectory bundle: `experiments/autoresearch/AR-019/trajectory_bundle.pt.gz`;
  SHA-256 `e86ede4ecd2dac75585452f08d52d43eaf64f6bd90848786618657566c3a18f3`.
- Candidate provenance preflight passed: `True`.
- Candidate-vs-root gate: candidate `6-4-0` (60%) over 10 games.
- Candidate panel: `9-21-0` (30%) over lb826, random, and first; frozen root
  panel on the same surface was also `9-21-0` (30%). Candidate was not promoted.
- No package, submission, MoE, RoPE-ND, or historical ETL/Parquet/packed-data
  work was run.

## Evidence

- `experiments/autoresearch/AR-019/report.md`
- `experiments/autoresearch/AR-019/manifest.json`
- `experiments/autoresearch/AR-019/metrics.json`
- `experiments/autoresearch/AR-019/logs/tests.log`
- `experiments/autoresearch/AR-019/sample.manifest.json`
- `experiments/autoresearch/AR-019/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-019/tournament_candidate_vs_root_10.json`
- `experiments/autoresearch/AR-019/tournament_candidate_panel_10.json`
- `experiments/autoresearch/AR-019/tournament_root_panel_10.json`
- `experiments/autoresearch/AR-019/candidate.pt`

## Metrics

- Returns: `[1.0, -1.0, 1.0, 1.0]`; population std `0.8660253882408142`.
- Logical decisions/substeps: `278/311`.
- Ratios: mean `1.0`, range
  `[0.9999923706054688, 1.0000028610229492]`.
- Loss/gradient: `0.1619904488325119` / `2.5191781520843506`.

## Limitations and next control point

This micro-update is not a competitive result. It uses a detached recurrent
learner boundary, no value loss, serial collection, and K=4 returns. The
direct root gate was positive but the opponent panel tied the frozen root, so
keep the candidate experimental and use sibling-fiber GRPO as the next probe.
