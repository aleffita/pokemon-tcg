# Autoresearch current state

Captured: 2026-08-16

- Current best and fallback: frozen Stage 4 root
- Stage 4 root SHA-256: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`
- Active experiment: AR-019 provenance-repaired trajectory-group GRPO micro-update
- AR-018 code commits: `434d3f6`, repaired by `28c2b96` and `4cfe5e8`
- AR-018 status: corrected gate passed review; keep foundation
- AR-019 implementation commit: `5dd6cfc`
- AR-019 regenerated evidence: working-tree artifacts produced from `5dd6cfc`; provenance evidence commit pending
- AR-019 status: K=4 policy-only micro-update complete; candidate provenance preflight and strict FP32 reload passed; no tournament; not promoted
- AR-019 candidate SHA-256: `3e23d7c3c191fa301baabc559dd9add82d6ffbec05ae8b8e6fd3327cdc17d183`
- AR-019 sample manifest: `experiments/autoresearch/AR-019/sample.manifest.json`
- AR-019 sample manifest file SHA-256: `c757a496457983bacf851bb31815f399d2e64829cc57a7613c304789e971a30a`
- AR-019 sample manifest content SHA-256: `f4b02aec85d4c1bad46194316920646d6f264f90ff205d6cad5d61232d600e08`
- AR-019 trajectory bundle: `experiments/autoresearch/AR-019/trajectory_bundle.pt.gz`
- AR-019 trajectory bundle SHA-256: `e86ede4ecd2dac75585452f08d52d43eaf64f6bd90848786618657566c3a18f3`
- AR-019 returns: `[+1.0, -1.0, +1.0, +1.0]`; 278 logical decisions,
  311 substeps; 44.355 decisions/s; update 1.775 s
- AR-019 candidate parameter delta: L2 `0.009198722439288157`; changed parameters `1293304`
- AR-019 rollout persistence: compact bounded provenance evidence only; no unbounded rollout buffer
- Invariants: retry-safe independent recurrent lanes, reset/continuity true,
  composite logical behavior logprob, end-to-end ratio identity, legal-action
  checks, candidate root/sample/bundle hashes linked, candidate preflight passed,
  Stage 4 root preserved
- GRPO is implemented and exercised in AR-019. No RoPE-ND, tournament, package,
  submission, promotion, MoE, Parquet, or packed-data path was run.
- Next action: commit the regenerated evidence, then candidate-vs-root tournament;
  after the tournament, start sibling-fiber GRPO.

Evidence: `experiment_ledger.jsonl`, `AR-019/report.md`,
`AR-019/manifest.json`, `AR-019/metrics.json`, `AR-019/logs/tests.log`,
`AR-019/logs/run.log`, `AR-019/sample.manifest.json`,
`AR-019/trajectory_bundle.pt.gz`, `AR-019/candidate.pt`,
`STATE_CAPSULE_019.md`.
