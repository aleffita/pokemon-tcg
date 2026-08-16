# Autoresearch current state

Captured: 2026-08-16

- Current best and fallback: frozen Stage 4 root
- Stage 4 root SHA-256: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`
- Active experiment: AR-020 dynamic prospective sibling-fiber GRPO micro-update
- AR-018 code commits: `434d3f6`, repaired by `28c2b96` and `4cfe5e8`
- AR-018 status: corrected gate passed review; keep foundation
- AR-019 implementation commit: `5dd6cfc`
- AR-019 regenerated evidence: working-tree artifacts produced from `5dd6cfc`; provenance evidence commit pending
- AR-019 status: K=4 policy-only micro-update complete; provenance preflight and strict FP32 reload passed; tournament complete; not promoted
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
- GRPO is implemented and exercised in AR-019. Candidate-vs-root: `6-4-0`
  (60%, n=10). Candidate panel: `9-21-0` (30%, n=30); frozen root panel:
  `9-21-0` (30%). Candidate remains experimental; root remains fallback.
- No RoPE-ND, package, submission, MoE, Parquet, or packed-data path was run.
- AR-020 implementation commit: `a171dc8`
- AR-020 candidate SHA-256: `89a70d4eddb3c856d7c4a4e1ad520e2d23bc7230c76b4c10904c45970eeb8637`
- AR-020 dynamic effective K: `[4, 3]` across two deck-stratified bases;
  569 logical decisions, 658 substeps; 64.372 decisions/s; 4.026 s update
- AR-020 tournament: candidate-vs-root `2-8-0`; candidate panel `8-22-0`;
  frozen-root panel `3-27-0`. Direct gate rejected promotion; root remains
  fallback. Candidate/root packaged decks differed, so panel delta is
  directional rather than a controlled same-deck estimate.
- No RoPE-ND, package, submission, MoE, Parquet, or packed-data path was run.
- Next action: start the next bounded research hypothesis with dynamic-K and
  multi-deck tournament evidence; preserve root fallback.

Evidence: `experiment_ledger.jsonl`, `AR-019/report.md`, `AR-020/report.md`,
`AR-019/manifest.json`, `AR-019/metrics.json`, `AR-019/logs/tests.log`,
`AR-019/logs/run.log`, `AR-019/sample.manifest.json`,
`AR-019/trajectory_bundle.pt.gz`, `AR-019/candidate.pt`,
`STATE_CAPSULE_019.md`, `AR-020/manifest.json`, `AR-020/metrics.json`,
`AR-020/tournament_candidate_vs_root_10.json`,
`AR-020/tournament_candidate_panel_10.json`,
`AR-020/tournament_root_panel_10.json`, `STATE_CAPSULE_020.md`.
