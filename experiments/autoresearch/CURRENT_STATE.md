# Autoresearch current state

Captured: 2026-08-16

- Current best and fallback: frozen Stage 4 root
- Stage 4 root SHA-256: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`
- Active experiment: AR-021 grouped dynamic prospective sibling-fiber GRPO
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
- AR-020 controlled same-deck tournament: candidate-vs-root `11-19-0` in 30;
  candidate panel `9-21-0`; frozen-root panel `6-24-0`. Direct gate rejected
  promotion; root remains fallback. The earlier public-deck diagnostic was
  `2-8-0` and is not the artifact-deck gate.
- No RoPE-ND, package, submission, MoE, Parquet, or packed-data path was run.
- AR-021 implementation commit: `a536142`
- AR-021 candidate SHA-256: `52702295763ecee036e4f6bfaac6660df6ca5ec1cfca66efab5146ae8b292718`
- AR-021 grouped dynamic-K run: 4 bases, effective K `[4,4,4,2]`, 14 fibers,
  1,079 logical decisions, 69.245 decisions/s, one grouped optimizer step.
- AR-021 tournament: same-deck candidate-vs-root `22-8-0` in 30; candidate
  panel `8-22-0`; root panel `7-23-0`. Keep experimental only: root-relative
  gain is strong but external-panel strength remains weak.
- No RoPE-ND, package, submission, MoE, Parquet, or packed-data path was run.
- Next action: train grouped sibling fibers with external-opponent deck strata;
  preserve root fallback until the panel improves.
- AR-022 implementation commit: `50c1b6d`
- AR-022 candidate SHA-256: `0fb2fed2282298cb2e1e2f9cf14ca28b101735c5e839f303abba6f9d49da0c1a`
- AR-022 external-deck strata: four groups over agent, lb826, Lucario, and
  Dragapult decks; effective K `[2,2,2,2]`; 685 logical decisions; 3 of 4
  groups zero-variance, 220 credited actions.
- AR-022 tournament: same-deck candidate-vs-root `13-17-0`; candidate panel
  `7-23-0`; root panel `8-22-0`. Rejected for promotion; root remains fallback.
- Next action: improve opponent-policy realism or branch diversity before any
  further scale-up; do not promote on root-relative evidence alone.

Evidence: `experiment_ledger.jsonl`, `AR-019/report.md`, `AR-020/report.md`,
`AR-019/manifest.json`, `AR-019/metrics.json`, `AR-019/logs/tests.log`,
`AR-019/logs/run.log`, `AR-019/sample.manifest.json`,
`AR-019/trajectory_bundle.pt.gz`, `AR-019/candidate.pt`,
`STATE_CAPSULE_019.md`, `AR-020/manifest.json`, `AR-020/metrics.json`,
`AR-020/tournament_candidate_vs_root_10.json`,
`AR-020/tournament_candidate_panel_10.json`,
`AR-020/tournament_root_panel_10.json`,
`AR-020/tournament_candidate_vs_root_same_deck_30.json`,
`AR-020/tournament_candidate_artifact_deck_panel_10.json`,
`AR-020/tournament_root_artifact_deck_panel_10.json`, `STATE_CAPSULE_020.md`,
`AR-021/report.md`, `AR-021/manifest.json`, `AR-021/metrics.json`,
`AR-021/tournament_candidate_vs_root_same_deck_30.json`,
`AR-021/tournament_candidate_panel_10.json`,
`AR-021/tournament_root_panel_10.json`, `STATE_CAPSULE_021.md`,
`AR-022/report.md`, `AR-022/manifest.json`, `AR-022/metrics.json`,
`AR-022/tournament_candidate_vs_root_same_deck_30.json`,
`AR-022/tournament_candidate_panel_10.json`,
`AR-022/tournament_root_panel_10.json`, `STATE_CAPSULE_022.md`.
