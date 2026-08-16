# Autoresearch current state

Captured: 2026-08-16

- Current best and fallback: frozen Stage 4 root
- Stage 4 root SHA-256: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`
- Active experiment: AR-019 trajectory-group GRPO micro-update
- AR-018 code commits: `434d3f6`, repaired by `28c2b96` and `4cfe5e8`
- AR-018 status: corrected gate passed review; keep foundation
- AR-019 code commit: `8b0c166`
- AR-019 evidence commit: `a97aed4`
- AR-019 status: K=4 policy-only micro-update complete; reviewer pending
- AR-019 candidate SHA-256: `2e592340522c697405676811f4d68c53c93129fd89b8f14b07427d4624ed77e7`
- AR-019 returns: `[+1.0, +1.0, -1.0, -1.0]`; 340 logical decisions,
  380 substeps; 46.277 decisions/s; update 2.152 s
- Smoke: 4 games, 672 logical decisions, 759 records, 118.369 records/s,
  104.801 decisions/s; both agent sides covered
- Invariants: retry-safe independent recurrent lanes, reset/continuity true,
  composite logical behavior logprob, end-to-end ratio identity, legal-action
  checks, symmetric terminal returns recorded for both lanes, agent-forfeit
  terminal notification
- No candidate weights were trained or promoted. No GRPO, RoPE-ND, tournament,
  package, Parquet, or packed-data path was used.
- Next action: reviewer audit of AR-019, then candidate-vs-root tournament if
  no P0 is found.

Evidence: `experiment_ledger.jsonl`, `AR-018/report.md`, `AR-018/manifest.json`,
`STATE_CAPSULE_018.md`.
