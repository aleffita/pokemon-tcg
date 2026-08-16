# Autoresearch current state

Captured: 2026-08-16

- Current best and fallback: frozen Stage 4 root
- Stage 4 root SHA-256: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`
- Active experiment: AR-018 true recurrent two-sided self-play
- AR-018 code commits: `434d3f6`, repaired by `28c2b96`
- AR-018 status: corrected gate passed review; keep foundation
- Smoke: 4 games, 672 logical decisions, 759 records, 118.369 records/s,
  104.801 decisions/s; both agent sides covered
- Invariants: retry-safe independent recurrent lanes, reset/continuity true,
  composite logical behavior logprob, end-to-end ratio identity, legal-action
  checks, symmetric terminal returns recorded for both lanes
- No candidate weights were trained or promoted. No GRPO, RoPE-ND, tournament,
  package, Parquet, or packed-data path was used.
- Next action: trajectory-group GRPO micro-update using the accepted two-sided
  self-play buffer, then comparable tournament gate.

Evidence: `experiment_ledger.jsonl`, `AR-018/report.md`, `AR-018/manifest.json`,
`STATE_CAPSULE_018.md`.
