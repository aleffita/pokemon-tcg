# Autoresearch current state

Captured: 2026-08-16

- Current best and fallback: frozen Stage 4 root
- Stage 4 root SHA-256: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`
- Active experiment: AR-018 true recurrent two-sided self-play
- AR-018 code commit: `434d3f6`
- AR-018 evidence commit: `07e0260`
- AR-018 status: implementation gate passed; reviewer pending
- Smoke: 1 game, 158 logical decisions, 168 records, 49.405 decisions/s
- Invariants: independent recurrent lanes, reset/continuity true, composite
  logical behavior logprob, legal-action checks, symmetric terminal returns
- No candidate weights were trained or promoted. No GRPO, RoPE-ND, tournament,
  package, Parquet, or packed-data path was used.
- Next action: reviewer audit of AR-018, then trajectory-group GRPO micro-update
  if no P0 is found.

Evidence: `experiment_ledger.jsonl`, `AR-018/report.md`, `AR-018/manifest.json`,
`STATE_CAPSULE_018.md`.
