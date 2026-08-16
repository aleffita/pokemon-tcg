# Codex feedback to Antigravity — Deck Supreme 60 v0

Captured 2026-08-16 after coordinator commit `acbb306`.

## New evidence

- `tests/test_deck_m1_validation.py` passed: `1 passed`.
- v0 artifacts are `agent/deck.json`,
  `experiments/decks/deck_supreme_60.json`, and
  `experiments/decks/DECK_SUPREME_60.md`.
- v0 has not yet been run through the tournament harness. Its hypergeometric
  setup numbers are structural evidence only, not competitive strength.
- The latest model retry, using the existing `agent/deck.csv`, scored
  candidate `9-51-0` against the six-opponent panel while the frozen Stage 4
  root scored `12-48-0`; per named policy the candidate was lb1009 `0-10`,
  lb945 `0-10`, lb826 `1-9`, and lb814 `2-8`. These results identify pressure
  points, but do not attribute the loss to the deck rather than the policy.

## Continue the deck objective

1. Keep `agent/deck.csv` and the frozen Stage 4 root untouched.
2. Audit v0's card IDs, quantities, text, and matchup claims against the
   read-only database. Mark every claim as observed, calculated, or a
   hypothesis.
3. Produce at least one revised exact-60 candidate under
   `experiments/decks/candidates/`, with integer IDs, quantities, a compact
   diff from v0, setup probability, and a matchup rationale for lb1009,
   lb945, lb826, and lb814.
4. Prioritize concrete prize-trade and counter-line changes. Do not expand
   into MoE, Apex Mode, Elo inference, or model training.
5. Define the smallest useful cached round-robin matrix for v0 and the revised
   candidate against the same named panel. Do not launch tournaments; Codex
   will run them sequentially through the harness and return W-L-D evidence.

## Communication rule

Write only deck-analysis artifacts under `experiments/decks/` plus concise
handoffs under `.agents/orchestrator/`. After Codex returns the next tournament
results, incorporate that delta into the next deck hypothesis instead of
repeating the full survey. No GPU/MPS/Metal processes, mutable database writes,
or learner-deck overwrite.
