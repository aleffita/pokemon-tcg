# Codex feedback to Antigravity — AR-027 retry

Captured 2026-08-16.

This is an iterative coordinator handoff. Treat `PROGRAM.md`,
`read-this-agent/*`, and the live repository as authoritative. Do not launch
training, MPS/Metal/PyTorch compute, or tournaments from the deck swarm.
SQLite access remains read-only and must not interfere with the Codex run.

## Closed experimental evidence

The authoritative retry was trained from the frozen Stage 4 root with:

- code commit: `509b948bee7fb1b43503a900257268d9bd14f848`;
- candidate: `experiments/autoresearch/AR-027-retry/candidate.pt`;
- dynamic sibling K: `[2, 2, 2, 2, 3, 4, 4, 2, 4, 2, 4, 2, 2, 2, 2, 2]`;
- 16 external-policy groups, 41 fibers, 3,031 logical decisions;
- branch policy/uniform mixture: `0.5`;
- continuation credit: enabled with discount `0.97`;
- collection: `100.359 decisions/s`;
- update: `16.869 s`, one optimizer step;
- ten groups were zero-variance.

The matched tournament surface is:

| Surface | W-L-D | Win rate |
| --- | ---: | ---: |
| Candidate vs frozen root, same deck, 30 | 13-17-0 | 43.3% |
| Candidate external panel, 60 | 9-51-0 | 15.0% |
| Frozen-root external panel, 60 | 12-48-0 | 20.0% |

Per external policy, candidate was `0-10` vs lb1009, `0-10` vs lb945,
`1-9` vs lb826, and `2-8` vs lb814. The candidate is rejected for promotion;
the frozen Stage 4 root remains the fallback.

The initial AR-027 output is superseded because its execution/output boundary
was not cleanly attributable relative to the optimization transition. Use
AR-027-retry for conclusions.

## Deck-swarm task

Continue the current competitive-deck objective, but make the next iteration
evidence-led:

1. Explain the candidate/root per-opponent gap using read-only SQLite, public
   deck lists, and the existing survey artifacts. Separate observed matchup
   facts from hypotheses.
2. Treat `experiments/decks/deck_supreme_60.json` and `agent/deck.json` as the
   current v0 deck artifact. Its structural validation passed
   (`tests/test_deck_m1_validation.py`: 1 passed).
3. Produce revised 60-card candidate(s) only under `experiments/decks/`, each
   with exact integer IDs, quantities, matchup rationale, setup probability,
   and a compact change log from v0. Do not silently overwrite the current
   learner deck or `agent/deck.csv`.
4. Prioritize counter-lines for lb1009/lb945/lb826/lb814 and prize-trade
   efficiency. Flag whether a candidate is a deck change, a model-policy
   limitation, or both.
5. For round-robin planning, define a small cached matrix of named deck
   candidates against the same opponent panel. Reuse completed JSON reports;
   do not launch it. The Codex coordinator will execute the tournament
   sequentially and record W/L/D by opponent.
6. After each new tournament result, update a new coordinator feedback file
   with the observed delta and the next focused deck hypothesis. Do not expand
   into MoE, Apex Mode, Elo inference, or architecture redesign.

Required handoff outputs for the next iteration:

- one or more candidate JSON decks under `experiments/decks/candidates/`;
- a short matchup diagnosis with source locators;
- a ranked next-deck hypothesis list with uncertainty;
- no GPU/MPS/Metal processes and no mutable tournament/database writes.

## Coordinator control

Codex owns model training, candidate generation, tournament execution, keep or
revert decisions, and final provenance. Antigravity owns tactical deck
analysis and candidate-deck files. Communicate only through these repository
artifacts and concise handoffs.
