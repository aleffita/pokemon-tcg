# Codex feedback to Antigravity — AR-028 deck panel

Captured 2026-08-16 after the first explicit-deck tournament screen.

## Tournament evidence

The v0 Deck Supreme 60 completed the six-opponent, 10-game panel:

| Deck | random | first | lb1009 | lb945 | lb826 | lb814 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0 | 7-3 | 2-8 | 0-10 | 0-10 | 0-10 | 4-6 | 13-47-0 |

The prior frozen-root panel was `12-48-0`; v0's one-win aggregate delta is
not promotion evidence. The Lucario bottleneck remains `0-20`, and the
Alakazam matchup was `0-10` in this screen.

Candidate screening at five games per opponent:

| Deck | random | first | lb1009 | lb945 | lb826 | lb814 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 anti-Lucario tempo | 2-3 | 0-5 | 0-5 | 1-4 | 0-5 | 0-5 | 3-27-0 |
| v2 anti-control lock | 0-5 | 2-3 | 0-5 | 0-5 | 1-4 | 3-2 | 6-24-0 |

Reports:

- `experiments/decks/AR-028-v0-panel-10.json`
- `experiments/decks/AR-028-deck-v1-panel-5.json`
- `experiments/decks/AR-028-deck-v2-panel-5.json`

v1 is rejected: it did not move either Lucario matchup and collapsed against
Crustle. v2 is retained as a matchup-specific diagnostic only: it improved
Alakazam and Crustle in this small screen but remained `0-10` against the two
Lucario policies and lost overall to v0.

## Artifact correction

The emitted `deck_v2_anti_control_lock.json` contained 59 cards. Codex added
one Basic Grass Energy (ID 1) to restore the exact-60 contract while preserving
the stated control hypothesis. The candidate now passes the parser.

An additional `deck_v3_apex_sovereign.json` appeared with 58 cards and no
complete handoff rationale. Do not treat it as evaluable. Repair it to an
exact 60-card list with a documented delta and source-backed rationale before
Codex considers it for a tournament.

## Next handoff

Continue from the measured bottlenecks, not from aggregate setup probability:

1. produce a valid v3 or a focused v2 revision that changes the Lucario line;
2. keep candidate files under `experiments/decks/candidates/` and do not touch
   `agent/deck.csv`;
3. include exact IDs, quantities, 60-card count, and matchup hypotheses;
4. let Codex run the next sequential panel and return W-L-D deltas.

No GPU/MPS/Metal, training, tournament, or database writes from the swarm.
