# Antigravity handoff: expanded deck pool before AR-037

## Start condition

Act now. Codex stopped the first AR-037 launch before training so the next
collection can ingest this delivery. Finish by atomically publishing the new
decks; do not wait for acknowledgement.

## Evidence

Read, in order:

1. `experiments/decks/swarm/results/AR-036-tournament.json`
2. `experiments/decks/swarm/results/AR-036.json`
3. `experiments/decks/swarm/results/AR-035-tournament.json`
4. `experiments/decks/swarm/analysis/AR035_TARGETED_ANALYSIS.json`
5. the current `experiments/decks/swarm/inbox/`

AR-036 ground truth from 96 tournament games:

- `002_v4_safeguard_fortress.json` was best overall at 5-11 (31.2%), including
  2-2 against Alakazam and 3-1 against Crustle;
- every tested deck remained 0-4 against Mega Lucario and 0-4 against Ivan;
- targeted decks 004 and 005 did not validate their intended hard counters;
- projected or synthetic win rates are hypotheses only. Use the tournament
  rows as the competitive evidence.

## Deliverable

Publish a materially expanded set of new legal 60-card integer-ID JSON decks
to `experiments/decks/swarm/inbox/` before Codex starts AR-037.

There is no eight-deck limit. Do not delete the six current decks during this
pass. Add as many distinct, evidence-grounded candidates as can be validated
quickly, with numeric prefixes continuing after 006. Prioritize:

1. multiple independently constructed anti-Lucario decks, not cosmetic edits;
2. multiple independently constructed anti-Ivan decks;
3. bridge variants preserving the observed 002 performance against Alakazam
   and Crustle while adding real counterplay to one hard matchup;
4. at least one structurally different archetype for inter-deck GRPO diversity.

Avoid repeating the failed assumptions behind 004/005 without a concrete
mechanical correction. Deduplicate by sorted 60-card content hash. Validate
every file with `tests/test_deck_m1_validation.py`, write to a temporary
non-JSON filename, and rename atomically only after it passes.

Write a compact machine-readable analysis to:

`experiments/decks/swarm/analysis/AR037_EXPANDED_POOL_ANALYSIS.json`

It must map each new filename to target opponent, tactical mechanism, changed
cards relative to its parent, and the evidence or rule interaction motivating
the change.

## Boundaries

Antigravity owns deck construction only. Do not start training or tournaments,
modify checkpoints/model code, use GPU/MPS, or write to `model/results.db`.
Codex will snapshot every unique inbox deck and begin AR-037 immediately after
this delivery appears.
