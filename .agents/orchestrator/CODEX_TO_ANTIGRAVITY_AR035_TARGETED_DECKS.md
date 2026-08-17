# Antigravity handoff: targeted exploitation after AR-034

## Objective

Produce a small targeted deck expansion for the final continuous-training
lineage. Codex owns architecture, training, tournaments, and checkpoint choice.
Antigravity owns deck analysis and deck proposals only.

## Evidence to read

1. `experiments/decks/swarm/results/AR-034-tournament.json`
2. `experiments/decks/swarm/results/AR-033-tournament.json`
3. `experiments/decks/swarm/results/AR-032-parent-control.json`
4. `experiments/decks/swarm/results/AR-035-tournament.json` when it appears
5. Existing live candidates in `experiments/decks/swarm/inbox/`

Observed priority: the trained agent remains at 0% in the current panel against
`lb1009_mega_lucario_ex_islet` and `lb945_multiply_ivan`; it has begun winning
against `lb826_alakazam_seok` and is strongest against `lb814_crustle_emre`.

## Deliverable

Write 2 to 4 new legal 60-card integer-ID JSON decks into:

`experiments/decks/swarm/inbox/`

Use atomic temporary-file rename. Keep the total inbox at no more than 8 unique
content hashes. Preserve the current incumbents; do not delete or replace them
during this manual pass.

Target deck roles:

1. At least one anti-Lucario candidate.
2. At least one anti-Ivan candidate.
3. Optionally one robust bridge candidate that preserves the observed
   Crustle/Alakazam performance while addressing either hard matchup.

Do not train a model, use GPU, modify checkpoints, edit tournament code, write
to `model/results.db`, or wait for Codex. Finish after publishing the deck files
and a compact analysis JSON beside them or under
`experiments/decks/swarm/analysis/`.

Codex automatically consumes unique inbox decks on the next RL collection.
