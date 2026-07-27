# GPT-5.5 implementation task — Pokémon TCG local overhaul

## Mission

Implement the complete local Pokémon TCG platform overhaul described by the approved design documents. Do not reduce the scope to an MVP and do not replace relational modeling with JSON, blobs, text enums or shortcuts.

The implementation target is the local synchronous product: normalized SQLite, full replay persistence and reconstruction, model/deck/submission identity, source-separated Elo, experiments/anamnese, local arena, daily pipeline and the complete Streamlit dashboard.

The blockchain/HashMath compute-ledger idea is future research and must not be implemented in this task.

## Required reading order

1. `.claude/HANDOFF_REPLAY_SCHEMA.md` — product requirements and authority boundary.
2. `TASK.md` — complete phased implementation plan and acceptance criteria.
3. `docs/implementation-spec.md` — normative tables, columns, services, UI, flows and requirement matrix.
4. `docs/local-overhaul-design.md` — decisions and idempotency contract.
5. Current code: `rl/results_db.py`, `scripts/tournament.py`, `scripts/dashboard.py`, deck-builder modules, replay ingestion and daily pipeline.

Treat the current code as evidence of existing behavior, not as the target schema. Preserve unrelated user changes.

## Non-negotiable rules

- Stable numeric IDs are primary keys; names are editable metadata.
- Models have stable identities and immutable revisions.
- Deck families are renameable; deck revisions are immutable.
- A submission is one concrete model revision plus one concrete deck revision.
- Local reuse of the same model/deck pair preserves its submission Elo lineage.
- Every new remote Kaggle send starts a separate Elo lineage at 600.
- Submission, model, deck and card ratings are separate, source-aware layers.
- Deck Elo is composition/outcome evidence, not a card-Elo average.
- Experiments do not imply retraining; anamnese is append-only temporal history.
- Every enum/domain has its own reference table and FK.
- Enable SQLite foreign keys on every connection.
- No JSON columns, JSON strings, blobs, serialized replay arrays or persisted HTML.
- All writes, imports, dashboard mutations and administrative commands are idempotent.
- Historical evidence is append-only; corrections create revisions/events.
- The dashboard is a first-class delivery stream, not a final cosmetic pass.

## Implementation order

### 1. Evidence inventory

Map all current database readers/writers, replay fields, tournament flows, deck sources, Elo updates and dashboard queries. Confirm the official visualizer request contract from an isolated generated replay fixture. Record fields that cannot be losslessly mapped.

### 2. Database foundation

Replace the current results schema with the physical contract in `docs/implementation-spec.md`: reference domains; models/revisions; cards; deck families/revisions/compositions; submissions/events; experiments/training/anamnese; tournament/config/match tables; normalized replay tables; rating policies/epochs/events; system and remote provenance; operation receipts.

Remove `matchups.replay_html`, text agent identity fields, magic enum integers and serialized replay payloads. Rebuild the database if necessary instead of preserving incompatible structure.

### 3. Idempotent services

Implement repository/service boundaries so Streamlit and scripts do not write SQL directly. Provide idempotent commands for catalog registration, model/deck revisions, submission creation, onboarding, lifecycle changes, tournament creation, replay import, rating updates, configuration and administrative operations. Use stable natural keys or caller idempotency keys, atomic transactions and repeat-safe results.

### 4. Replay ingestion

Parse `env.render(mode='json')` as an input source only. Persist every ordered step, decision, option, action, event, target, zone, card serial, movement, snapshot, Pokémon state, HP, energy, tool, status, evolution, counters, selection context, timing and reward required for faithful reconstruction. Never persist the source JSON or HTML.

Build the adapter that reconstructs the official visualizer payload on demand and opens it in a separate window for Player 1 or Player 2. Keep the exact external request details isolated behind this adapter and verified by fixtures.

### 5. Identity, experiments and ratings

Implement onboarding and lifecycle for models/agents; immutable model and deck revisions; deck builder discovery, creation, comparison, search and favorites; local/remote submission mapping; generic experiments and temporal observations; normalized training provenance; independent submission/model/deck/card Elo with local persistence, remote reset and explicit rating epochs.

### 6. Arena and pipeline

Run synchronous local tournaments using persistent dashboard configuration. Keep sweep-decks, vs-self and local X1 available. Persist participants, matchups, matches and replays transactionally. Apply each rating policy exactly once. Implement idempotent daily intake with explicit local/remote provenance and no remote contamination of local replay viewing or rating pools.

### 7. Complete dashboard

Implement service-backed views for Overview, Cards, Decks/Deck Builder, Models/Agents, Submissions, Arena, Replays, Experiments/Anamnese and Config. Include source/time/entity filters, empty/stale/error states, mutation receipts, cross-analysis (model × deck × cards × submission × matches), external replay launch and all lifecycle controls.

## Acceptance gates

- Empty database creation succeeds with FK enforcement.
- Every reference table, FK, unique key, index and check constraint is tested.
- Repeating every command/import/admin operation creates no duplicate mutation.
- A failed multi-row operation leaves no partial rows.
- The same replay can be imported repeatedly with no duplicate steps/events.
- A complete replay can be reconstructed without source JSON or HTML.
- The official visualizer opens externally for both players.
- Local model/deck submission Elo persists across tournaments.
- New remote submissions begin at 600 and remain source-separated.
- Deck/model/card aggregate ratings remain distinct and cross-queryable.
- Renaming identities does not alter historical revisions or matches.
- Experiments work without a training run and anamnese is append-only.
- Sweep-decks, vs-self and X1 execute from persistent configuration.
- Every dashboard surface in the specification is implemented and uses services.
- Daily pipeline reruns safely.
- Legacy schema readers/writers are removed.
- `docs/implementation-spec.md`, `TASK.md` and Wikifita status remain synchronized.
- Run the project’s relevant tests, `git diff --check`, and report any unresolved external visualizer contract issue rather than inventing a behavior.

## Delivery protocol

Work in vertical, reviewable phases. After each phase, report:

1. files changed;
2. requirements completed;
3. tests run and results;
4. unresolved blockers or decisions;
5. migration/backfill impact.

Do not claim the overhaul is complete until every acceptance gate passes. Do not implement blockchain, HashMath, Proof-of-Work, decentralized matchmaking or the future always-on home-lab service.
