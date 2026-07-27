# Pokémon TCG local overhaul

Status: design baseline for review. This document formalizes the local product and its physical SQLite contract. It does not implement the schema.

## Authority and scope

The replay/dashboard/schema requirements in `.claude/HANDOFF_REPLAY_SCHEMA.md` and the decisions captured in the relevant session transcript are authoritative. The current code is evidence of the existing behavior, not the target design. The future on-chain tournament/compute-ledger idea is an extensibility consideration only; it is out of scope for this local overhaul.

## Product boundary

The local system is one product: catalog and deck workbench, model and experiment history, synchronous arena, normalized replay storage, ratings, daily remote intake, and Streamlit dashboard. SQLite is the local source of truth for this phase. Any future chain/index layer must be able to project the same domain without changing the domain identities.

## Decisions

1. Stable numeric IDs are primary keys. Names are editable metadata.
2. A model is a first-class identity. Model artifacts/revisions are immutable.
3. A deck family is renameable; each deck revision is immutable.
4. A submission is a concrete model artifact plus concrete deck revision and source/policy. Local reuse of the same pair continues its submission lineage; each new remote Kaggle send is a fresh remote submission lineage starting at 600.
5. Local, remote, submission, model, deck, and card ratings are distinct lineages. Initial Elo is 600. Deck Elo is its own rating, not a card-Elo average; card evidence and deck outcomes remain cross-queryable.
6. An experiment is a generic research container. Training is only one experiment subtype. Anamnese is append-only temporal observation data.
7. Replays are canonical relational data. JSON render output is an ingestion source; HTML is never persisted. The dashboard launches an external official visualizer after reconstructing its payload.
8. Every enum/domain has a domain-specific reference table and FK. No magic text/int enums, JSON fields, blobs, or generic enum table.
9. All writes and replay ingestion are idempotent.
10. Historical rows are append-only where they represent evidence. Corrections are new events/revisions, never destructive edits.

## Physical schema blueprint

### Reference domains

`ref_source`, `ref_model_status`, `ref_submission_status`, `ref_submission_source`, `ref_experiment_type`, `ref_experiment_status`, `ref_match_status`, `ref_match_result`, `ref_zone`, `ref_card_category`, `ref_card_stage`, `ref_select_type`, `ref_option_type`, `ref_action_type`, `ref_event_type`, `ref_pokemon_slot`, `ref_agent_role`, and other domains discovered from the official replay contract. Each table has a stable ID, canonical code, editable label, and uniqueness constraint on the code.

### Catalog and identities

- `models`: stable identity, editable name, lifecycle status, created-at and current metadata.
- `model_revisions`: immutable artifact identity, model FK, content digest, artifact metadata, provenance and created-at.
- `cards`: canonical card identity and catalog facts.
- `deck_families`: renameable deck identity and ownership/provenance.
- `deck_revisions`: immutable composition revision, deck-family FK, content digest and created-at.
- `deck_revision_cards`: deck-revision/card N:N rows with quantity and explicit position/role where required.
- `submissions`: immutable concrete model-revision/deck-revision pairing, source, lifecycle and external mapping.
- `submission_aliases` and `submission_events`: editable labels and append-only lifecycle history.

### Experiments and anamnese

- `experiments`: generic identity, type/status FKs, hypothesis, timestamps and owner.
- `experiment_models`, `experiment_deck_revisions`, `experiment_submissions`: explicit N:N links.
- `experiment_observations`: append-only timestamped anamnese entries with typed observation domain and provenance.
- `training_configs`: normalized immutable configuration facts.
- `training_runs`: experiment/model-revision linkage, config FK, execution metadata and outcome.
- `experiment_deck_tests`, `experiment_matchups`, `replay_analyses`: type-specific result tables.

### Tournament and matches

- `tournament_configs`: dashboard-editable persistent settings and feature flags represented by typed columns/FKs.
- `tournaments`: run identity, config revision, source and lifecycle.
- `tournament_participants`: submission FK, side/seed/status and eligibility snapshot.
- `matchups`: tournament FK, participant pair, scheduled round and result.
- `matches`: matchup FK, exact participants, source, seed, rules/config revision, result and timing.

### Normalized replay

- `replay_imports`: source identity, content digest, parser version and idempotency key.
- `match_steps`: match FK, sequence number, active player, turn/phase, decision context, status and reward.
- `step_options`: step FK, option sequence, option type FK, referenced card/target where applicable and selection result.
- `step_actions`: step FK, action type FK and normalized action facts.
- `step_events`: step FK, event type FK, actor/owner, source/target references, quantities and effects.
- `board_snapshots`: step/player state snapshots.
- `zone_snapshots`: snapshot/player/zone rows.
- `zone_cards`: zone snapshot/card/serial/position/visibility rows.
- `pokemon_on_field`: snapshot/slot/card/serial and state facts.
- `card_state_effects`: serial-scoped damage, energy, tools, status, evolution and counters.
- `card_movements`: serial-scoped from-zone/to-zone transitions with step/event FK.
- `match_card_usage`: match/card/player-side evidence used by rating aggregation.

Every replay child row carries the narrowest parent FK and a uniqueness constraint on its natural sequence identity. A replay is reconstructed by ordering relational rows; no serialized action, options, events, zone list or state blob is allowed.

### Ratings

- `rating_policies`: source/initial value/K-factor and effective configuration.
- `submission_ratings`: local or remote submission lineage, current value and counters.
- `model_ratings`, `deck_ratings`, `card_ratings`: independent aggregate lineages with source FK and evidence counters.
- `rating_events`: append-only per-match update facts, policy/config FK and before/after values.
- `rating_epochs`: explicit reset boundaries; a reset creates a new epoch rather than rewriting history.

### System and provenance

- `system_configs`: typed dashboard-editable operational values.
- `leaderboard_snapshots`, `remote_submissions`, `remote_submission_scores`: remote competitive context separated from local evidence.
- `operation_receipts`: idempotency key, operation type, actor, request digest, applied transaction and timestamp.

## Idempotency contract

Every command accepts a caller-supplied idempotency key or derives a stable natural key from immutable source identity. The database enforces uniqueness for those keys. A repeated command returns the original result and performs no second mutation.

Replay imports are keyed by source plus content digest and parser version. A repeated import returns the existing match/import identity. Child rows use parent identity plus sequence/serial keys. Rating updates are keyed by the match and rating policy/epoch, so retrying a completed match cannot apply Elo twice. Administrative commands are append-only events with unique command keys; repeated resets, revocations or promotions are no-ops after the first application.

All multi-row writes occur in one transaction. Foreign keys are enabled on every connection and checked in tests. Failed transactions leave no partial replay, participant, or rating state.

## Current-to-target removal

The current `matchups.replay_html` column is unused legacy and is not part of the target schema. Existing text agent fields, magic enum integers, serialized action/options/event data and mutable deck identity are replaced by the relations above. The database may be rebuilt; correctness of the target model takes precedence over preserving incompatible rows.

## Validation contract

- Schema creation succeeds from an empty database with FK enforcement enabled.
- Reference codes, natural keys and all FKs are tested.
- Re-importing the same replay produces no additional rows.
- Repeating every dashboard write and administrative command is idempotent.
- Reconstructed replay data reproduces the official visualizer flow in a separate window for both players.
- Local and remote ratings cannot cross-contaminate.
- A historical deck/model revision remains unchanged after a rename or new revision.
- A full replay can be reconstructed from relational rows without reading the original JSON or HTML.
- Dashboard reads use the target relations, not legacy text fields.

