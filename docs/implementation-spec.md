# Pokémon TCG local overhaul — implementation specification

This document is the execution-oriented companion to `TASK.md`. It resolves the design into concrete tables, service boundaries, commands, queries, UI surfaces and acceptance tests. An implementation agent must follow this contract instead of inventing a smaller model. The future blockchain/HashMath compute ledger is explicitly excluded.

## 1. Implementation order

Implement in this order and keep each phase runnable:

1. `results_db` replacement, reference seeds and FK enforcement.
2. Catalog/model/deck/submission services and idempotent command receipts.
3. Tournament/match persistence and rating policies.
4. Replay parser, normalized state and reconstruction adapter.
5. Experiment/training/anamnese services.
6. Arena integration and daily pipeline.
7. Dashboard queries and views, in the order listed below.
8. Official visualizer launch contract verification and adapter.
9. Full acceptance suite and removal of legacy readers/writers.

No phase may reintroduce JSON/blob/HTML storage to shorten implementation.

## 2. Storage conventions

- SQLite, one database file for the local phase.
- `PRAGMA foreign_keys = ON` immediately after every connection opens.
- Integer primary keys named `id`; all timestamps stored as UTC ISO text or an explicitly consistent numeric representation.
- All immutable artifacts carry `content_digest` and a uniqueness constraint.
- All tables have `created_at`; mutable identity tables additionally have `updated_at`.
- Historical tables are append-only; lifecycle changes are event rows.
- Every external or imported identifier has a source/provider column and a source-scoped uniqueness constraint.
- No table stores an encoded list, dictionary, JSON, HTML, or opaque observation blob.

## 3. Reference tables

Create one table for each domain below. Each table has `id INTEGER PRIMARY KEY`, `code TEXT NOT NULL UNIQUE`, `label TEXT NOT NULL`, `description TEXT`, `is_active INTEGER NOT NULL DEFAULT 1`, and `created_at`.

`ref_source`, `ref_model_status`, `ref_submission_status`, `ref_submission_source`, `ref_experiment_type`, `ref_experiment_status`, `ref_match_status`, `ref_match_result`, `ref_zone`, `ref_card_category`, `ref_card_stage`, `ref_select_type`, `ref_option_type`, `ref_action_type`, `ref_event_type`, `ref_pokemon_slot`, `ref_agent_role`, `ref_visibility`, `ref_observation_type`, `ref_rating_scope`, `ref_rating_source`.

Seed all codes required by current fixtures and the official replay contract. Never encode a code as a magic integer in application logic.

## 4. Concrete schema

The following is the minimum physical contract. Column names are normative; add a column only when a verified source field requires it.

### 4.1 Models and catalog

`models(id, name, status_id FK, description, created_at, updated_at)`

`model_revisions(id, model_id FK, revision_number, content_digest, artifact_uri, artifact_format, training_run_id nullable FK, provenance_text, created_at, UNIQUE(model_id, revision_number), UNIQUE(content_digest))`

`cards(id, external_card_id, name, category_id FK, stage_id FK, hp, energy_type_id nullable FK, weakness_type_id nullable FK, rule_text, source_id FK, UNIQUE(source_id, external_card_id))`

`deck_families(id, name, owner_model_id nullable FK, source_id FK, description, is_favorite, created_at, updated_at)`

`deck_revisions(id, deck_family_id FK, revision_number, content_digest, archetype_id nullable FK, source_id FK, created_at, UNIQUE(deck_family_id, revision_number), UNIQUE(content_digest))`

`deck_revision_cards(deck_revision_id FK, card_id FK, quantity, position nullable, role_id nullable FK, PRIMARY KEY(deck_revision_id, card_id, position))`

`submissions(id, model_revision_id FK, deck_revision_id FK, source_id FK, submission_source_id FK, status_id FK, external_submission_id nullable, name, created_at, updated_at, UNIQUE(source_id, model_revision_id, deck_revision_id, external_submission_id))`

`submission_aliases(id, submission_id FK, alias, is_primary, created_at, UNIQUE(submission_id, alias))`

`submission_events(id, submission_id FK, event_type_id FK, reason, operation_receipt_id FK, created_at)`

The local submission uniqueness key is `(source=local, model_revision_id, deck_revision_id)`. Remote sends are distinct rows even when model/deck are reused.

### 4.2 Experiments and training

`experiments(id, type_id FK, status_id FK, name, hypothesis, description, started_at, ended_at, created_at, updated_at)`

`experiment_models(experiment_id FK, model_revision_id FK, role_id FK, PRIMARY KEY(experiment_id, model_revision_id, role_id))`

`experiment_decks(experiment_id FK, deck_revision_id FK, role_id FK, PRIMARY KEY(experiment_id, deck_revision_id, role_id))`

`experiment_submissions(experiment_id FK, submission_id FK, role_id FK, PRIMARY KEY(experiment_id, submission_id, role_id))`

`experiment_observations(id, experiment_id FK, model_id nullable FK, observation_type_id FK, observation_text, value_numeric nullable, value_unit nullable, observed_at, created_at)`

`training_configs(id, content_digest UNIQUE, learning_rate, batch_size, epochs, sequence_length, accumulation_steps, dtype_id FK, config_version, created_at)`

`training_runs(id, experiment_id FK, model_revision_id FK, training_config_id FK, dataset_digest, checkpoint_digest, started_at, ended_at, status_id FK, metrics_summary_fields, created_at)`

`experiment_deck_tests(id, experiment_id FK, model_revision_id FK, deck_revision_id FK, outcome_id FK, created_at)`

`experiment_matchups(id, experiment_id FK, match_id FK, created_at)`

`replay_analyses(id, experiment_id FK, match_id FK, finding, created_at)`

`metrics_summary_fields` above means typed scalar metric columns, not a JSON field. Add metric rows if more than a fixed small set is required.

### 4.3 Tournament and match

`tournament_configs(id, name, games_per_opponent, sweep_decks_enabled, vs_self_enabled, x1_enabled, initial_elo, k_factor, source_id FK, version, created_at, updated_at, UNIQUE(name, version))`

`tournaments(id, config_id FK, experiment_id nullable FK, source_id FK, status_id FK, seed, started_at, ended_at, created_at)`

`tournament_participants(id, tournament_id FK, submission_id FK, side_id FK, seed, eligibility_status_id FK, created_at, UNIQUE(tournament_id, submission_id))`

`matchups(id, tournament_id FK, round_number, participant_a_id FK, participant_b_id FK, status_id FK, result_id nullable FK, scheduled_at, completed_at, created_at, UNIQUE(tournament_id, round_number, participant_a_id, participant_b_id))`

`matches(id, matchup_id FK, source_id FK, submission_a_id FK, submission_b_id FK, seed, rules_version, result_id FK, winner_submission_id nullable FK, started_at, ended_at, step_count, created_at, UNIQUE(source_id, seed, submission_a_id, submission_b_id, started_at))`

### 4.4 Replay

`replay_imports(id, match_id FK, source_locator, content_digest, parser_version, status_id FK, operation_receipt_id FK, created_at, UNIQUE(source_locator, content_digest, parser_version))`

`match_steps(id, match_id FK, step_number, active_player_id FK, turn_number, phase_code_id FK, select_type_id FK, select_context_code_id FK, status_id FK, reward_numeric, remaining_time_a, remaining_time_b, created_at, UNIQUE(match_id, step_number))`

`step_options(id, step_id FK, option_number, option_type_id FK, source_card_serial nullable, target_card_serial nullable, source_slot nullable, target_slot nullable, was_selected, created_at, UNIQUE(step_id, option_number))`

`step_actions(id, step_id FK, action_type_id FK, actor_player_id FK, submit_sequence, created_at, UNIQUE(step_id))`

`step_events(id, step_id FK, event_number, event_type_id FK, actor_player_id FK, source_zone_id nullable FK, target_zone_id nullable FK, source_card_serial nullable, target_card_serial nullable, value_numeric nullable, created_at, UNIQUE(step_id, event_number))`

`board_snapshots(id, step_id FK, player_id FK, active_card_serial nullable, hand_count, deck_count, discard_count, prize_count, status_code_id nullable FK, created_at, UNIQUE(step_id, player_id))`

`zone_snapshots(id, board_snapshot_id FK, zone_id FK, visibility_id FK, created_at, UNIQUE(board_snapshot_id, zone_id))`

`zone_cards(id, zone_snapshot_id FK, card_serial, card_id FK, position, visibility_id FK, created_at, UNIQUE(zone_snapshot_id, card_serial, position))`

`pokemon_on_field(id, board_snapshot_id FK, slot_id FK, slot_index, card_serial, card_id FK, hp, max_hp, energy_count, tool_count, preevolution_count, created_at, UNIQUE(board_snapshot_id, slot_id, slot_index))`

`card_state_effects(id, pokemon_on_field_id FK, effect_type_id FK, value_numeric, source_card_serial nullable, created_at)`

`card_movements(id, match_id FK, step_id FK, event_id FK, card_serial, card_id FK, from_zone_id FK, to_zone_id FK, from_position nullable, to_position nullable, player_id FK, created_at, UNIQUE(match_id, step_id, event_id, card_serial))`

`match_card_usage(match_id FK, card_id FK, player_id FK, quantity, first_step_id nullable FK, last_step_id nullable FK, PRIMARY KEY(match_id, card_id, player_id))`

If the official payload contains a field not represented above, add a typed relation or scalar column after fixture evidence. Do not put it in a JSON escape hatch.

### 4.5 Ratings and provenance

`rating_policies(id, scope_id FK, source_id FK, initial_value, k_factor, version, created_at, UNIQUE(scope_id, source_id, version))`

`rating_epochs(id, policy_id FK, name, reset_reason, started_at, ended_at nullable, created_at)`

`submission_ratings(id, submission_id FK, epoch_id FK, rating, games, wins, losses, draws, created_at, updated_at, UNIQUE(submission_id, epoch_id))`

`model_ratings(id, model_id FK, source_id FK, rating, games, wins, losses, draws, created_at, updated_at, UNIQUE(model_id, source_id))`

`deck_ratings(id, deck_revision_id FK, source_id FK, rating, games, wins, losses, draws, created_at, updated_at, UNIQUE(deck_revision_id, source_id))`

`card_ratings(id, card_id FK, source_id FK, rating, games, wins, losses, draws, created_at, updated_at, UNIQUE(card_id, source_id))`

`rating_events(id, match_id FK, policy_id FK, epoch_id FK, submission_rating_id nullable FK, model_rating_id nullable FK, deck_rating_id nullable FK, card_rating_id nullable FK, before_value, after_value, delta, created_at, UNIQUE(match_id, policy_id, epoch_id, rating_scope_key))`

`system_configs(id, config_code UNIQUE, value_type_id FK, value_text nullable, value_integer nullable, value_real nullable, value_boolean nullable, updated_at)`

`leaderboard_snapshots(id, competition_code, captured_at, source_locator, content_digest, created_at, UNIQUE(competition_code, captured_at, content_digest))`

`remote_submissions(id, leaderboard_snapshot_id FK, local_submission_id nullable FK, external_submission_id, score_numeric nullable, rank_numeric nullable, status_id FK, created_at, UNIQUE(leaderboard_snapshot_id, external_submission_id))`

`operation_receipts(id, idempotency_key UNIQUE, operation_code, request_digest, result_entity_type, result_entity_id, applied_at, created_at)`

## 5. Service contracts

Implement services behind the dashboard and scripts. Services return typed domain objects and operation receipts; UI code must not write SQL directly.

### CatalogService

`register_card`, `create_model`, `register_model_revision`, `create_deck_family`, `create_deck_revision`, `rename_deck_family`, `find_decks`, `compare_deck_revisions`.

### SubmissionService

`create_local_submission(model_revision_id, deck_revision_id)`, `record_remote_submission(local_origin, external_id)`, `set_submission_status`, `promote_submission`, `suspend_submission`, `list_eligible_submissions`.

### ExperimentService

`create_experiment`, `attach_model`, `attach_deck`, `attach_submission`, `append_observation`, `record_training_run`, `record_deck_test`, `record_matchup`, `record_replay_analysis`.

### TournamentService

`create_tournament`, `register_participants`, `schedule_sweep_decks`, `schedule_vs_self`, `schedule_x1`, `run_match`, `complete_match`, `get_tournament_state`.

### ReplayService

`import_replay`, `rebuild_visualizer_payload`, `list_local_replays`, `get_replay_metadata`, `get_replay_steps`. `import_replay` must be safe to call repeatedly with the same source/digest/parser version.

### RatingService

`get_or_create_local_submission_rating`, `create_remote_submission_rating`, `apply_match_result`, `reset_epoch`, `get_submission_rating`, `get_model_rating`, `get_deck_rating`, `get_card_rating`, `cross_rating_evidence`.

### ConfigService

`get_config`, `set_config`, `create_config_version`, `list_config_history`. Values must use typed columns and reference FKs.

## 6. Dashboard contract

The Streamlit dashboard must call services, never manipulate the database directly.

### Overview

Show current local arena state, active tournaments, submissions, local/remote filters, latest matches and rating summaries. Every card links to the relevant detail view.

### Cards

Show card rating by source, games/wins/losses, usage, deck contexts, match evidence and trend.

### Decks / Deck Builder

Search families and revisions; create a revision; edit composition by card rows; rename family; favorite; compare two revisions; show card evidence; choose submission context.

### Models / Agents

Onboard artifact/YAML; register revision; rename; favorite; search; show experiment history; show submissions and lifecycle; promote/suspend.

### Submissions

Show model revision, deck revision, source, local/remote lineage, external mapping, status, Elo and match history. Make the local-versus-remote distinction visible.

### Arena

Configure and launch synchronous sweep-decks, vs-self and X1; select eligible submissions; show run progress and outcomes; link every match to replay and rating evidence.

### Replays

List local replays; filter by tournament, experiment, submission, model, deck, result and date; show normalized metadata; provide separate Player 1 and Player 2 external-visualizer actions.

### Experiments

Create a generic experiment; attach entities; append anamnese observations; display subtype results and comparisons.

### Configuration

Edit tournament, rating and system settings through typed forms. Show audit/history and idempotent operation result.

## 7. Deterministic workflows

### Local match

1. Resolve persistent local submissions.
2. Verify eligibility and config version.
3. Create matchup/match with seed and exact participant revisions.
4. Run environment.
5. Parse and persist replay in one transaction.
6. Apply each rating policy once.
7. Mark match/tournament state and return receipt.

### Replay retry

1. Compute source digest.
2. Lookup `(source_locator, digest, parser_version)`.
3. Return existing result if present.
4. Otherwise insert import, all child rows and metadata atomically.

### Local rating reset

1. Create a new rating epoch with reason.
2. Initialize requested local submission rating at 600 in that epoch.
3. Preserve prior rating events and aggregates.
4. Return the same epoch for a repeated idempotency key.

### Remote send

1. Register the concrete external submission mapping.
2. Create a remote rating lineage at 600.
3. Never reuse local submission rating state for the remote row.

## 8. Verification gates

Before implementation is considered complete:

- schema creation from empty DB passes with FK enforcement;
- all reference seeds and constraints are tested;
- every service command has a repeat/idempotency test;
- replay import is lossless against fixtures and reconstructible without source JSON;
- official visualizer contract is verified empirically for both players;
- rating updates cannot double-apply;
- local and remote source filters are enforced in SQL and UI;
- renames preserve historical revisions and match links;
- experiments work without a training run;
- dashboard covers every view above and has failure/retry states;
- daily pipeline is re-runnable;
- legacy readers/writers and `replay_html` are removed;
- `uv run` audit/test commands and `git diff --check` pass.

## 9. Explicit non-goals

Do not implement the blockchain/HashMath ledger, Proof-of-Work tournament, decentralized matchmaking, on-chain identity or always-on home-lab service in this task. Keep stable content identities and service boundaries compatible with those future directions, but do not let them expand the local implementation scope.

## 10. Complete requirement-to-implementation matrix

This matrix is normative. Every row requires implementation, a focused test and a dashboard/query path where applicable.

| Requirement | Persistence | Service/flow | Dashboard surface | Acceptance evidence |
|---|---|---|---|---|
| Model identity is separate from model revision | `models`, `model_revisions` | `register_model_revision` | Models detail/history | Rename model; old revision links unchanged |
| Deck family can be renamed | `deck_families` | `rename_deck_family` | Deck search/editor | Rename does not alter revision digest |
| Deck composition is immutable | `deck_revisions`, `deck_revision_cards` | `create_deck_revision` | Builder creates revision, never edits old row | Historical match retains old composition |
| Same model with different decks is distinct submission | `submissions` | `create_local_submission` | Submission matrix | Two decks produce two lineages |
| Local model/deck Elo persists | `submission_ratings`, `rating_epochs` | `get_or_create_local_submission_rating` | Submission Elo | Second tournament continues rating |
| Remote send resets Elo | remote `submissions`, `submission_ratings` | `create_remote_submission_rating` | Kaggle context | New send starts at 600 |
| Model Elo is separate | `model_ratings` | aggregate update | Model analytics | Model aggregate does not overwrite submission |
| Deck Elo is composition outcome | `deck_ratings` | aggregate update | Deck analytics | Strong cards can occur in low-Elo deck |
| Card Elo accumulates local evidence | `card_ratings`, `match_card_usage` | card evidence update | Card analytics | Cross-submission accumulation |
| Local/remote source isolation | source FKs in every evidence family | scoped queries | global source filter | No cross-pool rating contamination |
| Experiment does not imply training | `experiments`, subtype tables | `create_experiment` | Experiments | Deck test exists without training run |
| Anamnese is temporal | `experiment_observations` | `append_observation` | Timeline | No mutable note replaces history |
| Training provenance is normalized | `training_configs`, `training_runs` | `record_training_run` | Training detail | Config survives disk change |
| Agent onboarding from existing artifacts | model/revision/provenance tables | `onboard_agent` | Models/Agents | Existing YAML reconciles without invented facts |
| Agent lifecycle is explicit | statuses + submission events | promote/suspend/retain | lifecycle controls | Suspended agent cannot enter new match |
| Public-agent deck discovery | cards/decks/source provenance | `find_decks` | Deck browser | Public deck searchable and selectable |
| Base-deck fallback | deck family/source relation | submission creation | onboarding/arena | Missing selected deck resolves documented fallback |
| Sweep-decks always available | tournament config/participants | `schedule_sweep_decks` | Arena | Ranked candidates scheduled |
| Vs-self always available | tournament participants/matchups | `schedule_vs_self` | Arena | Prior eligible submissions included |
| Local X1 exists | matchups | `schedule_x1` | Arena | Two selected submissions play |
| Tournament config is persistent | `tournament_configs`, `system_configs` | ConfigService | Config | CLI cannot silently override source of truth |
| Full replay options retained | `step_options` | parser | Replay inspection | All options reproduce decision context |
| Full replay actions retained | `step_actions` | parser | Replay metadata | No serialized action array |
| Full replay events retained | `step_events` | parser | Replay evidence | Source/target/effect relations present |
| Zones are reconstructible | snapshots/zones/zone cards | parser | Replay detail | Cards move through zones by serial |
| Field state is reconstructible | board/pokemon/effects | parser | Replay detail | HP, energy, tools/status/evolution retained |
| Official viewer opens externally | no HTML persistence | reconstruction adapter | Replay actions | Player 1/2 window opens |
| Remote replays are not local viewer corpus | source/provenance | pipeline scope | source filters | Dashboard local replay list excludes remote-only rows |
| Replay import is idempotent | `replay_imports`, natural keys | `import_replay` | pipeline status | Retry creates zero duplicate rows |
| All writes are idempotent | `operation_receipts` + unique keys | every command | all forms | Repeated request returns original result |
| FK integrity is enforced | all FK columns | DB connection setup | n/a | Orphan insert fails |
| Dashboard is integrated product surface | all query services | Streamlit service calls | all tabs | No direct legacy DB writes |
| Daily pipeline is repeatable | pipeline run/provenance tables | pipeline service | Config/status | Retry is safe and source-scoped |

## 11. File-by-file implementation map

The agent must inspect and then update these current modules; it must not leave a parallel schema that the application does not use.

### `rl/results_db.py`

Replace the current schema with the target creation/migration module. Add connection pragma, reference seeding, transaction helpers, operation receipts, repositories and typed row mappers. Remove `replay_html`, text agent fields and magic enum persistence. Keep SQL in repository/service modules, not dashboard code.

### `scripts/tournament.py`

Resolve model revisions and persistent submissions before a run. Read `tournament_configs`. Create participants, matchup and match rows before execution. Capture the complete JSON render source, send it through `ReplayService`, apply ratings exactly once, and return the persisted match identity. No HTML persistence.

### `scripts/dashboard.py`

Replace direct legacy queries with service/query objects. Implement all Overview, Cards, Decks/Builder, Models/Agents, Submissions, Arena, Replays, Experiments and Config surfaces defined in Section 8. Every mutation uses an idempotency key and displays source/lineage context.

### `scripts/deck_builder*` and catalog scripts

Expose deck-family/revision semantics, source provenance, public-agent discovery, comparison, search/favorite, fallback selection and immutable save. A “save” operation creates a revision; it does not update a composition in place.

### `scripts/daily_pipeline.py` and ingestion scripts

Create source-scoped pipeline runs, select the latest configured dataset, import supported remote aggregates, run local arena integration, deduplicate by digest and expose failures/retries. Never merge remote replay evidence into local viewer rows.

### `scripts/build_card_stats.py` and analytics queries

Read normalized `match_card_usage`, ratings and match outcomes. Produce independent card/deck/model/submission aggregates with explicit source and epoch filters. Do not infer deck Elo as card-Elo average.

### Replay/render adapter modules

Isolate official visualizer request reconstruction behind one adapter. Add fixture tests for the observed request method, route, player selection and payload. If the external contract changes, only this adapter should change.

### Tests and fixtures

Create schema fixtures, replay fixtures for every zone/action/effect class, tournament fixtures for self/sweep/X1, model/deck/submission fixtures, remote/local rating fixtures, dashboard query fixtures and idempotency retry fixtures. Every requirement row in Section 10 must point to at least one test.

## 12. Dashboard behavior details

### Global state and filters

Every analytical tab exposes source (`local`, `remote`, `all` where semantically safe), time range, model, model revision, deck family, deck revision, submission, tournament and experiment filters. The selected source must be visible in labels and export/query results.

### Empty, stale and invalid states

The dashboard must distinguish “no evidence yet”, “not eligible”, “source unavailable”, “stale snapshot” and “operation failed”. It must never fabricate a rating, deck or replay link to make a view look complete.

### Mutation feedback

Every create/update/import/reset action displays whether it was newly applied or recognized as an idempotent repeat, and links to the created/existing entity. Failed transactions display a recoverable error and do not imply success.

### Cross-analysis views

Provide explicit views for model × deck, submission × opponent, deck × card, card × source, rating over time, matchup matrix and experiment observation timeline. Preserve the distinction between evidence and derived aggregate.

## 13. Data migration/backfill decisions

The new database may be rebuilt. Before deleting or replacing the current file, export a read-only inventory and retain it as a local artifact outside the canonical DB. Backfill only facts with an unambiguous mapping: cards, deck compositions, tournament outcomes and replay fields that can be normalized without loss. Do not backfill `replay_html`, opaque serialized fields or ambiguous agent strings into target relations. Record every skipped field and reason.

## 14. Completion definition

The overhaul is complete only when the implementation, this specification, `TASK.md` and Wikifita agree on status; all Section 10 requirements have code and tests; the dashboard can operate the local product end-to-end; a replay can be opened externally from normalized rows; and an interrupted/retried pipeline produces the same state as a single run.
