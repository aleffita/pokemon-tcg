# Pokémon TCG local platform overhaul

Status: approved implementation task. This is the complete implementation contract for the local overhaul. It is intentionally exhaustive: do not convert it into an MVP by silently removing replay fidelity, relational modeling, dashboard scope, idempotency or evidence requirements.

## 0. Authority, scope and epistemic boundary

The user-approved design is documented in `docs/local-overhaul-design.md` and the Wikifita pages linked from it. The live repository is authoritative for current behavior. The handoff and transcript provide explicit product requirements and decisions; they do not prove that an implementation already exists.

Current scope is local and synchronous: normalized SQLite, local arena, replay reconstruction, Streamlit dashboard, model/deck/submission identity, experiments, Elo and source-separated remote context. A future on-chain/HashMath-inspired compute ledger is an extension boundary only; do not implement it in this task.

## 1. Non-negotiable engineering rules

- Use a relational model. Every enum/domain has its own reference table and FK.
- No JSON columns, JSON strings, blobs, serialized action/option/event fields or persisted replay HTML.
- Names are editable metadata, never primary keys.
- Model artifacts and deck compositions are immutable revisions; families/identities remain renameable.
- Local and remote data sources and ratings never contaminate one another.
- Every write, import, dashboard action and administrative command is idempotent.
- Historical evidence is append-only. Corrections create revisions/events, never destructive edits.
- Enable and test SQLite foreign keys on every connection.
- Preserve full replay information needed to reconstruct the official visualizer.
- Explain implementation decisions in English code/docs where project conventions require it; preserve Portuguese product semantics in user-facing material where appropriate.

## 2. Phase A — repository and contract inventory

1. Inventory every current reader/writer of `results.db`, replay JSON, tournament results, deck files, Elo and dashboard state.
2. Confirm current behavior against `rl/results_db.py`, `scripts/tournament.py`, `scripts/dashboard.py`, ingestion/pipeline scripts and deck builder code.
3. Record observed fields that are currently lost during replay ingestion.
4. Confirm the official visualizer request contract by isolated inspection of the generated environment output; do not assume route, method, payload shape or iframe behavior.
5. Add fixtures/manifests for representative local replay, multi-select action, zone movement, Pokémon evolution, status/effects, self-play, sweep-decks and X1 cases.

Deliverable: an evidence map from current source fields/flows to target tables and a list of intentionally removed legacy fields.

## 3. Phase B — physical SQLite schema

Implement a schema module and deterministic empty-database creation. Use stable integer primary keys and explicit reference tables for every domain enum.

### 3.1 Reference domains

Create domain-specific tables and seed canonical codes for sources, model/submission/experiment/match statuses, experiment types, match results, zones, card categories/stages, select/action/option/event types, Pokémon slots, agent roles, visibility and other values discovered in the visualizer contract.

### 3.2 Identity and catalog

Implement `models`, immutable `model_revisions`, `cards`, renameable `deck_families`, immutable `deck_revisions`, `deck_revision_cards`, and catalog provenance. Add content digests and uniqueness constraints for immutable artifacts.

### 3.3 Submissions and lifecycle

Implement concrete `submissions` linking one model revision to one deck revision, with source/policy, lifecycle, external Kaggle mapping, aliases and append-only submission events. Local reuse of the same model/deck pair must resolve to the same local lineage; a new remote Kaggle send must create a new remote lineage.

### 3.4 Experiments, training and anamnese

Implement generic `experiments`, explicit model/deck/submission links, append-only `experiment_observations`, normalized immutable `training_configs`, `training_runs`, `experiment_deck_tests`, `experiment_matchups` and `replay_analyses`. An experiment must not imply a retrain.

### 3.5 Tournament and match structure

Implement dashboard-editable `tournament_configs`, tournament runs, participants, matchups and matches. Match rows must retain exact source, participants, submissions, seed, rules/config revision, outcome, timing and experiment/tournament links.

### 3.6 Replay relations

Implement `replay_imports`, ordered `match_steps`, `step_options`, normalized `step_actions`, `step_events`, board/player snapshots, zone snapshots, zone cards, field Pokémon, serial-scoped effects and card movements, plus card usage evidence. Enforce parent FKs and natural sequence/serial uniqueness.

### 3.7 Ratings

Implement rating policies, epochs, submission ratings, model ratings, deck ratings, card ratings and append-only rating events. Initial value is 600. Submission Elo, model Elo, deck Elo and card Elo are distinct. Deck Elo is outcome-based composition evidence, not a card-Elo average. A reset creates a new epoch.

### 3.8 System/provenance

Implement typed `system_configs`, leaderboard snapshots, remote submissions/scores and `operation_receipts` for idempotency.

### 3.9 Legacy removal

The target schema must not contain `matchups.replay_html`, text agent identity fields, magic enum integers or serialized replay payloads. The database may be rebuilt; do not preserve incompatible legacy shape for convenience.

Deliverable: schema DDL/module, seeds, FK enforcement, indexes, constraints and schema-level tests.

## 4. Phase C — idempotent persistence and commands

1. Define command contracts and stable idempotency keys for catalog upserts, artifact registration, deck revisions, submissions, tournament creation, replay import, rating updates and administrative actions.
2. Make replay imports unique by source identity, content digest and parser version.
3. Make child replay rows unique by parent plus sequence/serial identity.
4. Make rating application unique by match, rating policy and epoch so retries cannot double-apply Elo.
5. Make onboarding, rename, favorite, suspension, promotion, reset and configuration commands repeat-safe.
6. Wrap every multi-row operation in an atomic transaction; verify rollback leaves no partial state.
7. Return the original result for a repeated command rather than silently creating a second object.

Deliverable: command/service layer and failure/retry tests.

## 5. Phase D — complete replay ingestion and reconstruction

1. Parse the environment JSON render output without persisting the raw JSON.
2. Persist every step, decision, option, action, event, source/target relation and reward.
3. Persist player state, turn/phase, action counters, selection context/effect/card/bounds, remaining time and all visualizer-required flags.
4. Persist zone membership and movement for hand, deck, discard, prize, stadium, active, bench and revealed/searchable areas.
5. Preserve card serial identity, ownership, positions, HP, energies, tools, pre-evolutions, damage and status effects.
6. Support complete reconstruction from relational rows after the original input file is unavailable.
7. Reconstruct the official visualizer payload on demand and open it externally for Player 1 or Player 2; never store HTML or embed the viewer in Streamlit.
8. Keep remote replay ingestion limited to the explicitly supported aggregate/competitive context unless a separate local-viewer source is defined.

Deliverable: parser, reconstruction adapter, fixtures and round-trip fidelity tests.

## 6. Phase E — model, deck and submission workflows

1. Onboard existing public/local models through the dashboard and reconcile portable YAML identity with database facts.
2. Register model revisions and provenance without using names as keys.
3. Integrate deck builder discovery, creation, immutable revisioning, rename/search/favorite, side-by-side comparison and fallback to an agent base deck.
4. Create local submissions from model revision plus deck revision and preserve the lineage across tournaments.
5. Create remote submission records separately and map them back to local origin without merging rating pools.
6. Implement agent lifecycle: arena candidates, active submissions, coliseu/retained agents, suspension and promotion.

Deliverable: domain services, dashboard flows and identity/submission tests.

## 7. Phase F — local arena, tournament and rating execution

1. Replace text agent/deck parameters with submission relations.
2. Read tournament behavior from persistent configuration, not CLI values as source of truth.
3. Keep sweep-decks and vs-self enabled in the local tournament concept and support local X1 matches.
4. Run synchronous local matches initially; keep boundaries suitable for a future always-on CPU-bound service.
5. Persist every match and replay idempotently.
6. Update submission Elo, model/deck/card aggregates and rating events exactly once per match/policy/epoch.
7. Provide explicit local-vs-remote filters in all aggregate queries.
8. Preserve the future ability to compare model, deck, card and submission strength without conflating levels.

Deliverable: tournament service, rating service, arena tests and representative local runs.

## 8. Phase G — dashboard overhaul

The dashboard is a first-class delivery stream, not a final polish step. Build it against the domain services and target schema as those contracts stabilize.

### 8.1 Overview

Show arena status, active tournaments, submissions, local/remote source filters, rating summaries and recent evidence with links into details.

### 8.2 Cards

Show card Elo, usage, win/loss evidence, deck contexts, source filters and relationships to decks and matches.

### 8.3 Decks and deck builder

Browse public-agent decks, create/save revisions, compare against base decks, search, rename, favorite, estimate evidence-backed strength and select a deck for an arena run.

### 8.4 Models/agents

Onboard, rename, search, favorite, inspect revisions/provenance, manage lifecycle, connect experiments and promote/suspend candidates.

### 8.5 Submissions

Display concrete model/deck pairing, source, lineage, status, Elo, Kaggle mapping and local origin.

### 8.6 Arena

Configure and launch synchronous sweep-decks, vs-self and X1 runs; select eligible submissions; show progress, outcomes and links to evidence.

### 8.7 Replays

List local arena replays; filter by model/deck/submission/tournament/result; inspect normalized metadata; launch the official visualizer externally with Player 1/Player 2 selection.

### 8.8 Experiments/anamnese

Create generic experiments, attach models/decks/submissions, record temporal observations, inspect training/deck-test/matchup/replay-analysis outcomes and compare runs.

### 8.9 Configuration

Edit typed tournament/system/rating settings with audit history and idempotent commands. Never make source-code CLI defaults the sole authority.

Deliverable: all dashboard tabs, query services, forms, filters, error states and UI tests against fixtures.

## 9. Phase H — daily pipeline and remote context

1. Select the latest replay dataset by configuration.
2. Ingest local/remote sources with explicit provenance and deduplication.
3. Run local arena and aggregate supported remote competitive evidence.
4. Keep remote leaderboard/submission context separate from local replay viewing and local Elo.
5. Surface pipeline run identity, inputs, outputs, failures and re-runnable status in the dashboard.

Deliverable: idempotent daily pipeline and source-separation tests.

## 10. Migration and compatibility

The existing database may be rebuilt. Before replacement, inventory readers/writers and create fixtures from the current catalog and representative replays. Provide an explicit one-shot import only for facts that can be mapped without lossy coercion. Do not migrate `replay_html`, magic enums or opaque serialized payloads into the new schema.

## 11. Validation and acceptance

- Empty database creation succeeds with FK enforcement enabled.
- Every reference code, FK, uniqueness rule, index and check constraint is tested.
- Repeating every write/import/admin action produces the original result and no duplicate rows.
- A failed multi-row operation leaves no partial state.
- Reimporting the same replay is a no-op.
- Full replay reconstruction works without original JSON or HTML.
- Official visualizer opens externally for both players from reconstructed data.
- Model/deck renames do not alter historical revisions or matches.
- Local Elo persists for the same model/deck pair; a new remote send starts at 600.
- Deck and card Elo remain distinct and cross-queryable.
- Local and remote data cannot contaminate each other's ratings or replay views.
- Experiments support non-training work and append-only anamnese.
- Dashboard covers every required workflow and reads target relations only.
- Tournament modes, settings and pipeline rerun safely.
- Documentation and Wikifita remain synchronized with implementation status.

## 12. Deferred ideas

The HashMath/Bitcoin-inspired compute ledger, on-chain tournament state, cryptographic agent identities, deterministic on-chain matchmaking and useful-work mining are future research hypotheses. They may inform stable content identities and service boundaries, but they are not part of this local implementation task.
