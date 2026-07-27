# Pokémon TCG MLX — Complete Product and Architecture Handoff

## 1. Purpose and authority

This document replaces the earlier handoff as the implementation authority for the next phase of the Pokémon TCG project. It was reconstructed from the user's original messages in the exported session, beginning with the replay/dashboard discussion, plus the verified project findings recorded during that session.

When a generic design suggestion conflicts with an explicit requirement below, the explicit requirement wins. Do not reinterpret it as an MVP, a later phase, or an optional improvement.

### Non-negotiable operating principles

- Investigate before deciding. Use targeted code inspection and isolated checks to establish facts, then distinguish facts from proposals.
- Do not simplify across domain boundaries. Simplicity comes from proper separation of responsibilities and explicit relations.
- Do not change strategy or silently reduce scope. If an implementation would need a different strategy, stop and ask.
- The database can be rebuilt. Correctness of the new architecture is more important than migrating the old SQLite contents.
- Explain in Portuguese; keep implementation and code in English.

## 2. Product outcome

Build a unified local research platform for Pokémon TCG agents. The Streamlit dashboard is the operating surface for local arena results, replay inspection, deck work, agents, experiments, configurations, and later Kaggle submission/leaderboard context.

This is one integrated system:

```
daily replay intake + local arena
             ↓
normalized SQLite source of truth
             ↓
Streamlit dashboard: analysis, replay, decks, agents, experiments, configuration
             ↓
local research decisions → optional Kaggle submission and competitive feedback
```

There is no separate “later replay MVP.” Local replay viewing is a current requirement of the dashboard and must be supported by the same relational SQLite system.

## 3. Replay system and official visualizer

### Scope

- The dashboard should list and inspect **local arena replays**. It should not become a viewer for all remote Kaggle replays.
- Remote replay data may still be ingested for aggregate statistics, card/deck Elo and competition context.
- Replays need enough normalized state to rebuild a game faithfully and to invoke the official competition visualizer.

### Required data path

1. A local match runs through the environment.
2. Its replay JSON / `env.render(mode='json')` data is parsed at save time.
3. Every needed fact is persisted in normalized relational tables, not as serialized replay content.
4. The application reconstructs the official replay payload from those tables when the user opens a local replay.
5. The first integration test opens the remotely hosted official visualizer for the selected player. Only after this isolated flow works should embedding be evaluated.

### Explicitly forbidden storage shortcuts

- No replay JSON stored in SQLite.
- No JSON string fields, JSON blobs, generic observation blobs, or serialized HTML.
- No `env.render(mode='html')` output stored as a file or database content.
- No external replay file paths as the internal source of truth.

`env.render(mode='json')` is an extraction source, not a database payload. The local HTML result is useful only as an investigation aid to understand the official visualizer handoff; it is not the product viewer.

### Official visualizer facts to preserve

- The environment HTML contains buttons to choose Player 1 or Player 2.
- Those buttons submit reconstructed replay data to the externally hosted official visualizer.
- The intended experience is the official visualizer, with its card visuals, board and animation—not a substitute local HTML viewer.
- The exact request method, target route, player index and payload contract must be verified from the generated result and isolated before dashboard integration. Do not assume an iframe, redirect or request format without observing it.

### Replay persistence domains

The schema must capture at least:

- Match identity, source, participants, selected sides, outcome, timing and tournament/experiment links.
- Per-player, per-step state: action, status, reward, turn, first-player information and decision context.
- Selection constraints and options, modeled as rows rather than opaque arrays.
- Event logs and their card, player, source-zone, target-zone and target-card relationships.
- Board snapshots; Pokémon in active/bench slots; HP, energy, tools, pre-evolution and serial identity.
- Zone card membership for hand, deck, discard, prize, stadium and searchable/revealed areas, with card serial and player ownership.
- All fields needed by the official replay format, including action counters, supporter/stadium/energy/retreat flags, looking state, select context/effect/card/bounds and remaining time.

## 4. Relational database standards

### Required modeling rule

Every enumerated domain must have its own reference table and foreign key. A text enum, integer magic value, comment that explains an integer, or global “generic enum” abstraction is not acceptable.

Examples include, but are not limited to:

- zones
- match statuses
- data sources
- deck sources
- replay event types
- option types
- select types
- card categories and stages
- Pokémon slots
- agent statuses
- experiment statuses and types
- submission outcomes

Relations must represent the actual domain. Use domain-specific N:N tables where facts have independent identity, rather than collapsing unrelated things into a generic attribute table.

### Core schema families

| Family | Responsibilities |
|---|---|
| Reference domains | Every enum/value domain and seed values |
| Catalog | Cards, decks, deck cards, archetypes and source provenance |
| Ratings | Agent–deck snapshots, deck Elo and card Elo split by source and evidence |
| Tournament | Configurations, tournament runs, matchups and matches |
| Replay | Steps, options, events, snapshots, field Pokémon, zone cards and card usage |
| Agents | Agent identity, version/configuration history, notes, status and local filesystem onboarding association |
| Experiments | Generic experiment identity plus type-specific result tables and continuous observations |
| Training | Normalized train configurations and training runs |
| Competitive context | Leaderboard snapshots, known teams, submissions and competition episodes |
| System configuration | Dashboard-editable operational defaults and limits |

### Rebuild policy

The existing database does not require a migration. Delete/recreate it, seed reference values and repopulate from the current catalog, local arena runs and daily remote intake. Any implementation plan must identify every database reader and writer before changing the schema.

## 5. Decks, cards and Elo

### Deck workbench

Integrate the existing deck builder into the dashboard so a user can:

- Explore decks of public agents.
- Create and save new decks for specific agents.
- Load a deck from a listing into the builder.
- Compare a working deck side-by-side with a base deck.
- Estimate deck strength using card-level evidence.
- Choose a deck for a tournament/arena run.
- Rename and find decks from the dashboard.

Fallback behavior: when a selected or ranked deck is unavailable, the agent can use its base deck.

### Ratings model

- Initial Elo is **600** in every new relevant rating lineage.
- Ratings are not a single property of an agent. The meaningful local competitive entity is the agent–deck combination.
- Running the same frozen agent with a different deck creates a fresh local submission-like lineage.
- Reusing the same agent and deck across independent tournament runs continues that lineage.
- A tournament can run one agent across several decks, preserving independent ratings for those combinations.
- Global deck Elo and card Elo are derived from the population of local agent–deck evidence; they are not simplistic labels attached in isolation.
- Local and remote evidence are distinct sources. Do not let remote games contaminate local games, win rates or Elo displays.

### Arena selection

Sweep-decks and vs-self are always enabled in the local tournament concept:

- Sweep-decks tests the agent with top-ranked/ranked candidate decks.
- Vs-self includes prior eligible submissions/versions.
- The arena also supports local X1 matches between agents.
- Later, this becomes the basis for a continuously running, CPU-bound local tournament that informs the limited daily Kaggle submission budget.

## 6. Agents, lifecycle and onboarding

### Lifecycle

```
existing/public agent → dashboard onboarding → arena
new trained agent → experiment recorded → arena
arena candidate → promotion → submissions
arena failure → coliseu
coliseu agent → continues to play; may regain relevance as meta changes
```

- `public_agents/arena/`: candidates being evaluated locally before submission.
- `public_agents/submissions/`: agents that have been sent to Kaggle.
- `public_agents/coliseu/`: prior losers retained for ongoing competitive diversity.
- Existing public agents must be discoverable and onboarded from the dashboard, not treated as disconnected folders.

### Identity and YAML

An agent receives a YAML identity record that travels with it. It describes agent identity/version and relevant linked facts such as deck, archetype and training provenance.

Do not use agent name as a primary key. The database owns stable IDs and relations. The YAML is a portable identity/config artifact whose data is reconciled with indexed database facts.

The dashboard onboarding flow must help a user attach existing agents to the database. It must not ask the user to invent deck, archetype or training facts that the project can derive from cataloged decks, known files, experiment records or discovered provenance.

## 7. Experiments and training history

### Experiment definition

An experiment is a generic research container, not a synonym for “new model training.” Examples:

- A new training run.
- An existing agent tested with a new deck.
- An existing agent tested with an opponent’s deck.
- Analysis/replay-based testing of a submitted agent under different deck choices.
- Arena evaluation of multiple agent/deck combinations.

The base experiment row stores only universal identity and lifecycle facts. It links to separate type-specific tables for training runs, deck tests, matchups and replay analysis.

### Continuous anamnese

Observations are not a one-time `note` column. Agents and experiments need append-only, time-stamped observation rows throughout their lifecycle. This supports evolving hypotheses, documentation, later analysis and comparative research.

### Training configuration

The training JSON on disk is not durable provenance. Normalize and internalize the train configuration used by an agent/run so it remains available after the disk config changes. This supports future comparison of controlled parameter variations and p99-style performance analysis. Do not serialize the whole configuration into a generic text field.

## 8. Tournament, configuration and dashboard

### Configuration source of truth

Tournament behavior must read persistent configuration tables, not rely on CLI parameters as the source of truth. The dashboard should edit these configuration values, including games per opponent, deck sweep count, Elo values, initial ratings and feature flags/limits.

The dashboard can start a tournament manually in the current phase. The future always-on local tournament must be anticipated by the same schema, not used as an excuse to defer the basal model.

### Dashboard responsibilities

- Local arena overview, results and ratings.
- Local replay listing and launch to the official visualizer.
- Deck browsing, editing, comparison and selection.
- Agent discovery, onboarding, rename, favorite and search.
- Experiment creation, lifecycle, observations and comparisons.
- X1 operations and promotion decisions.
- Tournament and system configuration.
- Kaggle leaderboard/submission context associated with known local agents when available.

Avoid a sidebar-dependent or path-dependent design where a relational view is appropriate. The dashboard must make local and remote evidence visibly distinct.

## 9. Kaggle and daily pipeline

### Kaggle integration

The first Kaggle submission has not been made. This does not block schema design, contract investigation or implementation. Verify the API/CLI/package contracts for leaderboard, teams, submissions and episodes without treating a first submission as a prerequisite.

When competitive data is available, the dashboard should relate remote performance to the corresponding local agent/version where identity is known. This makes it possible to distinguish a deck issue from an architectural/training issue.

### Daily flow

1. Obtain the latest official replay data.
2. Process remote data into normalized database evidence for cards, decks and competitive context.
3. Run local arena evaluation and store normalized match/replay data.
4. Recompute or update appropriate source-separated ratings.
5. Present the results in the dashboard.

Avoid permanent dataset paths inside the relational design. Paths are acceptable only where an external platform contract requires them; otherwise internal database relations carry the workflow state.

## 10. Refactor surface and validation

Before implementation, audit all interactions with SQLite. The confirmed high-risk surface includes:

- `rl/results_db.py`: schema, seeds, writes and rating computation.
- `scripts/tournament.py`: local matches, replay extraction and persistence.
- `scripts/build_card_stats.py`: remote replay ingestion and aggregate card/deck evidence.
- `scripts/populate_cards.py` and `scripts/populate_decks.py`: catalog sources.
- `scripts/dashboard.py`: all readers and dashboard workflows.
- `scripts/daily_pipeline.py`: simplify into a database-centric daily flow.
- `scripts/submit.py`: agent promotion/submission lifecycle integration.
- Existing deck builder and public agent directories: discovery and onboarding.

### Required acceptance checks

1. Fresh database rebuild, reference seed validation and catalog population.
2. Local tournament smoke run that persists local-only rating evidence correctly.
3. Remote update that does not alter local rating counts or views.
4. Replay extraction test: reconstruct a local replay solely from relational rows.
5. Official visualizer contract test for both player choices.
6. Dashboard test covering local replay launch, local/remote rating separation and deck selection.
7. Agent onboarding test for an existing public agent and a new locally trained agent.
8. Experiment test for a non-training experiment with multiple observations and deck tests.
9. Configuration test proving tournament behavior uses database values.

## 11. Implementation boundary

This document is a design and handoff artifact, not authorization to begin a broad rewrite without a reviewed implementation plan. The next execution step is a complete impact analysis and schema/plan review against this document. Any deviation, simplification or missing contract must be surfaced for approval before code changes.

## 12. Transcript traceability index

The following original-message clusters establish the core requirements:

- 23:07–23:16: unified dashboard/replay/SQLite scope; no JSON/blob storage; full relational design.
- 23:19–23:47: deck builder, deck/card Elo, local/remote sources, arena behavior and local-only replay visibility.
- 13:04–13:23: always-on sweep-decks and vs-self; delivery must be complete rather than superficial.
- 13:58–14:45: official visualizer investigation, extraction from normalized replay data and isolated contract validation.
- 14:46–15:05: reference tables for enums, full schema audit and rebuild instead of migration.
- 15:27–15:59: dashboard operations, agent lifecycle, dynamic tournament configuration and Kaggle contract without first submission.
- 16:08–16:30: experiment generality, normalized training configuration, continuous anamnese and daily workflow without path dependence.
