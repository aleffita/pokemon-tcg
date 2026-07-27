# Pokémon TCG MLX — Replay System, Schema Overhaul & Dashboard Vision

## Handoff Document — 2026-07-27

> This document captures the complete evolution of the discussion on replay systems, schema redesign, dashboard improvements, agent management, experiments tracking, and the full architectural vision. It is a living artifact from the conversation between Alefita and Claude.

---

## Part 1: User Requirements (Original Messages)

### 1.1 — Replay System Integration

**Request:** Study how Mikaelzinho's `scripts/run_battle.py` works with `result.html` and the official competition visualizer at `ptcgvis.heroz.jp`. Integrate the visualizer into the Streamlit dashboard.

**What was discovered:**
- `env.render(mode='html')` generates a self-contained React app (Vite-bundled) with canvas-based game board
- The HTML has "Open Visualizer Player 1" and "Open Visualizer Player 2" buttons
- Clicking these buttons creates a `<form>` with `method="POST"` and `target="_blank"`
- The form POSTs to `https://ptcgvis.heroz.jp/Visualizer/Replay/{episodeId}/{playerIdx}`
- The body contains `json={game_data_json}` — the complete game state from `env.render(mode='json')`
- The official visualizer renders with full card images, animations, proper game board

**Key insight:** The visualizer is hosted externally. The flow is: reconstruct game JSON → POST to ptcgvis.heroz.jp → opens official visualizer in new tab. No local HTML generation needed.

### 1.2 — Schema Overhaul

**Request:** The current SQLite schema has multiple problems that need fixing before the visualizer and new features can work.

**Problems identified:**
1. Enums stored as TEXT or magic INTEGER without reference tables
2. Missing foreign keys throughout the schema
3. Missing data fields needed by the visualizer
4. `zone_cards` table doesn't exist (needed for card lists per zone)
5. No proper agent tracking system
6. No experiment tracking
7. Tournament configs hardcoded via CLI, not stored in DB

### 1.3 — Dashboard Improvements

**Request:** The dashboard needs:
- Dual Elo display (local + remote) for cards
- Deck builder with card Elo, strength estimation
- Load deck from listing into builder
- Side-by-side deck comparison with base deck
- "Use this deck" button for tournament
- Arena tab with deck performance summary
- Visualizer integration for replays
- Configurable tournament parameters
- Agent management (onboarding, renaming, favorites)
- Experiment tracking
- Leaderboard integration with Kaggle API
- X1 matches between local agents
- Promotion system (arena → submission)

### 1.4 — Agent Management System

**Request:**
- `public_agents/arena/` — agents being tested before submission
- `public_agents/coliseu/` — defeated agents that keep competing
- `public_agents/submissions/` — submitted to Kaggle
- Agent identity via YAML files (name, version, deck, archetype, trained_with)
- Dashboard onboarding flow for existing agents
- Rename, favorite, search capabilities
- Each agent/deck combo tracked separately for Elo

### 1.5 — Experiment System

**Request:** NOT just a "note" field. An experiment is a flexible container that can represent:
- A new training run
- Testing an existing agent with a new deck
- Testing an existing agent with an opponent's deck
- Running Kaggle replays with different decks
- Arena testing with various combos

Each experiment has:
- Cross-referenced tables for results (training_runs, experiment_deck_tests, etc.)
- Continuous notes/observations (like a patient's anamnese — throughout the lifecycle, not just at creation)
- The experiment itself is generic; specific results live in type-specific tables

### 1.6 — Tournament System

**Request:**
- Tournament reads configs from DB tables, not CLI params
- Always runs sweep-decks (top N from Elo) + vs-self (previous submissions)
- Our agent tests with multiple decks, finds the best
- Each deck change = new Elo lineage (like a new submission)
- Elo starts at 600 for all new entities
- Global deck Elo = aggregation of local agent-deck Elo snapshots
- Kaggle leaderboard integration (competitive context)

### 1.7 — Daily Pipeline

**Request:**
- Simplified from current `daily_pipeline.py`
- Flow: download replays → process into DB → update card/deck Elo → run arena → dashboard shows results
- No file paths in the pipeline — everything relational in the DB

### 1.8 — Design Principles

- **SOLID principles** — simplicity comes from proper separation of concerns, not from cramming everything together
- **Normalization** — every enum gets its own table, no TEXT enums, no magic numbers
- **No paths unless necessary** — data lives in the DB, not referenced by file paths
- **Scalable architecture** — schema must support future features without redesign
- **Config as tables** — tournament configs editable from dashboard, not CLI
- **Agent YAML** — travels with the agent, describes identity
- **Anamnese** — continuous observation history, not a one-time "note" field

---

## Part 2: Analysis of Current State

### 2.1 — Current Database Schema Problems

**Enums as TEXT (violates relational design):**
- `pokemon_on_field.slot` — TEXT 'active'/'bench'
- `match_steps.status` — TEXT 'ACTIVE'/'INACTIVE'/'DONE'
- `card_elo.source`, `deck_elo.source`, `decks.source`, `matches.source` — TEXT
- `cards.category`, `cards.stage`, `cards.energy_type` — TEXT

**Enums as INTEGER without reference (magic numbers):**
- `step_events.event_type` — 0, 2, 3, 6, etc
- `step_options.option_type` — 0, 3, 7, 8, etc
- `match_steps.select_type` — 0, 1

**Missing foreign keys:**
- `pokemon_on_field.card_id` → cards.id
- `step_events.card_id` → cards.id
- `step_events.target_card_id` → cards.id

**Missing data for visualizer (15+ fields):**
- `current.firstPlayer`, `current.result`, `current.supporterPlayed`, `current.energyAttached`, `current.retreated`, `current.looking`, `current.turnActionCount`, `current.stadium`
- `players[i].deck[]`, `.hand[]`, `.discard[]`, `.prize[]` (card lists, not just counts)
- `select.contextCard`, `select.deck[]`, `select.effect`, `select.minCount/maxCount`
- `remainingOverageTime`
- `logs[]` with `from_area`, `to_area`

**Missing tables:**
- `zone_cards` — card lists per zone per snapshot
- All 13 enum tables
- Agent system tables
- Experiment system tables
- Leaderboard tables
- System config tables
- Tournament config tables

### 2.2 — Files That Interact with Database

**Write to DB:**
- `rl/results_db.py` — schema creation, all INSERT/UPDATE
- `scripts/tournament.py` — save_match_replay(), matchup/match inserts
- `scripts/build_card_stats.py` — remote match processing
- `scripts/populate_cards.py` — initial card data
- `scripts/populate_decks.py` — initial deck data
- `scripts/migrate_results.py` — one-time txt migration

**Read from DB:**
- `scripts/dashboard.py` — ~15 cached data-loading functions
- `scripts/tournament.py` — deck lookups, top_decks
- `scripts/build_card_stats.py` — deck identification
- `scripts/daily_pipeline.py` — status checks

### 2.3 — Kaggle API Capabilities

```
competition_leaderboard_view(comp) → [{team_id, team_name, score, submission_date}]
competition_submissions(comp)      → [{ref, fileName, date, status, publicScore, privateScore}]
competition_team_submissions(team_id) → submissions for a team
competition_list_episodes(submission_id) → episodes from a submission
competition_episode_replay(episode_id) → downloads replay data
competition_submit(file, message)  → submits new entry
```

Leaderboard is visible without submission. Episodes and submissions list need team/submission access.

### 2.4 — Visualizer Integration Flow

```
SQLite (normalized data) → reconstruct game JSON → POST to ptcgvis.heroz.jp/Visualizer/Replay/{episodeId}/{playerIdx}
```

The JSON format expected by the visualizer is the same as `env.render(mode='json')`. The visualizer is external, hosted by Heroz (competition organizers).

---

## Part 3: Proposed Schema (Final Design)

### ENUM TABLES (14)

```sql
CREATE TABLE zones (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
-- 1=deck, 2=hand, 3=discard, 4=active, 5=bench, 6=prize, 7=stadium, 12=looking

CREATE TABLE match_statuses (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
-- active, inactive, done

CREATE TABLE data_sources (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
-- local, remote

CREATE TABLE deck_sources (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
-- starter, meta, kaggle, public_agent, arena, coliseu, submission, builder

CREATE TABLE event_types (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
-- Engine event types: turn_start, draw, end_turn, move_card, play, attach, evolve, attack, damage, etc.

CREATE TABLE option_types (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
-- Engine option types: number, select_card, tool_card, energy_card, play, attack, ability, etc.

CREATE TABLE select_types (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
-- main, special

CREATE TABLE card_categories (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
-- pokemon, trainer, energy

CREATE TABLE card_stages (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
-- basic, stage1, stage2, item, supporter, stadium, tool, basic_energy, special_energy

CREATE TABLE pokemon_slots (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
-- active, bench

CREATE TABLE agent_statuses (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
-- arena, coliseu, submission, archived

CREATE TABLE experiment_statuses (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
-- pending, running, completed, failed

CREATE TABLE experiment_types (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
-- training, deck_test, replay_analysis, arena_run

CREATE TABLE submission_outcomes (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
-- pending, submitted, scored, error
```

### TOURNAMENT SYSTEM

```sql
CREATE TABLE tournament_configs (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    games_per_opp INTEGER DEFAULT 20,
    sweep_decks_count INTEGER DEFAULT 3,
    elo_k_factor INTEGER DEFAULT 32,
    elo_initial INTEGER DEFAULT 600,
    include_vs_self INTEGER DEFAULT 1,
    include_external INTEGER DEFAULT 1
);

CREATE TABLE tournaments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id INTEGER REFERENCES tournament_configs(id),
    timestamp TEXT NOT NULL,
    note TEXT DEFAULT '',
    total_w INTEGER DEFAULT 0, total_l INTEGER DEFAULT 0, total_d INTEGER DEFAULT 0,
    win_rate REAL DEFAULT 0.0, total_time_s REAL DEFAULT 0.0
);

CREATE TABLE matchups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_id INTEGER NOT NULL REFERENCES tournaments(id),
    opponent TEXT NOT NULL, lb_score INTEGER,
    w INTEGER DEFAULT 0, l INTEGER DEFAULT 0, d INTEGER DEFAULT 0,
    win_rate REAL DEFAULT 0.0
);

CREATE TABLE matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    matchup_id INTEGER REFERENCES matchups(id),
    experiment_id INTEGER REFERENCES experiments(id),
    game_index INTEGER NOT NULL,
    source_id INTEGER NOT NULL REFERENCES data_sources(id),
    our_agent_id INTEGER REFERENCES agents(id),
    our_deck_id INTEGER REFERENCES decks(id),
    opp_agent TEXT NOT NULL,
    opp_deck_id INTEGER REFERENCES decks(id),
    our_side INTEGER NOT NULL, result INTEGER NOT NULL,
    n_steps INTEGER DEFAULT 0, first_player INTEGER,
    remaining_time REAL DEFAULT 600.0
);
```

### AGENT SYSTEM

```sql
CREATE TABLE agents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name TEXT NOT NULL,
    folder_path TEXT NOT NULL,
    agent_type TEXT NOT NULL,      -- local, public, submission
    status_id INTEGER NOT NULL REFERENCES agent_statuses(id),
    is_favorite INTEGER DEFAULT 0,
    deck_id INTEGER REFERENCES decks(id),
    elo REAL DEFAULT 600.0,
    games_played INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0, losses INTEGER DEFAULT 0,
    win_rate REAL DEFAULT 0.0,
    lb_score INTEGER, lb_rank INTEGER,
    created_at TEXT, updated_at TEXT
);

CREATE TABLE agent_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id INTEGER NOT NULL REFERENCES agents(id),
    yaml_content TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE agent_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id INTEGER NOT NULL REFERENCES agents(id),
    content TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE agent_elo_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id INTEGER NOT NULL REFERENCES agents(id),
    deck_id INTEGER NOT NULL REFERENCES decks(id),
    elo REAL DEFAULT 600.0,
    games_played INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0, losses INTEGER DEFAULT 0,
    tournament_id INTEGER REFERENCES tournaments(id),
    created_at TEXT
);
```

### EXPERIMENT SYSTEM

```sql
CREATE TABLE experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    agent_id INTEGER REFERENCES agents(id),
    experiment_type_id INTEGER NOT NULL REFERENCES experiment_types(id),
    status_id INTEGER NOT NULL REFERENCES experiment_statuses(id),
    created_at TEXT, completed_at TEXT
);

-- Training-specific (referencia experiments)
CREATE TABLE training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    train_config_id INTEGER NOT NULL REFERENCES train_configs(id),
    val_acc REAL, val_loss REAL, train_loss REAL,
    total_steps INTEGER, training_time_s REAL,
    dataset_rows INTEGER,
    created_at TEXT, completed_at TEXT
);

-- Deck tests within experiment
CREATE TABLE experiment_deck_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    deck_id INTEGER NOT NULL REFERENCES decks(id),
    elo REAL DEFAULT 600.0,
    games_played INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0, losses INTEGER DEFAULT 0,
    win_rate REAL DEFAULT 0.0,
    is_default INTEGER DEFAULT 0
);

-- Matchups within experiment
CREATE TABLE experiment_matchups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    deck_test_id INTEGER REFERENCES experiment_deck_tests(id),
    opponent TEXT NOT NULL, lb_score INTEGER,
    w INTEGER DEFAULT 0, l INTEGER DEFAULT 0, d INTEGER DEFAULT 0,
    win_rate REAL DEFAULT 0.0
);

-- Continuous notes (anamnese)
CREATE TABLE experiment_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    content TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Replay analysis
CREATE TABLE replay_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER REFERENCES experiments(id),
    match_id INTEGER REFERENCES matches(id),
    analysis_type TEXT NOT NULL,
    result TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
```

### TRAIN CONFIGS

```sql
CREATE TABLE train_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    d_model INTEGER NOT NULL, nhead INTEGER NOT NULL, nlayers INTEGER NOT NULL,
    ff_dim INTEGER NOT NULL, static INTEGER NOT NULL, split_heads INTEGER NOT NULL,
    scratch_registers INTEGER NOT NULL, epochs INTEGER NOT NULL,
    batch_size INTEGER NOT NULL, accum_steps INTEGER NOT NULL,
    lr REAL NOT NULL, lr_schedule TEXT NOT NULL, warmup_steps INTEGER NOT NULL,
    lr_min_ratio REAL NOT NULL, max_grad_norm REAL NOT NULL,
    slab_rows INTEGER NOT NULL, val_frac REAL NOT NULL,
    tbptt_chunk INTEGER NOT NULL, seed INTEGER NOT NULL,
    bc_would_ko INTEGER NOT NULL, bc_wk_nvar INTEGER NOT NULL,
    config_hash TEXT NOT NULL UNIQUE,
    created_at TEXT DEFAULT (datetime('now'))
);
```

### REPLAY TABLES

```sql
CREATE TABLE match_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(id),
    step_num INTEGER NOT NULL, player_idx INTEGER NOT NULL,
    turn INTEGER DEFAULT 0, turn_action_count INTEGER DEFAULT 0,
    supporter_played INTEGER DEFAULT 0, stadium_played INTEGER DEFAULT 0,
    energy_attached INTEGER DEFAULT 0, retreated INTEGER DEFAULT 0,
    looking INTEGER, looking_count INTEGER,
    select_type_id INTEGER REFERENCES select_types(id),
    select_context INTEGER, n_options INTEGER DEFAULT 0,
    select_context_card_id INTEGER REFERENCES cards(id),
    select_effect_id INTEGER REFERENCES cards(id),
    select_min_count INTEGER DEFAULT 1, select_max_count INTEGER DEFAULT 1,
    action TEXT DEFAULT '[]',
    status_id INTEGER NOT NULL REFERENCES match_statuses(id),
    reward INTEGER DEFAULT 0
);

CREATE TABLE step_options (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step_id INTEGER NOT NULL REFERENCES match_steps(id),
    option_idx INTEGER NOT NULL,
    option_type_id INTEGER NOT NULL REFERENCES option_types(id),
    was_selected INTEGER DEFAULT 0,
    area INTEGER REFERENCES zones(id),
    in_play_area INTEGER REFERENCES zones(id),
    in_play_index INTEGER, card_index INTEGER
);

CREATE TABLE step_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step_id INTEGER NOT NULL REFERENCES match_steps(id),
    event_type_id INTEGER NOT NULL REFERENCES event_types(id),
    player_idx INTEGER, card_id INTEGER REFERENCES cards(id),
    serial INTEGER, target_card_id INTEGER REFERENCES cards(id),
    target_serial INTEGER,
    from_zone_id INTEGER REFERENCES zones(id),
    to_zone_id INTEGER REFERENCES zones(id),
    value INTEGER
);

CREATE TABLE board_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step_id INTEGER NOT NULL REFERENCES match_steps(id),
    player_idx INTEGER NOT NULL, turn INTEGER NOT NULL,
    deck_count INTEGER DEFAULT 0, hand_count INTEGER DEFAULT 0,
    prize_count INTEGER DEFAULT 0, discard_count INTEGER DEFAULT 0,
    poisoned INTEGER DEFAULT 0, burned INTEGER DEFAULT 0,
    asleep INTEGER DEFAULT 0, paralyzed INTEGER DEFAULT 0,
    confused INTEGER DEFAULT 0
);

CREATE TABLE pokemon_on_field (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL REFERENCES board_snapshots(id),
    slot_id INTEGER NOT NULL REFERENCES pokemon_slots(id),
    slot_idx INTEGER NOT NULL,
    card_id INTEGER NOT NULL REFERENCES cards(id),
    serial INTEGER NOT NULL,
    hp INTEGER NOT NULL, max_hp INTEGER NOT NULL,
    n_energies INTEGER DEFAULT 0, n_tools INTEGER DEFAULT 0,
    n_preevo INTEGER DEFAULT 0
);

CREATE TABLE zone_cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL REFERENCES board_snapshots(id),
    zone_id INTEGER NOT NULL REFERENCES zones(id),
    slot_idx INTEGER NOT NULL,
    card_id INTEGER NOT NULL REFERENCES cards(id),
    serial INTEGER NOT NULL, player_idx INTEGER NOT NULL
);

CREATE TABLE match_card_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(id),
    card_id INTEGER NOT NULL REFERENCES cards(id),
    player_side INTEGER NOT NULL, quantity INTEGER DEFAULT 1
);
```

### CATALOG + ELO

```sql
CREATE TABLE cards (id INTEGER PRIMARY KEY, name TEXT NOT NULL,
    category_id INTEGER REFERENCES card_categories(id),
    stage_id INTEGER REFERENCES card_stages(id),
    hp INTEGER, energy_type TEXT, weakness TEXT, rule TEXT);

CREATE TABLE decks (id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL, source_id INTEGER NOT NULL REFERENCES deck_sources(id),
    archetype TEXT, card_count INTEGER DEFAULT 60);

CREATE TABLE deck_cards (id INTEGER PRIMARY KEY AUTOINCREMENT,
    deck_id INTEGER REFERENCES decks(id), card_id INTEGER REFERENCES cards(id),
    quantity INTEGER DEFAULT 1);

CREATE TABLE card_elo (id INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id INTEGER REFERENCES cards(id), elo REAL DEFAULT 600.0,
    games_played INTEGER DEFAULT 0, wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0, win_rate REAL DEFAULT 0.0,
    source_id INTEGER NOT NULL REFERENCES data_sources(id));

CREATE TABLE deck_elo (id INTEGER PRIMARY KEY AUTOINCREMENT,
    deck_id INTEGER REFERENCES decks(id), elo REAL DEFAULT 600.0,
    games_played INTEGER DEFAULT 0, wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0, win_rate REAL DEFAULT 0.0,
    source_id INTEGER NOT NULL REFERENCES data_sources(id));
```

### SYSTEM CONFIGS

```sql
CREATE TABLE system_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE, value TEXT NOT NULL,
    description TEXT DEFAULT ''
);
-- Seeds: elo_k_factor=32, elo_initial=600, max_submissions_per_day=5
```

### LEADERBOARD

```sql
CREATE TABLE leaderboard_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL, team_name TEXT NOT NULL,
    score REAL NOT NULL, rank INTEGER,
    submission_date TEXT,
    fetched_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE my_teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL UNIQUE,
    team_name TEXT NOT NULL, is_primary INTEGER DEFAULT 1
);

CREATE TABLE my_submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER REFERENCES my_teams(id),
    submission_ref TEXT NOT NULL, file_name TEXT,
    score REAL, status TEXT,
    public_score REAL, private_score REAL,
    submitted_at TEXT, fetched_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE competition_episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id INTEGER REFERENCES my_submissions(id),
    episode_id INTEGER NOT NULL UNIQUE,
    fetched_at TEXT DEFAULT (datetime('now'))
);
```

---

## Part 4: Files Requiring Refactoring

| File | Changes |
|---|---|
| `rl/results_db.py` | REWRITE — new schema, enum seeds, all queries use JOINs |
| `scripts/tournament.py` | REWRITE `save_match_replay()` — new fields, enum FKs, zone_cards |
| `scripts/build_card_stats.py` | Adapt to new schema (enum FKs) |
| `scripts/populate_cards.py` | Adapt to enum FKs |
| `scripts/populate_decks.py` | Adapt to enum FKs |
| `scripts/dashboard.py` | REWRITE — new queries, new tabs (Arena, Coliseu, Experiments, X1, Configs) |
| `scripts/daily_pipeline.py` | Simplify with DB-centric flow |
| `scripts/submit.py` | Integrate with agent promotion |
| `agent/main.py` | No changes |

---

## Part 5: Visualizer Integration

**Flow:** Reconstruct game JSON from normalized DB → POST to `ptcgvis.heroz.jp/Visualizer/Replay/`

**Data mapping (DB → JSON):**
- `match_steps` → `steps[i][p].action`, `.status`, `.reward`
- `board_snapshots` → `current.players[i].deckCount`, `.handCount`, etc.
- `pokemon_on_field` → `current.players[i].active[]`, `.bench[]`
- `zone_cards` → `current.players[i].deck[]`, `.hand[]`, `.discard[]`, `.prize[]`
- `step_events` → `observation.logs[]`
- `step_options` → `observation.select.option[]`
- `match_steps.select_*` → `observation.select.type`, `.context`, etc.

**Additional fields needed in `match_steps`:**
- `first_player`, `turn_action_count`, `supporter_played`, `stadium_played`, `energy_attached`, `retreated`, `looking`, `looking_count`, `select_context_card_id`, `select_effect_id`, `select_min_count`, `select_max_count`, `remaining_time`

---

## Part 6: Daily Pipeline Flow

```
1. tcg-data --last                    → baixa replay zip
2. tcg-build-card-stats               → processa remotos → card_elo, deck_elo
3. tcg-tournament                     → arena local → matches, card_elo, deck_elo
4. tcg-dashboard                      → visualiza tudo
```

No file paths in the pipeline — everything relational in the DB.

---

## Part 7: Agent Lifecycle

```
public_agents/starters/  →  onboarding na dashboard  →  agents.status=arena
public_agents/lb*/       →  onboarding na dashboard  →  agents.status=public_agent
agent/main.py (treino)   →  experiment criado        →  agents.status=arena
arena → torneio → bom desempenho → botão "promover" →  agents.status=submission → submissions/
arena → torneio → derrota     →                        agents.status=coliseu → coliseu/
submission → Kaggle score    →                        agents.lb_score, lb_rank atualizados
```

---

## Part 8: Design Principles

1. **SOLID** — cada tabela com sua responsabilidade, relações triviais
2. **Normalization** — todo enum = tabela, zero TEXT enums, zero magic numbers
3. **No paths** — dados vivem no banco, não referenciados por paths
4. **Scalable** — schema suporta features futuras sem redesign
5. **Config as tables** — editáveis via dashboard, não CLI
6. **Anamnese contínua** — observações acompanham a vida toda, não snapshot único
7. **Train config normalizada** — referenciada por experiments, não duplicada

---

## Part 9: What's Done vs What's Pending

### Done (committed on develop)
- ✅ MLX Migration (Phases A-F)
- ✅ SQLite schema v1 (15 tables)
- ✅ Card/Deck Elo computation
- ✅ Tournament with sweep-decks + vs-self
- ✅ Streamlit dashboard (8 tabs)
- ✅ Deck builder integration
- ✅ Config system (CLI > JSON > defaults)
- ✅ Kaggle data downloader
- ✅ Daily pipeline
- ✅ Submission script

### Pending (this design document)
- ⬜ Schema v2 (14 enum tables + new tables)
- ⬜ Replay reconstruction from DB
- ⬜ Visualizer integration
- ⬜ Agent management system
- ⬜ Experiment tracking
- ⬜ Leaderboard integration
- ⬜ Dashboard rewrite (Arena, Coliseu, Experiments, X1, Configs)
- ⬜ Tournament config from DB
- ⬜ Agent YAML onboarding flow
- ⬜ X1 matches between local agents
