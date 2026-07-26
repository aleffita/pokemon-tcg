# Pokémon TCG MLX — Complete Implementation Plan

> **Goal:** MLX migration (Phases A–F) + complete Elo/arena/dashboard system with deck builder, card/deck Elo, replay viewer, and self-play tracking.

---

## Progress — MLX Migration (Phases A–F)

| Phase | Status | Commits | Tests |
|-------|--------|---------|-------|
| Pre-Flight | ✅ Done | 1 | — |
| A — Canonical Contract | ✅ Done | 5 | 10/10 |
| B — Semantic P0 Fixes | ✅ Done | 2 | 7/7 |
| C — FP16-Native Trainer | ✅ Done | 1 | 7/7 |
| D — Data & Shapes | ✅ Done | 1 | 11/11 |
| E — Inference Semantics | ✅ Done | 1 | 9/9 |
| F — Minimal Recurrence | ✅ Done | 1 | 13/13 |
| Pipeline + Config + Entrypoints | ✅ Done | 4 | ✅ |
| Dashboard + SQLite | ⬜ Pending | — | — |

## Progress — Elo/Arena System

| Phase | Status | Description |
|-------|--------|-------------|
| G — SQLite Schema | ⬜ | Normalized schema: tournaments, matches, steps, cards, decks, Elo |
| H — Card/Deck Catalog | ⬜ | Populate cards table from EN_Card_Data.csv, decks from rl/deck/ |
| I — Replay Pipeline | ⬜ | Extract card/deck stats from Kaggle replays, compute Elo |
| J — Tournament Overhaul | ⬜ | Agent × Deck combos, sweep mode, self-play vs submissions |
| K — Streamlit Dashboard | ⬜ | Full dashboard: Cards, Decks, Agents, Arena, Replays, Builder |
| L — Deck Builder Integration | ⬜ | Visual deck builder inside dashboard |
| M — Daily Pipeline | ⬜ | Automated: download → stats → Elo → arena → dashboard |

---

## Execution Rules

1. **Modify existing files in-place.** No parallel trainers, no "v2" copies.
2. **`uv run` for everything.** No raw `python`/`python3`.
3. **Real data only.** No synthetic mocks. Smoke tests limit volume via config.
4. **Config hierarchy:** CLI args > `--config` file > `configs/train_config.json` > defaults.
5. **No PyTorch fallback.** MLX-only.
6. **No JSON in SQLite.** All data normalized into proper relational tables.
7. **Sequential agents on `develop` branch.** Each agent commits, next starts from that commit.
8. **Entrypoints for everything.** All scripts accessible via `uv run tcg-*`.

---

## Phase G — SQLite Schema

**Objective:** Complete normalized database covering tournaments, matches, steps, cards, decks, and Elo ratings.

### Tables

```sql
-- Existing (to migrate from results_db.py)
tournaments, matchups

-- Game data
matches          — individual games with agent × deck combos
match_steps      — each decision per game
step_options     — available options per step
step_events      — game log events per step
board_snapshots  — board state per step per player
pokemon_on_field — Pokemon in play per snapshot

-- Card/Deck catalog
cards            — all cards from EN_Card_Data.csv
decks            — known decks with source/archetype
deck_cards       — deck composition (N:M)

-- Elo ratings
card_elo         — per-card Elo (replay + arena)
deck_elo         — per-deck Elo (replay + arena)

-- Match composition
match_card_usage — which cards were in each match's deck
```

### Tasks

**G.1 — Extend `rl/results_db.py`**
- Add all new tables (cards, decks, deck_cards, card_elo, deck_elo, matches, match_steps, step_options, step_events, board_snapshots, pokemon_on_field, match_card_usage)
- Add foreign keys and indexes
- Migrate existing tournaments/matchups data

**G.2 — Create `scripts/populate_cards.py`**
- Read `EN_Card_Data.csv`, parse all unique cards
- Insert into `cards` table with: id, name, category, stage, hp, energy_type, weakness, rule

**G.3 — Create `scripts/populate_decks.py`**
- Read all decks from `rl/deck/` (decks.py, decks_meta.py, decks_kaggle.py, decks_generated.py, decks_train.py)
- Read `public_agents/*/deck.csv`
- Insert into `decks` table with source/archetype
- Insert into `deck_cards` table (deck_id, card_id, quantity)

**G.4 — Validation**
- Verify all foreign keys resolve
- Verify card count matches EN_Card_Data.csv
- Verify deck count matches all sources
- Run: `uv run tcg-db verify`

**Commit:** `feat(G): complete SQLite schema with cards, decks, Elo tables`

---

## Phase H — Card/Deck Elo System

**Objective:** Compute Elo ratings for individual cards and deck archetypes from match results.

### Elo Algorithm

**Card Elo:**
- When a match ends, each card in the winner's deck gets +delta, each in loser's gets -delta
- K-factor scaled by card quantity (4 copies = 4x weight)
- Cards that appear in many winning decks rise; cards in losing decks fall

**Deck Elo:**
- Standard Elo on the deck archetype identity
- Decks identified by card composition (hash or match against known archetypes)
- Unknown decks create new entries in the catalog

### Tasks

**H.1 — Card Elo computation in `rl/results_db.py`**
- `compute_card_elo(source='replay')` — from match_card_usage + match results
- `compute_deck_elo(source='replay')` — from matches + deck_id
- Store results in `card_elo` and `deck_elo` tables

**H.2 — Deck identification**
- `identify_deck(card_ids)` — match 60-card list against known decks in `decks` table
- Fuzzy match: sort card IDs, compare sets, threshold for "close enough"
- Unknown decks get auto-created with source='discovered'

**H.3 — Entrypoint `tcg-build-card-stats`**
- Process replay zips: extract deck choices + match results
- Update card_elo and deck_elo tables
- Run: `uv run tcg-build-card-stats --date 2026-07-25`

**H.4 — Validation**
- Verify Elo computation produces reasonable ratings
- Verify top cards match expected meta (Lillie's Determination, Boss's Orders, etc.)
- Verify deck identification works for known archetypes

**Commit:** `feat(H): card and deck Elo computation from match results`

---

## Phase I — Replay Pipeline

**Objective:** Extract structured replay data from Kaggle zips into SQLite (no JSON blobs).

### Tasks

**I.1 — Create `scripts/build_replay_db.py`**
- Stream episodes from replay zips (same approach as `build_bc_from_zips.py`)
- For each episode:
  1. Create `matches` row (with identified deck_id for each side)
  2. For each step per player: create `match_steps` row
  3. For each step: create `step_options`, `step_events`, `board_snapshots`, `pokemon_on_field` rows
  4. Create `match_card_usage` rows (60 cards per player per match)
- All data normalized — no JSON columns

**I.2 — Entrypoint `tcg-build-replays`**
- `uv run tcg-build-replays --date 2026-07-25`
- `uv run tcg-build-replays --range 2026-07-20 2026-07-25`
- Idempotent: skips already-processed episodes (by episode_id)

**I.3 — Validation**
- Verify step count matches game length
- Verify all cards referenced exist in `cards` table
- Verify board_snapshots are consistent (HP decreases, cards move zones)

**Commit:** `feat(I): replay pipeline — structured game data into SQLite`

---

## Phase J — Tournament Overhaul

**Objective:** Tournament supports Agent × Deck combinations, sweep mode, and self-play vs previous submissions.

### Agent Categories

| Category | Location | Deck | Used as |
|---|---|---|---|
| External | `public_agents/lb*/` | Their own `deck.csv` | Opponent |
| Our submissions | `public_agents/submissions/v*/` | Their own `deck.csv` | Opponent |
| Starters | `public_agents/starters/` | Their own `deck.csv` | Opponent |
| Current agent | `agent/main.py` | From deck catalog (Elo-ranked) | Us |

### Tournament Modes

```bash
# Standard: our agent vs all opponents
uv run tcg-tournament --games 20

# With specific deck
uv run tcg-tournament --games 20 --deck mega_lucario_ex

# Sweep: test top N decks from Elo rankings
uv run tcg-tournament --games 10 --sweep-decks 5

# Self-play only: vs our previous submissions
uv run tcg-tournament --games 20 --vs-self

# External only: vs leaderboard agents
uv run tcg-tournament --games 20 --vs-external
```

### Tasks

**J.1 — Modify `scripts/tournament.py`**
- Add `--deck` flag: override deck.csv for our agent
- Add `--sweep-decks N` flag: test top N decks from deck_elo
- Add `--vs-self` flag: only play against `public_agents/submissions/`
- Add `--vs-external` flag: only play against `public_agents/lb*/`
- Register matches with `deck_id` for both sides
- Record `match_card_usage` for each game

**J.2 — Submission workflow in `scripts/submit.py`**
- After successful submission, copy model to `public_agents/submissions/v{N}_{elo}_{date}/`
- Include `deck.csv` and `main.py` in the submission copy
- Register as new agent in the database

**J.3 — Validation**
- Verify sweep mode tests multiple decks
- Verify self-play includes our submissions
- Verify match_card_usage is recorded correctly

**Commit:** `feat(J): tournament overhaul — agent×deck combos, sweep, self-play`

---

## Phase K — Streamlit Dashboard

**Objective:** Complete dashboard with Cards, Decks, Agents, Arena, Replays, and Config tabs. No sidebar, relative paths, reads everything from SQLite.

### Tabs

| Tab | Content | Data Source |
|---|---|---|
| **Overview** | Latest Elo, win rate, best deck, Elo over time | tournaments, deck_elo |
| **Cards** | Card Elo rankings, filters by type/energy, usage stats | cards, card_elo |
| **Decks** | Deck Elo rankings, composition, archetype, history | decks, deck_elo, deck_cards |
| **Agents** | Our agents + opponents, each with deck and Elo | matchups, public_agents |
| **Arena** | Agent×Deck matrix, matchup results, sweep results | matches, matchups |
| **Replays** | Game list, step-by-step viewer | matches, match_steps, board_snapshots |
| **Config** | Smoke vs Train config with descriptions | configs/*.json, schema |

### Replay Viewer (inside Replays tab)

- Game selector: list of matches with opponent, result, steps
- Step slider: advance step-by-step
- Per step shows:
  - Board state: active Pokemon (HP, energies, tools), bench, hand count, deck count
  - Select context: what decision was made
  - Options: what was available
  - Action: what was chosen
  - Events: what happened (attacks, damage, card movements)

### Tasks

**K.1 — Rewrite `scripts/dashboard.py`**
- Remove sidebar entirely
- Read all data from `ResultsDB` (SQLite)
- Use relative paths in display
- 7 tabs as listed above
- Cards tab with sortable/filterable table
- Decks tab with composition viewer
- Arena tab with agent×deck matrix
- Replay tab with step-by-step viewer
- Config tab with schema descriptions

**K.2 — Replay viewer component**
- Step slider (Streamlit slider or number input)
- Board state display: Pokemon cards with HP bars, energy icons
- Action display: what option was selected
- Events display: game log entries
- Navigation: prev/next step, jump to turn

**K.3 — Card/Deck detail views**
- Click a card → shows: Elo, usage stats, which decks use it, win rate
- Click a deck → shows: composition (with card images), Elo history, matchup results

**Commit:** `feat(K): complete Streamlit dashboard with all tabs and replay viewer`

---

## Phase L — Deck Builder Integration

**Objective:** Visual deck builder inside the Streamlit dashboard, integrated with card Elo and deck catalog.

### Tasks

**L.1 — Port deck builder to Streamlit**
- Convert `scripts/deck_builder/build_deck_tool.py` from standalone HTML to Streamlit
- Card grid with images (from `scripts/deck_builder/card_images/`)
- Search, filter by type/energy/stage
- Add/remove cards, max 60
- Show card Elo alongside each card
- Load/save deck CSV
- Save new deck to `decks` table in SQLite

**L.2 — Deck comparison**
- Compare two decks side-by-side
- Show Elo difference per card
- Suggest improvements based on card Elo

**L.3 — Export deck for tournament**
- "Use in tournament" button → sets deck for next `tcg-tournament` run
- "Export CSV" → saves deck.csv for submission

**Commit:** `feat(L): deck builder integrated in Streamlit dashboard`

---

## Phase M — Daily Pipeline

**Objective:** Automated daily flow: download → stats → Elo → arena → dashboard.

### Pipeline Steps

```
1. tcg-data --last                    # Download latest replay zip
2. tcg-build-replays                  # Extract structured replay data → SQLite
3. tcg-build-card-stats               # Compute card/deck Elo from replays
4. tcg-tournament --sweep-decks 5     # Test top 5 decks in local arena
5. tcg-dashboard                      # Visualize everything
6. If better than previous submission:
   tcg-submit -m "vN deck_name"      # Submit to Kaggle
   Copy model → public_agents/submissions/
```

### Tasks

**M.1 — Create `scripts/daily_pipeline.py`**
- Orchestrates all steps in sequence
- Configurable via config.json
- Logs progress to SQLite (pipeline_runs table)
- Idempotent: skips already-processed dates

**M.2 — Entrypoint `tcg-daily`**
- `uv run tcg-daily` — run full pipeline
- `uv run tcg-daily --dry-run` — show what would happen
- `uv run tcg-daily --step replay` — run single step

**M.3 — Pipeline config in `configs/pipeline.json`**
```json
{
  "auto_submit": false,
  "sweep_decks_count": 5,
  "games_per_opponent": 20,
  "min_improvement_to_submit": 2.0,
  "replay_dates_to_keep": 30
}
```

**Commit:** `feat(M): daily pipeline — automated download → stats → Elo → arena`

---

## Agent Dispatch Table

| # | Agent | Phase | Files | Commit |
|---|-------|-------|-------|--------|
| G.1 | SQLite Schema | G | `rl/results_db.py` | `feat(G.1):` |
| G.2 | Populate Cards | G | `scripts/populate_cards.py` | `feat(G.2):` |
| G.3 | Populate Decks | G | `scripts/populate_decks.py` | `feat(G.3):` |
| G.4 | Schema Validation | G | `scripts/validate/test_schema.py` | `test(G.4):` |
| H.1 | Card/Deck Elo | H | `rl/results_db.py` | `feat(H.1):` |
| H.2 | Deck Identification | H | `rl/deck_identifier.py` | `feat(H.2):` |
| H.3 | Card Stats Pipeline | H | `scripts/build_card_stats.py` | `feat(H.3):` |
| H.4 | Elo Validation | H | `scripts/validate/test_elo.py` | `test(H.4):` |
| I.1 | Replay Pipeline | I | `scripts/build_replay_db.py` | `feat(I.1):` |
| I.2 | Replay Entrypoint | I | `scripts/build_replay_db.py` | `feat(I.2):` |
| I.3 | Replay Validation | I | `scripts/validate/test_replays.py` | `test(I.3):` |
| J.1 | Tournament Overhaul | J | `scripts/tournament.py` | `feat(J.1):` |
| J.2 | Submission Workflow | J | `scripts/submit.py` | `feat(J.2):` |
| J.3 | Tournament Validation | J | `scripts/validate/test_tournament.py` | `test(J.3):` |
| K.1 | Dashboard Rewrite | K | `scripts/dashboard.py` | `feat(K.1):` |
| K.2 | Replay Viewer | K | `scripts/dashboard.py` | `feat(K.2):` |
| K.3 | Card/Deck Detail | K | `scripts/dashboard.py` | `feat(K.3):` |
| L.1 | Deck Builder | L | `scripts/dashboard.py` | `feat(L.1):` |
| L.2 | Deck Comparison | L | `scripts/dashboard.py` | `feat(L.2):` |
| L.3 | Export for Tournament | L | `scripts/dashboard.py` | `feat(L.3):` |
| M.1 | Daily Pipeline | M | `scripts/daily_pipeline.py` | `feat(M.1):` |
| M.2 | Pipeline Entrypoint | M | `scripts/daily_pipeline.py` | `feat(M.2):` |
| M.3 | Pipeline Config | M | `configs/pipeline.json` | `feat(M.3):` |
| FINAL | End-to-End Smoke | All | — | `feat:` |

**Total: 23 agents, sequential on `develop`.**

---

## Entrypoints (complete list)

| Command | Description |
|---|---|
| `uv run tcg-data` | Kaggle dataset downloader |
| `uv run tcg-build-bc` | BC dataset builder from zips |
| `uv run tcg-build-daily` | Single-replay dataset builder |
| `uv run tcg-build-replays` | Structured replay data → SQLite |
| `uv run tcg-build-card-stats` | Card/deck Elo from replays |
| `uv run tcg-train` | MLX Metal GPU trainer |
| `uv run tcg-evaluate` | 1v1 evaluation |
| `uv run tcg-tournament` | Multi-opponent tournament (agent×deck) |
| `uv run tcg-submission` | Build submission.tar.gz |
| `uv run tcg-submit` | Submit to Kaggle |
| `uv run tcg-dashboard` | Streamlit Elo dashboard |
| `uv run tcg-daily` | Full daily pipeline |
| `uv run tcg-db` | Database utilities (verify, migrate, stats) |

---

## Related Wikifita Pages

- [[pokemon_tcg_mlx_migration]] — MLX implementation contract and phases
- [[pokemon_tcg_agent_architecture]] — current model and its ceiling
- [[pokemon_tcg_temporal_learning]] — sequence metadata and recurrence
- [[pokemon_tcg_ladder_and_research]] — Elo evaluation and deferred research
- [[uv_ecosystem]] — uv best practices and entrypoint conventions
