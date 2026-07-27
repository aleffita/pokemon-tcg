"""SQLite database for tournament results, model tracking, and Elo ratings.

Single source of truth for all evaluation data. Replaces eval_results.txt.

Usage:
    from rl.results_db import ResultsDB
    db = ResultsDB()  # default: model/results.db
    db.add_tournament(timestamp, agent, games_per_opp, note, matchups, overall)
    runs = db.get_all_runs()
    elos = db.compute_elos()
"""
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent / "model" / "results.db"

K = 32
INITIAL_ELO = 600


def _extract_lb_score(label: str) -> Optional[int]:
    import re
    m = re.search(r"lb(\d+)", label)
    return int(m.group(1)) if m else None


class ResultsDB:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            -- Tournament tracking (existing)
            CREATE TABLE IF NOT EXISTS tournaments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                agent TEXT NOT NULL,
                games_per_opp INTEGER NOT NULL DEFAULT 20,
                note TEXT DEFAULT '',
                total_w INTEGER NOT NULL DEFAULT 0,
                total_l INTEGER NOT NULL DEFAULT 0,
                total_d INTEGER NOT NULL DEFAULT 0,
                win_rate REAL NOT NULL DEFAULT 0.0,
                total_time_s REAL DEFAULT 0.0,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS matchups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tournament_id INTEGER NOT NULL,
                opponent TEXT NOT NULL,
                w INTEGER NOT NULL DEFAULT 0,
                l INTEGER NOT NULL DEFAULT 0,
                d INTEGER NOT NULL DEFAULT 0,
                win_rate REAL NOT NULL DEFAULT 0.0,
                lb_score INTEGER,
                replay_html TEXT,
                FOREIGN KEY (tournament_id) REFERENCES tournaments(id)
            );

            -- Card/Deck catalog (referenced by matches and elo)
            CREATE TABLE IF NOT EXISTS cards (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT,
                stage TEXT,
                hp INTEGER,
                energy_type TEXT,
                weakness TEXT,
                rule TEXT
            );

            CREATE TABLE IF NOT EXISTS decks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL,
                archetype TEXT,
                card_count INTEGER NOT NULL DEFAULT 60
            );

            CREATE TABLE IF NOT EXISTS deck_cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deck_id INTEGER NOT NULL REFERENCES decks(id) ON DELETE CASCADE,
                card_id INTEGER NOT NULL REFERENCES cards(id),
                quantity INTEGER NOT NULL DEFAULT 1,
                UNIQUE(deck_id, card_id)
            );

            -- Game data (matches and their steps)
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                matchup_id INTEGER REFERENCES matchups(id),
                game_index INTEGER NOT NULL,
                source TEXT NOT NULL DEFAULT 'local',
                our_agent TEXT NOT NULL,
                our_deck_id INTEGER REFERENCES decks(id),
                opp_agent TEXT NOT NULL,
                opp_deck_id INTEGER REFERENCES decks(id),
                our_side INTEGER NOT NULL,
                result INTEGER NOT NULL,
                n_steps INTEGER NOT NULL DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(matchup_id, game_index)
            );

            CREATE TABLE IF NOT EXISTS match_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
                step_num INTEGER NOT NULL,
                player_idx INTEGER NOT NULL,
                turn INTEGER NOT NULL DEFAULT 0,
                select_type INTEGER,
                select_context INTEGER,
                n_options INTEGER NOT NULL DEFAULT 0,
                action TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL,
                reward INTEGER NOT NULL DEFAULT 0,
                UNIQUE(match_id, step_num, player_idx)
            );

            CREATE TABLE IF NOT EXISTS step_options (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step_id INTEGER NOT NULL REFERENCES match_steps(id) ON DELETE CASCADE,
                option_idx INTEGER NOT NULL,
                option_type INTEGER NOT NULL,
                was_selected INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS step_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step_id INTEGER NOT NULL REFERENCES match_steps(id) ON DELETE CASCADE,
                event_type INTEGER NOT NULL,
                player_idx INTEGER,
                card_id INTEGER,
                serial INTEGER,
                target_card_id INTEGER,
                target_serial INTEGER,
                value INTEGER
            );

            CREATE TABLE IF NOT EXISTS board_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step_id INTEGER NOT NULL REFERENCES match_steps(id) ON DELETE CASCADE,
                player_idx INTEGER NOT NULL,
                turn INTEGER NOT NULL,
                deck_count INTEGER NOT NULL DEFAULT 0,
                hand_count INTEGER NOT NULL DEFAULT 0,
                prize_count INTEGER NOT NULL DEFAULT 0,
                discard_count INTEGER NOT NULL DEFAULT 0,
                poisoned INTEGER NOT NULL DEFAULT 0,
                burned INTEGER NOT NULL DEFAULT 0,
                asleep INTEGER NOT NULL DEFAULT 0,
                paralyzed INTEGER NOT NULL DEFAULT 0,
                confused INTEGER NOT NULL DEFAULT 0,
                UNIQUE(step_id, player_idx)
            );

            CREATE TABLE IF NOT EXISTS pokemon_on_field (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL REFERENCES board_snapshots(id) ON DELETE CASCADE,
                slot TEXT NOT NULL,
                slot_idx INTEGER NOT NULL,
                card_id INTEGER NOT NULL,
                serial INTEGER NOT NULL,
                hp INTEGER NOT NULL,
                max_hp INTEGER NOT NULL,
                n_energies INTEGER NOT NULL DEFAULT 0,
                n_tools INTEGER NOT NULL DEFAULT 0,
                n_preevo INTEGER NOT NULL DEFAULT 0
            );

            -- Elo ratings
            CREATE TABLE IF NOT EXISTS card_elo (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                card_id INTEGER NOT NULL REFERENCES cards(id),
                elo REAL NOT NULL DEFAULT 600.0,
                games_played INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                win_rate REAL NOT NULL DEFAULT 0.0,
                source TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(card_id, source)
            );

            CREATE TABLE IF NOT EXISTS deck_elo (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deck_id INTEGER NOT NULL REFERENCES decks(id),
                elo REAL NOT NULL DEFAULT 600.0,
                games_played INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                win_rate REAL NOT NULL DEFAULT 0.0,
                source TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(deck_id, source)
            );

            -- Match card composition
            CREATE TABLE IF NOT EXISTS match_card_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
                card_id INTEGER NOT NULL REFERENCES cards(id),
                player_side INTEGER NOT NULL,
                quantity INTEGER NOT NULL DEFAULT 1,
                UNIQUE(match_id, card_id, player_side)
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_matchups_tournament
                ON matchups(tournament_id);
            CREATE INDEX IF NOT EXISTS idx_matchups_opponent
                ON matchups(opponent);
            CREATE INDEX IF NOT EXISTS idx_tournaments_timestamp
                ON tournaments(timestamp);
            CREATE INDEX IF NOT EXISTS idx_matches_matchup
                ON matches(matchup_id);
            CREATE INDEX IF NOT EXISTS idx_matches_source
                ON matches(source);
            CREATE INDEX IF NOT EXISTS idx_steps_match
                ON match_steps(match_id);
            CREATE INDEX IF NOT EXISTS idx_options_step
                ON step_options(step_id);
            CREATE INDEX IF NOT EXISTS idx_events_step
                ON step_events(step_id);
            CREATE INDEX IF NOT EXISTS idx_snapshots_step
                ON board_snapshots(step_id);
            CREATE INDEX IF NOT EXISTS idx_pokemon_snapshot
                ON pokemon_on_field(snapshot_id);
            CREATE INDEX IF NOT EXISTS idx_deck_cards_deck
                ON deck_cards(deck_id);
            CREATE INDEX IF NOT EXISTS idx_deck_cards_card
                ON deck_cards(card_id);
            CREATE INDEX IF NOT EXISTS idx_card_elo_card
                ON card_elo(card_id);
            CREATE INDEX IF NOT EXISTS idx_deck_elo_deck
                ON deck_elo(deck_id);
            CREATE INDEX IF NOT EXISTS idx_card_usage_match
                ON match_card_usage(match_id);
            CREATE INDEX IF NOT EXISTS idx_card_usage_card
                ON match_card_usage(card_id);
        """)
        self.conn.commit()

    def add_tournament(self, timestamp: str, agent: str, games_per_opp: int,
                       note: str, matchups: list[dict], overall: dict,
                       total_time: float = 0.0):
        """Add a tournament run and its matchups."""
        cur = self.conn.execute(
            """INSERT INTO tournaments (timestamp, agent, games_per_opp, note,
               total_w, total_l, total_d, win_rate, total_time_s)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, agent, games_per_opp, note,
             overall["w"], overall["l"], overall["d"], overall["wr"], total_time))
        tid = cur.lastrowid
        for m in matchups:
            self.conn.execute(
                """INSERT INTO matchups (tournament_id, opponent, w, l, d, win_rate, lb_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (tid, m["opponent"], m["w"], m["l"], m["d"], m["wr"],
                 _extract_lb_score(m["opponent"])))
        self.conn.commit()
        return tid

    def get_all_runs(self) -> list[dict]:
        """Get all tournaments with their matchups."""
        rows = self.conn.execute(
            "SELECT * FROM tournaments ORDER BY timestamp").fetchall()
        runs = []
        for row in rows:
            matchups = self.conn.execute(
                "SELECT * FROM matchups WHERE tournament_id = ? ORDER BY id",
                (row["id"],)).fetchall()
            runs.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "agent": row["agent"],
                "games_per_opp": row["games_per_opp"],
                "note": row["note"],
                "total_w": row["total_w"],
                "total_l": row["total_l"],
                "total_d": row["total_d"],
                "win_rate": row["win_rate"],
                "total_time_s": row["total_time_s"],
                "matchups": [{
                    "opponent": m["opponent"], "w": m["w"], "l": m["l"],
                    "d": m["d"], "wr": m["win_rate"], "lb_score": m["lb_score"],
                } for m in matchups],
                "overall": {"w": row["total_w"], "l": row["total_l"],
                            "d": row["total_d"], "wr": row["win_rate"]},
            })
        return runs

    def compute_elos(self) -> dict[str, float]:
        """Compute Elo ratings from all tournament results."""
        runs = self.get_all_runs()
        elos = {}
        for run in runs:
            label = run["timestamp"]
            if label not in elos:
                elos[label] = INITIAL_ELO
            for m in run["matchups"]:
                opp = m["opponent"]
                if opp not in elos:
                    lb = m.get("lb_score") or _extract_lb_score(opp)
                    elos[opp] = lb if lb else INITIAL_ELO
        for run in runs:
            label = run["timestamp"]
            for m in run["matchups"]:
                total = m["w"] + m["l"] + m["d"]
                if total == 0:
                    continue
                score = (m["w"] + 0.5 * m["d"]) / total
                ea = 1 / (1 + 10 ** ((elos[m["opponent"]] - elos[label]) / 400))
                delta = K * (score - ea)
                elos[label] += delta
                elos[m["opponent"]] -= delta
        return elos

    def get_latest_run(self) -> Optional[dict]:
        """Get the most recent tournament run."""
        runs = self.get_all_runs()
        return runs[-1] if runs else None

    def get_best_run(self) -> Optional[dict]:
        """Get the tournament run with highest win rate."""
        runs = self.get_all_runs()
        return max(runs, key=lambda r: r["win_rate"]) if runs else None

    def get_opponent_history(self, opponent: str) -> list[dict]:
        """Get all matchup results against a specific opponent."""
        rows = self.conn.execute(
            """SELECT m.*, t.timestamp, t.agent, t.note
               FROM matchups m JOIN tournaments t ON m.tournament_id = t.id
               WHERE m.opponent = ? ORDER BY t.timestamp""",
            (opponent,)).fetchall()
        return [dict(r) for r in rows]

    # --- Match data ---

    def add_match(self, matchup_id: int, game_index: int, source: str,
                  our_agent: str, our_deck_id: int | None,
                  opp_agent: str, opp_deck_id: int | None,
                  our_side: int, result: int, n_steps: int = 0) -> int:
        """Add a single match. Returns match id.

        Always fetches the match id by querying after insert to avoid
        lastrowid issues with INSERT OR IGNORE on duplicates.
        """
        self.conn.execute(
            """INSERT OR IGNORE INTO matches
               (matchup_id, game_index, source, our_agent, our_deck_id,
                opp_agent, opp_deck_id, our_side, result, n_steps)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (matchup_id, game_index, source, our_agent, our_deck_id,
             opp_agent, opp_deck_id, our_side, result, n_steps))
        self.conn.commit()
        row = self.conn.execute(
            "SELECT id FROM matches WHERE matchup_id = ? AND game_index = ?",
            (matchup_id, game_index)).fetchone()
        return row["id"] if row else None

    def add_match_steps(self, match_id: int, steps_data: list[dict]):
        """Add steps for a match. steps_data is a list of dicts with keys:
        step_num, player_idx, turn, select_type, select_context,
        n_options, action, status, reward, and optionally
        options (list of dicts), events (list of dicts),
        snapshot (dict with board state and pokemon list).
        """
        for step in steps_data:
            cur = self.conn.execute(
                """INSERT OR IGNORE INTO match_steps
                   (match_id, step_num, player_idx, turn, select_type,
                    select_context, n_options, action, status, reward)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (match_id, step["step_num"], step["player_idx"],
                 step.get("turn", 0), step.get("select_type"),
                 step.get("select_context"), step.get("n_options", 0),
                 step.get("action", "[]"), step["status"],
                 step.get("reward", 0)))
            step_id = cur.lastrowid
            if not step_id:
                continue

            for opt in step.get("options", []):
                self.conn.execute(
                    """INSERT INTO step_options
                       (step_id, option_idx, option_type, was_selected)
                       VALUES (?, ?, ?, ?)""",
                    (step_id, opt["option_idx"], opt["option_type"],
                     opt.get("was_selected", 0)))

            for evt in step.get("events", []):
                self.conn.execute(
                    """INSERT INTO step_events
                       (step_id, event_type, player_idx, card_id, serial,
                        target_card_id, target_serial, value)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (step_id, evt["event_type"], evt.get("player_idx"),
                     evt.get("card_id"), evt.get("serial"),
                     evt.get("target_card_id"), evt.get("target_serial"),
                     evt.get("value")))

            snap = step.get("snapshot")
            if snap:
                snap_cur = self.conn.execute(
                    """INSERT OR IGNORE INTO board_snapshots
                       (step_id, player_idx, turn, deck_count, hand_count,
                        prize_count, discard_count, poisoned, burned,
                        asleep, paralyzed, confused)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (step_id, snap["player_idx"], snap.get("turn", 0),
                     snap.get("deck_count", 0), snap.get("hand_count", 0),
                     snap.get("prize_count", 0), snap.get("discard_count", 0),
                     snap.get("poisoned", 0), snap.get("burned", 0),
                     snap.get("asleep", 0), snap.get("paralyzed", 0),
                     snap.get("confused", 0)))
                snap_id = snap_cur.lastrowid
                if snap_id:
                    for pokemon in snap.get("pokemon", []):
                        self.conn.execute(
                            """INSERT INTO pokemon_on_field
                               (snapshot_id, slot, slot_idx, card_id, serial,
                                hp, max_hp, n_energies, n_tools, n_preevo)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (snap_id, pokemon["slot"], pokemon["slot_idx"],
                             pokemon["card_id"], pokemon["serial"],
                             pokemon["hp"], pokemon["max_hp"],
                             pokemon.get("n_energies", 0),
                             pokemon.get("n_tools", 0),
                             pokemon.get("n_preevo", 0)))
        self.conn.commit()

    # --- Card/Deck catalog ---

    def add_card(self, card_id: int, name: str, category: str | None = None,
                 stage: str | None = None, hp: int | None = None,
                 energy_type: str | None = None, weakness: str | None = None,
                 rule: str | None = None):
        """Add a card to the catalog."""
        self.conn.execute(
            """INSERT OR REPLACE INTO cards
               (id, name, category, stage, hp, energy_type, weakness, rule)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (card_id, name, category, stage, hp, energy_type, weakness, rule))
        self.conn.commit()

    def add_deck(self, name: str, source: str, archetype: str | None = None,
                 card_count: int = 60) -> int:
        """Add a deck. Returns deck_id."""
        cur = self.conn.execute(
            """INSERT OR IGNORE INTO decks (name, source, archetype, card_count)
               VALUES (?, ?, ?, ?)""",
            (name, source, archetype, card_count))
        self.conn.commit()
        if cur.lastrowid:
            return cur.lastrowid
        row = self.conn.execute(
            "SELECT id FROM decks WHERE name = ?", (name,)).fetchone()
        return row["id"] if row else None

    def add_deck_cards(self, deck_id: int,
                       card_quantities: list[tuple[int, int]]):
        """Add cards to a deck. card_quantities: list of (card_id, quantity)."""
        for card_id, qty in card_quantities:
            self.conn.execute(
                """INSERT OR REPLACE INTO deck_cards (deck_id, card_id, quantity)
                   VALUES (?, ?, ?)""",
                (deck_id, card_id, qty))
        self.conn.commit()

    # --- Elo lookups ---

    def get_card_elo(self, card_id: int, source: str = "replay") -> dict | None:
        """Get Elo for a card."""
        row = self.conn.execute(
            """SELECT * FROM card_elo WHERE card_id = ? AND source = ?""",
            (card_id, source)).fetchone()
        return dict(row) if row else None

    def get_deck_elo(self, deck_id: int, source: str = "replay") -> dict | None:
        """Get Elo for a deck."""
        row = self.conn.execute(
            """SELECT * FROM deck_elo WHERE deck_id = ? AND source = ?""",
            (deck_id, source)).fetchone()
        return dict(row) if row else None

    def get_top_cards(self, n: int = 50,
                      source: str = "replay") -> list[dict]:
        """Get top N cards by Elo."""
        rows = self.conn.execute(
            """SELECT c.id, c.name, c.category, c.energy_type,
                      ce.elo, ce.games_played, ce.win_rate
               FROM card_elo ce
               JOIN cards c ON ce.card_id = c.id
               WHERE ce.source = ?
               ORDER BY ce.elo DESC LIMIT ?""",
            (source, n)).fetchall()
        return [{"id": r[0], "name": r[1], "category": r[2],
                 "energy_type": r[3], "elo": r[4],
                 "games_played": r[5], "win_rate": r[6]} for r in rows]

    def get_top_decks(self, n: int = 20,
                      source: str = "replay") -> list[dict]:
        """Get top N decks by Elo."""
        rows = self.conn.execute(
            """SELECT d.id, d.name, d.source, d.archetype,
                      de.elo, de.games_played, de.win_rate
               FROM deck_elo de
               JOIN decks d ON de.deck_id = d.id
               WHERE de.source = ?
               ORDER BY de.elo DESC LIMIT ?""",
            (source, n)).fetchall()
        return [{"id": r[0], "name": r[1], "source": r[2],
                 "archetype": r[3], "elo": r[4],
                 "games_played": r[5], "win_rate": r[6]} for r in rows]

    def compute_card_elo(self, source='replay'):
        """Compute Elo for all cards from match results."""
        K = 32
        # Get all remote matches with results
        matches = self.conn.execute(
            "SELECT id, result FROM matches WHERE source = ?"
        , (source,)).fetchall()

        if not matches:
            return {}

        # Initialize all card elos
        card_elos = {}
        for row in self.conn.execute("SELECT id FROM cards").fetchall():
            card_elos[row[0]] = INITIAL_ELO

        for match_row in matches:
            match_id, result = match_row
            if result == 0:
                continue  # skip draws

            # Get cards for each side
            winner_side = 0 if result == 1 else 1
            loser_side = 1 - winner_side

            winner_cards = self.conn.execute(
                "SELECT card_id, quantity FROM match_card_usage WHERE match_id = ? AND player_side = ?",
                (match_id, winner_side)).fetchall()
            loser_cards = self.conn.execute(
                "SELECT card_id, quantity FROM match_card_usage WHERE match_id = ? AND player_side = ?",
                (match_id, loser_side)).fetchall()

            if not winner_cards or not loser_cards:
                continue

            # Average Elo of each deck
            winner_avg = sum(card_elos.get(c, INITIAL_ELO) for c, _ in winner_cards) / len(winner_cards)
            loser_avg = sum(card_elos.get(c, INITIAL_ELO) for c, _ in loser_cards) / len(loser_cards)

            # Elo delta
            ea = 1 / (1 + 10 ** ((loser_avg - winner_avg) / 400))
            delta = K * (1 - ea)

            # Apply to each card (weighted by quantity)
            for card_id, qty in winner_cards:
                card_elos[card_id] = card_elos.get(card_id, INITIAL_ELO) + delta * qty / 4
            for card_id, qty in loser_cards:
                card_elos[card_id] = card_elos.get(card_id, INITIAL_ELO) - delta * qty / 4

        # Save to card_elo table
        for card_id, elo in card_elos.items():
            wins = self.conn.execute(
                "SELECT COUNT(DISTINCT m.id) FROM matches m JOIN match_card_usage mcu ON m.id = mcu.match_id WHERE mcu.card_id = ? AND m.source = ? AND ((mcu.player_side = 0 AND m.result = 1) OR (mcu.player_side = 1 AND m.result = -1))",
                (card_id, source)).fetchone()[0]
            losses = self.conn.execute(
                "SELECT COUNT(DISTINCT m.id) FROM matches m JOIN match_card_usage mcu ON m.id = mcu.match_id WHERE mcu.card_id = ? AND m.source = ? AND ((mcu.player_side = 0 AND m.result = -1) OR (mcu.player_side = 1 AND m.result = 1))",
                (card_id, source)).fetchone()[0]
            games = wins + losses
            wr = wins / games if games > 0 else 0.0

            self.conn.execute(
                "INSERT OR REPLACE INTO card_elo (card_id, elo, games_played, wins, losses, win_rate, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (card_id, elo, games, wins, losses, wr, source))
        self.conn.commit()
        return card_elos

    def compute_deck_elo(self, source='replay'):
        """Compute Elo for all decks from matches."""
        K = 32
        deck_elos = {}

        matches = self.conn.execute(
            "SELECT id, our_deck_id, opp_deck_id, result FROM matches WHERE our_deck_id IS NOT NULL AND opp_deck_id IS NOT NULL AND source = ?",
            (source,)).fetchall()

        for match_id, our_deck, opp_deck, result in matches:
            if our_deck not in deck_elos:
                deck_elos[our_deck] = INITIAL_ELO
            if opp_deck not in deck_elos:
                deck_elos[opp_deck] = INITIAL_ELO

            if result == 0:
                continue

            ra = deck_elos[our_deck]
            rb = deck_elos[opp_deck]
            ea = 1 / (1 + 10 ** ((rb - ra) / 400))
            delta = K * ((1 if result == 1 else 0) - ea)

            deck_elos[our_deck] += delta
            deck_elos[opp_deck] -= delta

        for deck_id, elo in deck_elos.items():
            self.conn.execute(
                "INSERT OR REPLACE INTO deck_elo (deck_id, elo, games_played, wins, losses, win_rate, source) VALUES (?, ?, 0, 0, 0, 0, ?)",
                (deck_id, elo, source))
        self.conn.commit()
        return deck_elos

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
