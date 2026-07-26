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
INITIAL_ELO = 1000


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
                FOREIGN KEY (tournament_id) REFERENCES tournaments(id)
            );

            CREATE INDEX IF NOT EXISTS idx_matchups_tournament
                ON matchups(tournament_id);
            CREATE INDEX IF NOT EXISTS idx_matchups_opponent
                ON matchups(opponent);
            CREATE INDEX IF NOT EXISTS idx_tournaments_timestamp
                ON tournaments(timestamp);
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

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
