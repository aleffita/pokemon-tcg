"""Reconcile agent_elo_daily(source='remote') with the LIVE Kaggle leaderboard.

Uses the Kaggle Python API directly. Idempotent: replaces the source='remote'
slice per day before inserting. Agents not present in the Kaggle leaderboard
are dropped from source='remote' -- they naturally fall out of the top-N
curriculum filter used by bc_train_mlx.py.

Usage:
    uv run tcg-elo-reconcile
    uv run tcg-elo-reconcile --competition pokemon-tcg-ai-battle
    uv run tcg-elo-reconcile --db model/results.db
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rl.results_db import ResultsDB


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default="model/results.db",
        help="SQLite results database (default: model/results.db)",
    )
    parser.add_argument(
        "--competition",
        default="pokemon-tcg-ai-battle",
        help="Kaggle competition slug (default: pokemon-tcg-ai-battle)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"database not found: {db_path}")

    db = ResultsDB(str(db_path))
    try:
        matched = db.populate_agent_elo_from_kaggle_leaderboard(
            competition=args.competition
        )
    finally:
        db.close()

    n_matched = len(matched)
    if n_matched == 0:
        print(
            "[reconcile-elo] WARNING: no agents matched the Kaggle "
            "leaderboard by name; agent_elo_daily(source='remote') is empty."
        )
        return

    top = sorted(matched.items(), key=lambda kv: kv[1], reverse=True)[:10]
    bottom = sorted(matched.items(), key=lambda kv: kv[1])[:5]
    print(
        f"[reconcile-elo] matched {n_matched} of our agents to Kaggle "
        f"leaderboard; wrote {n_matched} rows per day into "
        f"agent_elo_daily(source='remote')."
    )
    print("[reconcile-elo] top 10 matched agents by REAL Kaggle score:")
    for name, score in top:
        print(f"  {score:>7.1f}   {name}")
    print("[reconcile-elo] bottom 5 matched agents by REAL Kaggle score:")
    for name, score in bottom:
        print(f"  {score:>7.1f}   {name}")


if __name__ == "__main__":
    main()
