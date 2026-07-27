"""Automated daily pipeline for Pokemon TCG MLX.

Steps:
1. tcg-data --last (download latest Kaggle replay)
2. tcg-build-card-stats (process replay -> card/deck Elo)
3. tcg-tournament (run local arena)
4. Report results

Usage:
    uv run tcg-daily
    uv run tcg-daily --dry-run
"""
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_step(name, cmd):
    """Run a pipeline step and return success status."""
    print(f"\n{'='*60}")
    print(f"  Step: {name}")
    print(f"  Command: {cmd}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, cwd=ROOT, timeout=3600)
        if result.returncode == 0:
            print(f"  [OK] {name} completed successfully")
            return True
        else:
            print(f"  [FAIL] {name} failed (exit code {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [FAIL] {name} timed out")
        return False


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="Show what would happen without executing")
    p.add_argument("--skip-download", action="store_true", help="Skip Kaggle download step")
    p.add_argument("--skip-stats", action="store_true", help="Skip card/deck stats computation")
    p.add_argument("--skip-tournament", action="store_true", help="Skip local tournament")
    p.add_argument("--tournament-games", type=int, default=10, help="Games per opponent in tournament")
    args = p.parse_args()

    start_time = datetime.now()
    print(f"Pokemon TCG Daily Pipeline -- {start_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Working directory: {ROOT}")

    steps = []

    # Step 1: Download
    if not args.skip_download:
        steps.append(("Download latest replay", "uv run tcg-data --last"))

    # Step 2: Card stats
    if not args.skip_stats:
        today = start_time.strftime('%Y-%m-%d')
        steps.append(("Build card stats", f"uv run tcg-build-card-stats --date {today}"))

    # Step 3: Tournament
    if not args.skip_tournament:
        steps.append(("Run tournament", f"uv run tcg-tournament --games {args.tournament_games}"))

    if not steps:
        print("Nothing to do. Use --skip-download, --skip-stats, --skip-tournament to control steps.")
        return

    results = []
    for name, cmd in steps:
        if args.dry_run:
            print(f"  [DRY RUN] Would execute: {cmd}")
            results.append((name, True))
        else:
            ok = run_step(name, cmd)
            results.append((name, ok))
            if not ok:
                print(f"\nPipeline stopped at '{name}'")
                break

    # Summary
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"  Pipeline Summary -- {elapsed.total_seconds():.0f}s")
    print(f"{'='*60}")
    for name, ok in results:
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {name}")

    # Show current state
    if not args.dry_run:
        try:
            from rl.results_db import ResultsDB
            db = ResultsDB()
            matches = db.conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            cards_elo = db.conn.execute("SELECT COUNT(*) FROM card_elo").fetchone()[0]
            decks_elo = db.conn.execute("SELECT COUNT(*) FROM deck_elo").fetchone()[0]
            db.close()
            print(f"\n  Database state:")
            print(f"    Matches: {matches}")
            print(f"    Card Elo entries: {cards_elo}")
            print(f"    Deck Elo entries: {decks_elo}")
        except Exception as e:
            print(f"\n  Could not read database state: {e}")

    print(f"\n  Dashboard: uv run tcg-dashboard")


if __name__ == "__main__":
    main()
