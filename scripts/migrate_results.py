"""Migrate eval_results.txt to SQLite database.

One-time migration script. Reads the old text format and inserts into model/results.db.

Usage:
    uv run python scripts/migrate_results.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rl.results_db import ResultsDB, DB_PATH


def parse_results(text):
    # Split on the full separator line (=== ... ===) which appears as header+footer
    # The data block (matchups) sits between two separator blocks
    raw_blocks = re.split(r"={10,}", text)
    # Merge: header (Tournament:...) + data (matchups) into single logical blocks
    blocks = []
    i = 0
    while i < len(raw_blocks):
        block = raw_blocks[i].strip()
        if re.search(r"Tournament:\s*\d{4}", block):
            # This is a header block. Check if the NEXT block has matchup data
            if i + 1 < len(raw_blocks) and re.search(r"W=\s*\d+.*L=\s*\d+", raw_blocks[i + 1]):
                block = block + "\n" + raw_blocks[i + 1]
                i += 1
        if block:
            blocks.append(block)
        i += 1

    runs = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        ts = re.search(r"Tournament:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})", block)
        if not ts:
            continue
        agent = re.search(r"Agent:\s*(.+)", block)
        games = re.search(r"Games per opponent:\s*(\d+)", block)
        note = re.search(r"Note:\s*(.+)", block)
        time_m = re.search(r"Total time:\s*([\d.]+)s", block)

        matchups = []
        overall = None
        for line in block.splitlines():
            m = re.match(r"\s*(\S.*?)\s+W=\s*(\d+)\s+L=\s*(\d+)\s+D=\s*(\d+)\s+wr=\s*([\d.]+)%", line)
            if m:
                matchups.append({"opponent": m.group(1).strip(), "w": int(m.group(2)),
                                 "l": int(m.group(3)), "d": int(m.group(4)), "wr": float(m.group(5))})
            om = re.match(r"\s*OVERALL.*W=\s*(\d+)\s+L=\s*(\d+)\s+D=\s*(\d+)\s+wr=\s*([\d.]+)%", line)
            if om:
                overall = {"w": int(om.group(1)), "l": int(om.group(2)),
                           "d": int(om.group(3)), "wr": float(om.group(4))}

        if matchups and overall:
            runs.append({
                "timestamp": ts.group(1),
                "agent": agent.group(1).strip() if agent else "unknown",
                "games_per_opp": int(games.group(1)) if games else 20,
                "note": note.group(1).strip() if note else "",
                "matchups": matchups, "overall": overall,
                "total_time": float(time_m.group(1)) if time_m else 0,
            })
    return runs


def main():
    results_file = Path(__file__).resolve().parent.parent / "model" / "eval_results.txt"
    if not results_file.exists():
        print("No eval_results.txt found. Nothing to migrate.")
        return

    text = results_file.read_text()
    runs = parse_results(text)
    if not runs:
        print("No tournament data found in eval_results.txt.")
        return

    print(f"Found {len(runs)} tournament runs in eval_results.txt")

    db = ResultsDB()
    existing = db.get_all_runs()
    if existing:
        print(f"Database already has {len(existing)} runs. Skipping migration.")
        print(f"  (Delete {DB_PATH} to re-migrate)")
        db.close()
        return

    for run in runs:
        tid = db.add_tournament(
            timestamp=run["timestamp"], agent=run["agent"],
            games_per_opp=run["games_per_opp"], note=run["note"],
            matchups=run["matchups"], overall=run["overall"],
            total_time=run["total_time"])
        print(f"  Migrated: {run['timestamp']} — {run['overall']['wr']:.1f}% (id={tid})")

    db.close()
    print(f"\nDone! {len(runs)} runs migrated to {DB_PATH}")
    print(f"  Old file preserved at: {results_file}")


if __name__ == "__main__":
    main()
