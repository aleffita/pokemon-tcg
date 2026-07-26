"""Populate cards table from EN_Card_Data.csv.

Each CSV row represents one attack/move. Multiple rows share the same Card ID
for multi-attack Pokémon. This script deduplicates by Card ID, taking the first
row per card for static metadata (name, category, stage, HP, type, weakness,
rule).

Usage:
    uv run python scripts/populate_cards.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rl.results_db import ResultsDB

CSV_PATH = Path(__file__).resolve().parent.parent / "EN_Card_Data.csv"

# The accented column name as it appears in the CSV
STAGE_COL = "Stage (Pokémon)/Type (Energy and Trainer)"


def _clean(value: str) -> str | None:
    """Return stripped value or None if empty / 'n/a'."""
    v = value.strip()
    if not v or v.lower() == "n/a":
        return None
    return v


def _clean_int(value: str) -> int | None:
    """Return int value or None if not parseable."""
    v = value.strip()
    if not v or v.lower() == "n/a":
        return None
    try:
        return int(v)
    except ValueError:
        return None


def main():
    db = ResultsDB()

    # Check if already populated
    count = db.conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
    if count > 0:
        print(f"Cards table already has {count} entries. Skipping.")
        db.close()
        return

    # Read CSV, deduplicate by Card ID (take first row per card)
    seen: dict[int, dict] = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            card_id = int(row["Card ID"])
            if card_id not in seen:
                seen[card_id] = {
                    "card_id": card_id,
                    "name": row["Card Name"].strip(),
                    "category": _clean(row.get("Category", "")),
                    "stage": _clean(row.get(STAGE_COL, "")),
                    "hp": _clean_int(row.get("HP", "")),
                    "energy_type": _clean(row.get("Type", "")),
                    "weakness": _clean(row.get("Weakness", "")),
                    "rule": _clean(row.get("Rule", "")),
                }

    # Insert sorted by card_id
    for card in sorted(seen.values(), key=lambda c: c["card_id"]):
        db.add_card(**card)

    db.close()
    print(f"Populated {len(seen)} cards from {CSV_PATH.name}")


if __name__ == "__main__":
    main()
