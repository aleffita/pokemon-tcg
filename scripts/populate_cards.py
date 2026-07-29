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


def populate_cards(
    db_or_path: ResultsDB | str | Path | None = None,
    *,
    csv_path: str | Path = CSV_PATH,
    skip_existing: bool = True,
    output=print,
) -> int:
    """Populate the canonical card catalog into a database or database path."""

    owns_db = not isinstance(db_or_path, ResultsDB)
    db = (
        db_or_path
        if isinstance(db_or_path, ResultsDB)
        else ResultsDB(db_or_path)
    )
    source = Path(csv_path)
    try:
        count = db.conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
        if count > 0 and skip_existing:
            output(f"Cards table already has {count} entries. Skipping.")
            return 0
        if count > 0:
            raise ValueError(
                "cards table is not empty; refusing non-idempotent population"
            )

        # Read CSV, deduplicate by Card ID (take first row per card).
        seen: dict[int, dict] = {}
        with source.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
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

        for card in sorted(seen.values(), key=lambda value: value["card_id"]):
            db.add_card(**card)

        output(f"Populated {len(seen)} cards from {source.name}")
        return len(seen)
    finally:
        if owns_db:
            db.close()


def main():
    populate_cards()


if __name__ == "__main__":
    main()
