"""Standalone manual migration script for model/results.db.

Ensures all decks have deterministic SHA-256 fingerprints and cleans up
duplicate deck entries idempotently without modifying results_db.py schema code.
"""
import sqlite3
from pathlib import Path
from rl.results_db import DB_PATH, canonical_deck_composition, deck_fingerprint


def migrate():
    db_file = DB_PATH
    if not db_file.is_file():
        print(f"No database found at {db_file}, nothing to migrate.")
        return

    print(f"Migrating database at {db_file}...")
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row

    # 1. Update fingerprints for decks missing fingerprint
    decks = conn.execute("SELECT id FROM decks WHERE fingerprint IS NULL").fetchall()
    updated_count = 0
    for (deck_id,) in decks:
        cards = conn.execute("SELECT card_id, quantity FROM deck_cards WHERE deck_id = ?", (deck_id,)).fetchall()
        if cards:
            fp = deck_fingerprint({r["card_id"]: r["quantity"] for r in cards})
            # Check if fingerprint already exists on another deck_id
            existing = conn.execute("SELECT id FROM decks WHERE fingerprint = ?", (fp,)).fetchone()
            if existing and existing["id"] != deck_id:
                # Merge deck_id into existing["id"]
                target_id = existing["id"]
                conn.execute("UPDATE OR IGNORE matches SET our_deck_id = ? WHERE our_deck_id = ?", (target_id, deck_id))
                conn.execute("UPDATE OR IGNORE matches SET opp_deck_id = ? WHERE opp_deck_id = ?", (target_id, deck_id))
                conn.execute("UPDATE OR IGNORE match_participants SET deck_id = ? WHERE deck_id = ?", (target_id, deck_id))
                conn.execute("DELETE FROM deck_cards WHERE deck_id = ?", (deck_id,))
                conn.execute("DELETE FROM decks WHERE id = ?", (deck_id,))
            else:
                conn.execute("UPDATE decks SET fingerprint = ? WHERE id = ?", (fp, deck_id))
            updated_count += 1

    conn.commit()
    conn.close()
    print(f"✓ Migration complete! Updated {updated_count} decks.")


if __name__ == "__main__":
    migrate()
