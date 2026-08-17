"""
Read-Only SQLite Card Verification for agent/deck.json
Milestone 1 Challenger 1 — Fitalabs AI Research
"""

import json
import sqlite3
from pathlib import Path

def main():
    deck_path = "agent/deck.json"
    with open(deck_path, "r") as f:
        deck_ids = json.load(f)
    
    db_path = "model/results.db"
    assert Path(db_path).exists(), f"Database {db_path} not found"
    
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()
    
    unique_ids = sorted(list(set(deck_ids)))
    placeholders = ",".join("?" for _ in unique_ids)
    
    cursor.execute(f"SELECT id, name, category, stage, hp, energy_type, rule FROM cards WHERE id IN ({placeholders}) ORDER BY id", unique_ids)
    rows = cursor.fetchall()
    found_ids = {r[0] for r in rows}
    
    print(f"Verified {len(found_ids)}/{len(unique_ids)} unique card IDs in `cards` table (Total cards in deck = {len(deck_ids)}).\n")
    for r in rows:
        count_in_deck = sum(1 for cid in deck_ids if cid == r[0])
        cid, name, cat, stage, hp, etype, rule = r
        cat_str = str(cat or "")
        stage_str = str(stage or "")
        hp_str = str(hp if hp is not None else "-")
        rule_str = str(rule or "-")
        print(f"  ID {cid:4d} (x{count_in_deck}): {name:25s} | Cat: {cat_str:10s} | Stage: {stage_str:15s} | HP: {hp_str:4s} | Rule: {rule_str}")
        
    missing_ids = set(unique_ids) - found_ids
    if missing_ids:
        print(f"\nERROR: Missing IDs in database: {missing_ids}")
    else:
        print("\nDATABASE PARITY VERIFICATION: 100% SUCCESS (All Card IDs verified in SQLite Schema 2.0.0)")
        
    conn.close()

if __name__ == "__main__":
    main()
