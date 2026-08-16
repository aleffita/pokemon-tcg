import os
import glob
import json
import sqlite3
from pathlib import Path

db_path = "file:model/results.db?mode=ro"
conn = sqlite3.connect(db_path, uri=True)
cursor = conn.cursor()

# Get cards map
cursor.execute("SELECT id, name, category, stage, hp, energy_type, weakness, rule FROM cards")
cards_map = {}
for r in cursor.fetchall():
    cards_map[r[0]] = {
        "id": r[0],
        "name": r[1],
        "category": r[2],
        "stage": r[3],
        "hp": r[4],
        "energy_type": r[5],
        "weakness": r[6],
        "rule": r[7]
    }

print(f"Total cards in cards table: {len(cards_map)}")

# Scan public_agents directories
agent_dirs = sorted(glob.glob("public_agents/**", recursive=True))
print("\n--- Scanning public_agents for deck files and main.py ---")

panels = {}

for ad in sorted(glob.glob("public_agents/*") + glob.glob("public_agents/*/*")):
    p = Path(ad)
    if not p.is_dir():
        continue
    deck_files = list(p.glob("*.json")) + list(p.glob("*.csv")) + list(p.glob("deck*"))
    py_files = list(p.glob("*.py"))
    if deck_files or py_files:
        print(f"\nDirectory: {p}")
        for df in deck_files:
            print(f"  Deck file: {df.name} (size: {df.stat().st_size} bytes)")
        for pf in py_files:
            print(f"  Py file: {pf.name} (size: {pf.stat().st_size} bytes)")

conn.close()
