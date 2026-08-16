import os
import glob
import json
import csv
import sqlite3
from pathlib import Path
from collections import Counter, defaultdict

db_path = "file:model/results.db?mode=ro"
conn = sqlite3.connect(db_path, uri=True)
cursor = conn.cursor()

# 1. Load card database from SQLite
cursor.execute("SELECT id, name, category, stage, hp, energy_type, weakness, rule FROM cards")
cards_db = {}
for row in cursor.fetchall():
    cards_db[row[0]] = {
        "id": row[0],
        "name": row[1],
        "category": row[2],
        "stage": row[3],
        "hp": row[4],
        "energy_type": row[5],
        "weakness": row[6],
        "rule": row[7]
    }

# Also let's check if there's EN_Card_Data.csv to get attack text and rules if needed
en_card_data_path = Path("public_agents/submissions/latest-submission-300elo/EN_Card_Data.csv")
en_card_info = {}
if en_card_data_path.exists():
    with open(en_card_data_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row.get("id", -1))
            if cid != -1:
                en_card_info[cid] = row

print(f"Loaded {len(cards_db)} cards from SQLite, {len(en_card_info)} cards from EN_Card_Data.csv")

# 2. Decks in DB
cursor.execute("SELECT id, fingerprint, name, source, archetype, card_count FROM decks")
db_decks = cursor.fetchall()

print("\n=== ALL DECKS IN SQLITE ===")
for d in db_decks:
    print(f"ID={d[0]}, Archetype='{d[4]}', Name='{d[2]}', Source='{d[3]}', Count={d[5]}")

# 3. Read specific deck compositions from deck_cards in DB
deck_compositions = {}
for d in db_decks:
    did = d[0]
    cursor.execute("""
        SELECT dc.card_id, dc.quantity, c.name, c.category, c.stage, c.hp, c.energy_type, c.weakness, c.rule
        FROM deck_cards dc
        JOIN cards c ON dc.card_id = c.id
        WHERE dc.deck_id = ?
        ORDER BY c.category DESC, c.stage, c.hp DESC
    """, (did,))
    deck_compositions[did] = cursor.fetchall()

# 4. Check Public Agent deck CSVs
def parse_deck_csv(csv_path):
    card_ids = []
    with open(csv_path, 'r') as f:
        content = f.read().strip()
        lines = content.split('\n')
        # could be comma separated integers or single column
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('card_id') or line.startswith('id'):
                continue
            parts = [p.strip() for p in line.replace(',', ' ').split() if p.strip()]
            for p in parts:
                try:
                    cid = int(p)
                    card_ids.append(cid)
                except ValueError:
                    pass
    return card_ids

public_deck_files = {
    "lb1009_mega_lucario_ex_islet": "public_agents/lb1009_mega_lucario_ex_islet/deck.csv",
    "lb945_multiply_ivan": "public_agents/lb945_multiply_ivan/deck.csv",
    "lb826_alakazam_seok": "public_agents/lb826_alakazam_seok/deck.csv",
    "lb814_crustle_emre": "public_agents/lb814_crustle_emre/deck.csv",
    "lb798_lucario_pilkwang": "public_agents/lb798_lucario_pilkwang/deck.csv",
    "lb600_dragapult_ex": "public_agents/starters/lb600_dragapult_ex/deck.csv",
    "lb600_mega_lucario_ex": "public_agents/starters/lb600_mega_lucario_ex/deck.csv",
    "lb510_mega_abomasnow_ex": "public_agents/starters/lb510_mega_abomasnow_ex/deck.csv",
    "lb526_iono": "public_agents/starters/lb526_iono/deck.csv",
    "first_sub_kaggle_2707": "public_agents/submissions/first_sub_kaggle_2707/top_deck_first_sub_kaggle_2707_20260812_130140.csv",
    "fitalabs_hero_deck251": "public_agents/submissions/latest-submission-300elo/deck.csv",
    "agent_deck_csv": "agent/deck.csv"
}

print("\n=== PUBLIC AGENT DECK PROFILES ===")
for name, pth in public_deck_files.items():
    if os.path.exists(pth):
        cids = parse_deck_csv(pth)
        counts = Counter(cids)
        print(f"\n--- Archetype / Agent: {name} (Total cards: {len(cids)}, Unique: {len(counts)}) ---")
        
        # Categorize
        pokemon = []
        trainers = []
        energies = []
        
        for cid, qty in counts.items():
            cinfo = cards_db.get(cid, {})
            cname = cinfo.get("name", f"Unknown ({cid})")
            cat = cinfo.get("category", "Unknown")
            stage = cinfo.get("stage", "")
            hp = cinfo.get("hp", "")
            etype = cinfo.get("energy_type", "")
            weakness = cinfo.get("weakness", "")
            rule = cinfo.get("rule", "")
            
            entry = {
                "id": cid, "name": cname, "qty": qty, "category": cat,
                "stage": stage, "hp": hp, "type": etype, "weakness": weakness, "rule": rule
            }
            if cat == "Pokemon" or "hp" in cinfo and cinfo["hp"]:
                pokemon.append(entry)
            elif cat == "Trainer" or "Item" in str(cat) or "Supporter" in str(cat):
                trainers.append(entry)
            elif cat == "Energy" or "Energy" in cname:
                energies.append(entry)
            else:
                # fallback check
                if "Energy" in cname:
                    energies.append(entry)
                elif hp:
                    pokemon.append(entry)
                else:
                    trainers.append(entry)
        
        print("  POKEMON:")
        for p in sorted(pokemon, key=lambda x: (x['stage'] or '', x['hp'] or 0), reverse=True):
            print(f"    - {p['qty']}x [{p['id']}] {p['name']} (Stage: {p['stage']}, HP: {p['hp']}, Type: {p['type']}, Weakness: {p['weakness']}, Rule: {p['rule']})")
        print("  TRAINERS:")
        for t in sorted(trainers, key=lambda x: x['name']):
            print(f"    - {t['qty']}x [{t['id']}] {t['name']}")
        print("  ENERGIES:")
        for e in sorted(energies, key=lambda x: x['name']):
            print(f"    - {e['qty']}x [{e['id']}] {e['name']}")

conn.close()
