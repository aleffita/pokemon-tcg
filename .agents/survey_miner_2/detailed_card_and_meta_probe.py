import sqlite3
import csv
import json
from pathlib import Path

# Load SQLite cards
db_path = "file:model/results.db?mode=ro"
conn = sqlite3.connect(db_path, uri=True)
cursor = conn.cursor()
cursor.execute("SELECT id, name, category, stage, hp, energy_type, weakness, rule FROM cards")
cards_db = {r[0]: {"id": r[0], "name": r[1], "category": r[2], "stage": r[3], "hp": r[4], "type": r[5], "weakness": r[6], "rule": r[7]} for r in cursor.fetchall()}

# Load EN_Card_Data.csv
en_card_data_path = Path("public_agents/submissions/latest-submission-300elo/EN_Card_Data.csv")
en_card_info = {}
if en_card_data_path.exists():
    with open(en_card_data_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row.get("id", -1))
            if cid != -1:
                en_card_info[cid] = row

# Collect all card IDs used across all public decks and DB decks
with open(".agents/survey_miner_2/archetypes_full.json", "r") as f:
    data = json.load(f)

all_cids = set()
for a_name, a_data in data["public_agents"].items():
    for p in a_data["pokemon"] + a_data["trainers"] + a_data["energies"]:
        all_cids.add(p["id"])

for d_id, d_data in data["db_decks"].items():
    for p in d_data["pokemon"] + d_data["trainers"] + d_data["energies"]:
        all_cids.add(p["id"])

print(f"Total distinct cards across all archetypes: {len(all_cids)}")

card_dossier = {}
for cid in sorted(all_cids):
    c = cards_db.get(cid, {})
    en = en_card_info.get(cid, {})
    
    name = en.get("name") or c.get("name") or f"Card {cid}"
    cat = en.get("category") or c.get("category") or ""
    stage = en.get("stage") or c.get("stage") or ""
    hp = en.get("hp") or c.get("hp") or ""
    etype = en.get("energy_type") or c.get("type") or ""
    weakness = en.get("weakness") or c.get("weakness") or ""
    retreat = en.get("retreat_cost") or ""
    rule = en.get("rule") or c.get("rule") or ""
    ability_name = en.get("ability_name", "")
    ability_type = en.get("ability_type", "")
    ability_text = en.get("ability_text", "")
    attack1_name = en.get("attack1_name", "")
    attack1_cost = en.get("attack1_cost", "")
    attack1_dmg = en.get("attack1_damage", "")
    attack1_text = en.get("attack1_text", "")
    attack2_name = en.get("attack2_name", "")
    attack2_cost = en.get("attack2_cost", "")
    attack2_dmg = en.get("attack2_damage", "")
    attack2_text = en.get("attack2_text", "")
    
    card_dossier[cid] = {
        "id": cid,
        "name": name,
        "category": cat,
        "stage": stage,
        "hp": hp,
        "energy_type": etype,
        "weakness": weakness,
        "retreat": retreat,
        "rule": rule,
        "ability": f"{ability_name} ({ability_type}): {ability_text}".strip() if ability_name else "",
        "attack1": f"{attack1_name} [Cost: {attack1_cost}] -> {attack1_dmg} dmg. {attack1_text}".strip() if attack1_name else "",
        "attack2": f"{attack2_name} [Cost: {attack2_cost}] -> {attack2_dmg} dmg. {attack2_text}".strip() if attack2_name else ""
    }

with open(".agents/survey_miner_2/card_dossier.json", "w") as f:
    json.dump(card_dossier, f, indent=2)

print("Saved card_dossier.json successfully!")

# Let's print out the key Pokémon cards and their exact text:
key_pokemon_ids = [
    678, 677, 674, 673, 675, 676, # Mega Lucario, Riolu, Hariyama, Makuhita, Lunatone, Solrock
    121, 120, 119, # Dragapult ex, Drakloak, Dreepy
    140, 184, 1071, 235, # Fezandipiti ex, Latias ex, Meowth ex, Budew
    743, 742, 741, 66, 65, 305, 343, # Alakazam, Kadabra, Abra, Dudunsparce, Dunsparce, Shaymin
    96, 920, # Teal Mask Ogerpon ex, Tapu Bulu
    723, 722, 721, # Mega Abomasnow ex, Snover, Kyogre
    269, 271, 265, 268, 270, # Iono Bellibolt line
    28, 29 # Crustle line if any
]

print("\n=== KEY POKÉMON DOSSIER ===")
for kid in key_pokemon_ids:
    if kid in card_dossier:
        kd = card_dossier[kid]
        print(f"\n[{kd['id']}] {kd['name']} | {kd['stage']} | HP: {kd['hp']} | Type: {kd['energy_type']} | Weakness: {kd['weakness']} | Retreat: {kd['retreat']} | Rule: {kd['rule']}")
        if kd['ability']:
            print(f"   ABILITY: {kd['ability']}")
        if kd['attack1']:
            print(f"   ATTACK 1: {kd['attack1']}")
        if kd['attack2']:
            print(f"   ATTACK 2: {kd['attack2']}")

conn.close()
