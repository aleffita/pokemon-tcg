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

# Load cards
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

# Read deck files
def parse_deck_csv(csv_path):
    card_ids = []
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, 'r') as f:
        content = f.read().strip()
        lines = content.split('\n')
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

archetype_summaries = {}

for name, pth in public_deck_files.items():
    cids = parse_deck_csv(pth)
    if not cids:
        continue
    counts = Counter(cids)
    
    pokemon = []
    trainers = []
    energies = []
    
    total_hp = 0
    basic_count = 0
    ex_count = 0
    energy_count = 0
    trainer_count = 0
    
    for cid, qty in counts.items():
        cinfo = cards_db.get(cid, {})
        en = en_card_info.get(cid, {})
        cname = cinfo.get("name", en.get("name", f"Card {cid}"))
        cat = cinfo.get("category", en.get("category", "Unknown"))
        stage = cinfo.get("stage", en.get("stage", ""))
        hp = cinfo.get("hp") or (int(en["hp"]) if en.get("hp") and en["hp"].isdigit() else None)
        etype = cinfo.get("type", en.get("energy_type", ""))
        weakness = cinfo.get("weakness", en.get("weakness", ""))
        rule = cinfo.get("rule", en.get("rule", ""))
        
        # Details from EN_Card_Data
        ability_name = en.get("ability_name", "")
        ability_text = en.get("ability_text", "")
        attack1_name = en.get("attack1_name", "")
        attack1_cost = en.get("attack1_cost", "")
        attack1_dmg = en.get("attack1_damage", "")
        attack1_text = en.get("attack1_text", "")
        attack2_name = en.get("attack2_name", "")
        attack2_cost = en.get("attack2_cost", "")
        attack2_dmg = en.get("attack2_damage", "")
        attack2_text = en.get("attack2_text", "")
        retreat_cost = en.get("retreat_cost", "")
        
        entry = {
            "id": cid, "name": cname, "qty": qty, "category": cat,
            "stage": stage, "hp": hp, "type": etype, "weakness": weakness, "rule": rule,
            "ability": f"{ability_name}: {ability_text}" if ability_name else "",
            "attack1": f"{attack1_name} ({attack1_cost}) -> {attack1_dmg} dmg. {attack1_text}".strip() if attack1_name else "",
            "attack2": f"{attack2_name} ({attack2_cost}) -> {attack2_dmg} dmg. {attack2_text}".strip() if attack2_name else "",
            "retreat": retreat_cost
        }
        
        if cat == "Pokemon" or (hp and hp > 0) or stage:
            pokemon.append(entry)
            if hp:
                total_hp += hp * qty
            if "Basic" in str(stage):
                basic_count += qty
            if "ex" in str(rule) or "ex" in cname or "Mega" in str(rule):
                ex_count += qty
        elif "Energy" in cat or "Energy" in cname:
            energies.append(entry)
            energy_count += qty
        else:
            trainers.append(entry)
            trainer_count += qty
            
    archetype_summaries[name] = {
        "file": pth,
        "total_cards": len(cids),
        "unique_cards": len(counts),
        "pokemon_count": sum(p['qty'] for p in pokemon),
        "basic_count": basic_count,
        "ex_count": ex_count,
        "trainer_count": trainer_count,
        "energy_count": energy_count,
        "pokemon": pokemon,
        "trainers": trainers,
        "energies": energies
    }

# Also let's inspect decks from DB (like Deck #633 Yan, Deck #251, #440, #582, #775, #785, #873)
cursor.execute("SELECT id, name, archetype, source, card_count FROM decks WHERE id IN (633, 251, 440, 582, 775, 785, 873)")
target_db_decks = cursor.fetchall()

db_deck_summaries = {}
for did, dname, darchetype, dsource, dcount in target_db_decks:
    cursor.execute("""
        SELECT dc.card_id, dc.quantity
        FROM deck_cards dc
        WHERE dc.deck_id = ?
    """, (did,))
    dcards = cursor.fetchall()
    
    pokemon = []
    trainers = []
    energies = []
    basic_count = 0
    ex_count = 0
    
    for cid, qty in dcards:
        cinfo = cards_db.get(cid, {})
        en = en_card_info.get(cid, {})
        cname = cinfo.get("name", en.get("name", f"Card {cid}"))
        cat = cinfo.get("category", en.get("category", "Unknown"))
        stage = cinfo.get("stage", en.get("stage", ""))
        hp = cinfo.get("hp") or (int(en["hp"]) if en.get("hp") and en["hp"].isdigit() else None)
        etype = cinfo.get("type", en.get("energy_type", ""))
        weakness = cinfo.get("weakness", en.get("weakness", ""))
        rule = cinfo.get("rule", en.get("rule", ""))
        
        ability_name = en.get("ability_name", "")
        ability_text = en.get("ability_text", "")
        attack1_name = en.get("attack1_name", "")
        attack1_cost = en.get("attack1_cost", "")
        attack1_dmg = en.get("attack1_damage", "")
        attack1_text = en.get("attack1_text", "")
        attack2_name = en.get("attack2_name", "")
        attack2_cost = en.get("attack2_cost", "")
        attack2_dmg = en.get("attack2_damage", "")
        attack2_text = en.get("attack2_text", "")
        retreat_cost = en.get("retreat_cost", "")
        
        entry = {
            "id": cid, "name": cname, "qty": qty, "category": cat,
            "stage": stage, "hp": hp, "type": etype, "weakness": weakness, "rule": rule,
            "ability": f"{ability_name}: {ability_text}" if ability_name else "",
            "attack1": f"{attack1_name} ({attack1_cost}) -> {attack1_dmg} dmg. {attack1_text}".strip() if attack1_name else "",
            "attack2": f"{attack2_name} ({attack2_cost}) -> {attack2_dmg} dmg. {attack2_text}".strip() if attack2_name else "",
            "retreat": retreat_cost
        }
        
        if cat == "Pokemon" or (hp and hp > 0) or stage:
            pokemon.append(entry)
            if "Basic" in str(stage):
                basic_count += qty
            if "ex" in str(rule) or "ex" in cname or "Mega" in str(rule):
                ex_count += qty
        elif "Energy" in cat or "Energy" in cname:
            energies.append(entry)
        else:
            trainers.append(entry)
            
    db_deck_summaries[did] = {
        "id": did,
        "name": dname,
        "archetype": darchetype,
        "source": dsource,
        "card_count": dcount,
        "pokemon_count": sum(p['qty'] for p in pokemon),
        "basic_count": basic_count,
        "ex_count": ex_count,
        "trainer_count": sum(t['qty'] for t in trainers),
        "energy_count": sum(e['qty'] for e in energies),
        "pokemon": pokemon,
        "trainers": trainers,
        "energies": energies
    }

# Save extracted JSON for analysis
with open(".agents/survey_miner_2/archetypes_full.json", "w") as f:
    json.dump({
        "public_agents": archetype_summaries,
        "db_decks": db_deck_summaries
    }, f, indent=2)

print("Saved archetypes_full.json successfully!")
conn.close()
