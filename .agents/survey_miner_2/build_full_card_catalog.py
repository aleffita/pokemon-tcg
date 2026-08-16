import csv
import json
from pathlib import Path
from collections import defaultdict

en_card_data_path = Path("public_agents/submissions/latest-submission-300elo/EN_Card_Data.csv")

cards_by_id = defaultdict(lambda: {
    "id": None, "name": "", "category": "", "stage": "", "hp": "", "type": "",
    "weakness": "", "resistance": "", "retreat": "", "rule": "", "moves": []
})

with open(en_card_data_path, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        raw_id = row.get("Card ID", "").strip()
        if not raw_id:
            continue
        cid = int(raw_id)
        c = cards_by_id[cid]
        c["id"] = cid
        c["name"] = row.get("Card Name", "").strip()
        c["category"] = row.get("Category", "").strip()
        c["stage"] = row.get("Stage (Pokémon)/Type (Energy and Trainer)", "").strip()
        c["hp"] = row.get("HP", "").strip()
        c["type"] = row.get("Type", "").strip()
        c["weakness"] = row.get("Weakness", "").strip()
        c["resistance"] = row.get("Resistance (Type)", "").strip()
        c["retreat"] = row.get("Retreat", "").strip()
        c["rule"] = row.get("Rule", "").strip()
        
        move_name = row.get("Move Name", "").strip()
        cost = row.get("Cost", "").strip()
        damage = row.get("Damage", "").strip()
        effect = row.get("Effect Explanation", "").strip()
        
        if move_name or effect:
            c["moves"].append({
                "name": move_name,
                "cost": cost,
                "damage": damage,
                "effect": effect
            })

print(f"Loaded {len(cards_by_id)} unique cards from EN_Card_Data.csv")

with open(".agents/survey_miner_2/card_catalog.json", "w") as f:
    json.dump(cards_by_id, f, indent=2)

print("Saved card_catalog.json!")
