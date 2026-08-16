import json
import sqlite3
from collections import Counter

# Load card catalog
with open(".agents/survey_miner_2/card_catalog.json", "r") as f:
    catalog = json.load(f)

# Helper to get card info
def get_card(cid):
    return catalog.get(str(cid), {})

# Let's inspect the 6 panels:
# Panel 1: lb826_alakazam_seok (and variants lb881, lb966, lb1004)
# Panel 2: lb1009_mega_lucario_ex_islet and lb945_multiply_ivan
# Panel 3: lb814_crustle_emre, lb798_lucario_pilkwang, lb600_dragapult_ex
# Panel 4: first_sub_kaggle_2707, fitalabs_hero_deck251, agent_deck_csv
# Panel 5: lb510_mega_abomasnow_ex, lb526_iono
# Panel 6: Deck #633 Yan (Teal Mask Ogerpon ex), Deck #440 goonew

with open(".agents/survey_miner_2/archetypes_full.json", "r") as f:
    arch_data = json.load(f)

panels = {
    "Panel 1: Control & Energy Punishment (Alakazam / Hand Disruption)": {
        "primary": "lb826_alakazam_seok",
        "variants": ["lb881_alakazam_v1", "lb966_akalazam_v2_deck_updated", "lb1004_alakazam_v3"],
        "archetype_role": "Control / Hand & Energy Disruption / Damage Fixing",
        "leaderboard_elo": 826.0,
        "key_mechanics": "Xerosic's Machinations (forces discard to 3), Enhanced Hammer / Crushing Hammer (energy denial), Nighttime Mine (+1 retreat penalty), Alakazam Telekinesis (spread damage), Dudunsparce draw engine."
    },
    "Panel 2: Top Leaderboard Fast Aggro (Mega Lucario & Multipliers)": {
        "primary": "lb1009_mega_lucario_ex_islet",
        "variants": ["lb945_multiply_ivan"],
        "archetype_role": "Fast Burst Aggro / OHKO Sweeper",
        "leaderboard_elo": 1009.0,
        "key_mechanics": "Mega Lucario ex (340 HP, Stage 1, massive Fighting damage), Carmine (Turn 1 Supporter when going first), Lillie's Determination (draw 6/8), Fighting Gong + Premium Power Pro (damage booster), Solrock/Lunatone support."
    },
    "Panel 3: Spread Damage & Bench Snipes (Crustle, Lucario, Dragapult ex)": {
        "primary": "lb600_dragapult_ex",
        "variants": ["lb814_crustle_emre", "lb798_lucario_pilkwang", "lb600_mega_lucario_ex"],
        "archetype_role": "Spread Damage / Bench Sniping / Energy Acceleration",
        "leaderboard_elo": 814.0,
        "key_mechanics": "Dragapult ex Phantom Dive (200 active + 60 damage counters on bench), Crispin + Brock's Scouting (dual energy acceleration Fire/Psychic), Unfair Stamp (ACE SPEC hand disruption), Fezandipiti ex (revenge draw 3)."
    },
    "Panel 4: Internal Baselines & First Submission": {
        "primary": "first_sub_kaggle_2707",
        "variants": ["fitalabs_hero_deck251", "agent_deck_csv"],
        "archetype_role": "1-Prize Alakazam Stage 2 Engine / Single Prize Attrition",
        "leaderboard_elo": 270.7,
        "key_mechanics": "Alakazam (140 HP single-prize), Dawn + Hilda search engine, Rare Candy rapid Stage 2 setup, Enhanced Hammer, Fezandipiti ex + Shaymin tech."
    },
    "Panel 5: Heavy Tank Ramp (Mega Abomasnow ex & Iono Bellibolt ex)": {
        "primary": "lb510_mega_abomasnow_ex",
        "variants": ["lb526_iono"],
        "archetype_role": "Superheavy HP Wall / Mono-Energy Saturation / Fast Mega Evolution",
        "leaderboard_elo": 510.0,
        "key_mechanics": "Mega Abomasnow ex (350 HP, highest in meta, 34 Water Energy, Precious Trolley ACE SPEC, Surfing Beach), Iono's Bellibolt ex (280 HP, 22 Lightning Energy, Canari acceleration, Levincia stadium)."
    },
    "Panel 6: High Win-Rate Turbo Acceleration (Deck #633 Yan / Teal Mask Ogerpon ex & Deck #440 goonew)": {
        "primary": "Deck #633 Yan",
        "variants": ["Deck #440 goonew", "Deck #582", "Deck #775", "Deck #785", "Deck #873"],
        "archetype_role": "Turbo Energy Acceleration / Fast 2-Prize Sweeper / Judge Disruption",
        "leaderboard_elo": 1150.0,
        "key_mechanics": "Teal Mask Ogerpon ex (Teal Dance ability: attach Grass energy from hand, draw 1 card; Myriad Leaf Shower: 30 + 30 per energy attached to both active Pokémon), Tapu Bulu (140 HP 1-prize tech), Bug Catching Set + Tera Orb, 4x Judge hand reset."
    }
}

output_dossier = {}

for panel_key, p_info in panels.items():
    primary_name = p_info["primary"]
    
    # Get deck cards
    if primary_name in arch_data["public_agents"]:
        d_data = arch_data["public_agents"][primary_name]
    elif primary_name.startswith("Deck #") and "633" in primary_name:
        d_data = arch_data["db_decks"]["633"]
    elif "440" in primary_name:
        d_data = arch_data["db_decks"]["440"]
    else:
        d_data = None
        
    print(f"\n=======================================================")
    print(f" {panel_key.upper()} ")
    print(f" Primary Representative: {primary_name}")
    print(f" Role: {p_info['archetype_role']}")
    print(f" Elo: {p_info['leaderboard_elo']}")
    print(f" Key Mechanics: {p_info['key_mechanics']}")
    print(f"=======================================================")
    
    if d_data:
        pokemon_list = []
        trainers_list = []
        energies_list = []
        
        for p in d_data.get("pokemon", []):
            cid = p["id"]
            c_info = get_card(cid)
            pokemon_list.append({
                "id": cid,
                "qty": p["qty"],
                "name": c_info.get("name", p.get("name")),
                "stage": c_info.get("stage", p.get("stage")),
                "hp": c_info.get("hp", p.get("hp")),
                "type": c_info.get("type", p.get("type")),
                "weakness": c_info.get("weakness", p.get("weakness")),
                "retreat": c_info.get("retreat", p.get("retreat")),
                "rule": c_info.get("rule", p.get("rule")),
                "moves": c_info.get("moves", [])
            })
            
        for t in d_data.get("trainers", []):
            cid = t["id"]
            c_info = get_card(cid)
            trainers_list.append({
                "id": cid,
                "qty": t["qty"],
                "name": c_info.get("name", t.get("name")),
                "category": c_info.get("stage") or t.get("category"),
                "rule": c_info.get("rule", ""),
                "moves": c_info.get("moves", [])
            })
            
        for e in d_data.get("energies", []):
            cid = e["id"]
            c_info = get_card(cid)
            energies_list.append({
                "id": cid,
                "qty": e["qty"],
                "name": c_info.get("name", e.get("name")),
                "type": c_info.get("type", e.get("type")),
                "rule": c_info.get("rule", ""),
                "moves": c_info.get("moves", [])
            })
            
        output_dossier[panel_key] = {
            "meta": p_info,
            "pokemon": pokemon_list,
            "trainers": trainers_list,
            "energies": energies_list
        }

with open(".agents/survey_miner_2/panels_compiled.json", "w") as f:
    json.dump(output_dossier, f, indent=2)

print("\nSuccessfully compiled panels_compiled.json!")
