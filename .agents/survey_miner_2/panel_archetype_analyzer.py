import json
import sqlite3
from collections import Counter

# Load archetypes_full.json
with open(".agents/survey_miner_2/archetypes_full.json", "r") as f:
    data = json.load(f)

public_agents = data["public_agents"]
db_decks = data["db_decks"]

print("================================================================================")
print("             DEEP DIVE: 6 OPPONENT PANEL ARCHETYPES AUDIT                      ")
print("================================================================================")

def print_deck_profile(title, d):
    print(f"\n================================================================================")
    print(f" ARCHETYPE PROFILE: {title}")
    print(f" Total Cards: {d.get('total_cards', d.get('card_count'))} | Pokemon: {d['pokemon_count']} (Basics: {d['basic_count']}, ex/Mega: {d['ex_count']}) | Trainers: {d['trainer_count']} | Energies: {d['energy_count']}")
    print(f"================================================================================")
    
    print("\n--- POKÉMON ROSTER ---")
    for p in d['pokemon']:
        hp_str = f"HP: {p['hp']}" if p['hp'] else "HP: N/A"
        type_str = f"Type: {p['type']}" if p['type'] else ""
        weak_str = f"Weakness: {p['weakness']}" if p['weakness'] else "Weakness: None"
        rule_str = f"Rule: {p['rule']}" if p['rule'] else ""
        ret_str = f"Retreat: {p['retreat']}" if p.get('retreat') else ""
        print(f"  [{p['qty']}x] (ID {p['id']}) {p['name']} | {p['stage']} | {hp_str} | {type_str} | {weak_str} | {rule_str} | {ret_str}")
        if p.get('ability'):
            print(f"      Ability: {p['ability']}")
        if p.get('attack1'):
            print(f"      Attack 1: {p['attack1']}")
        if p.get('attack2'):
            print(f"      Attack 2: {p['attack2']}")

    print("\n--- TRAINERS ROSTER ---")
    for t in d['trainers']:
        print(f"  [{t['qty']}x] (ID {t['id']}) {t['name']} ({t['category']})")

    print("\n--- ENERGIES ROSTER ---")
    for e in d['energies']:
        print(f"  [{e['qty']}x] (ID {e['id']}) {e['name']} ({e['category']})")

# Let's inspect the 6 panels in detail:
print("\n>>> PANEL 1: CONTROL / ENERGY PUNISHMENT / ALAKAZAM / MIMIKYU / HAND DISRUPTION")
print_deck_profile("lb826_alakazam_seok", public_agents["lb826_alakazam_seok"])

print("\n>>> PANEL 2: FAST AGGRO (TOP LEADERBOARD)")
print_deck_profile("lb1009_mega_lucario_ex_islet", public_agents["lb1009_mega_lucario_ex_islet"])
print_deck_profile("lb945_multiply_ivan", public_agents["lb945_multiply_ivan"])

print("\n>>> PANEL 3: SPREAD DAMAGE / BENCH SNIPES / ENERGY ACCELERATION")
print_deck_profile("lb814_crustle_emre", public_agents["lb814_crustle_emre"])
print_deck_profile("lb600_dragapult_ex", public_agents["lb600_dragapult_ex"])
print_deck_profile("lb798_lucario_pilkwang", public_agents["lb798_lucario_pilkwang"])

print("\n>>> PANEL 4: INTERNAL BASELINES & FIRST_SUB_KAGGLE_2707")
print_deck_profile("first_sub_kaggle_2707", public_agents["first_sub_kaggle_2707"])
print_deck_profile("fitalabs_hero_deck251", public_agents["fitalabs_hero_deck251"])

print("\n>>> PANEL 5: TANK RAMP / MEGA ABOMASNOW & IONO BELLIBOLT")
print_deck_profile("lb510_mega_abomasnow_ex", public_agents["lb510_mega_abomasnow_ex"])
print_deck_profile("lb526_iono", public_agents["lb526_iono"])

print("\n>>> PANEL 6: DECK #633 (YAN / TEAL MASK OGERPON EX) & DB TOP DECKS")
if "633" in db_decks:
    print_deck_profile("Deck #633 Yan (Teal Mask Ogerpon ex)", db_decks["633"])
if "440" in db_decks:
    print_deck_profile("Deck #440 goonew", db_decks["440"])
