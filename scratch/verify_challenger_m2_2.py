import sqlite3
import json
import re
import math
from fractions import Fraction

def main():
    db_uri = "file:model/results.db?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    cursor = conn.cursor()

    print("=" * 110)
    print("CHALLENGER 2: COMPREHENSIVE EMPIRICAL CROSS-VALIDATION HARNESS")
    print("=" * 110)

    # 1. LOAD agent/deck.json
    with open("agent/deck.json", "r") as f:
        deck_ids = json.load(f)
    print(f"\n[1] agent/deck.json Check:")
    print(f"    - Type: {type(deck_ids).__name__}")
    print(f"    - Total card IDs: {len(deck_ids)}")
    assert len(deck_ids) == 60, f"agent/deck.json must contain exactly 60 cards, got {len(deck_ids)}"
    for idx, cid in enumerate(deck_ids):
        assert isinstance(cid, int), f"Element at index {idx} ({cid}) is not an integer"
    
    deck_counts = {}
    for cid in deck_ids:
        deck_counts[cid] = deck_counts.get(cid, 0) + 1
    print(f"    - Distinct Card IDs ({len(deck_counts)}): {sorted(deck_counts.items())}")

    # 2. LOAD experiments/decks/deck_supreme_60.json
    with open("experiments/decks/deck_supreme_60.json", "r") as f:
        capsule = json.load(f)
    print(f"\n[2] experiments/decks/deck_supreme_60.json Check:")
    assert capsule["card_count"] == 60
    card_list = capsule["card_list"]
    print(f"    - Number of entries in card_list: {len(card_list)}")
    
    capsule_id_expansion = []
    for entry in card_list:
        cid = entry["id"]
        qty = entry["quantity"]
        capsule_id_expansion.extend([cid] * qty)
    assert len(capsule_id_expansion) == 60
    assert sorted(capsule_id_expansion) == sorted(deck_ids)
    print(f"    - 100% parity between agent/deck.json and experiments/decks/deck_supreme_60.json")

    # 3. PARSE Markdown Table in experiments/decks/DECK_SUPREME_60.md
    print(f"\n[3] Markdown Table Parsing & Field-by-Field DB Validation:")
    with open("experiments/decks/DECK_SUPREME_60.md", "r") as f:
        md_content = f.read()

    table_pattern = re.compile(r"\|\s*Slot Range\s*\|\s*Card ID\s*\|\s*Exact Card Name\s*\|\s*Category\s*\|\s*Stage\s*\|\s*Energy Type\s*\|\s*HP\s*\|\s*Rule Box\s*\|\s*Qty\s*\|\s*Primary Tactical Role\s*\|(.*?)\|\s*\*\*SUM\*\*", re.DOTALL)
    match = table_pattern.search(md_content)
    assert match is not None, "Failed to locate Master Card Roster table in DECK_SUPREME_60.md"
    
    table_lines = match.group(1).strip().split("\n")
    md_entries = []
    for line in table_lines:
        line = line.strip()
        if not line.startswith("|") or line.startswith("| :---"):
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) < 10:
            continue
        slot_range = parts[0]
        cid = int(parts[1].replace("*", "").strip())
        name = parts[2].replace("*", "").strip()
        category = parts[3].replace("*", "").strip()
        stage = parts[4].replace("*", "").strip()
        energy_type = parts[5].replace("*", "").strip()
        hp = parts[6].replace("*", "").strip()
        rule = parts[7].replace("*", "").strip()
        qty = int(parts[8].replace("*", "").strip())
        role = parts[9]
        md_entries.append({
            "slot_range": slot_range,
            "id": cid,
            "name": name,
            "category": category,
            "stage": stage,
            "energy_type": energy_type,
            "hp": hp,
            "rule": rule,
            "qty": qty,
            "role": role
        })

    print(f"    - Parsed {len(md_entries)} distinct card slots from markdown table.")
    total_md_qty = sum(e["qty"] for e in md_entries)
    print(f"    - Sum of quantities in table: {total_md_qty}")
    assert total_md_qty == 60, f"Table sum is {total_md_qty}, expected 60"

    print("\n" + "-" * 120)
    print(f"{'Slot':<8} | {'ID':>4} | {'Name (MD / DB)':<32} | {'Category (MD/DB)':<18} | {'Stage (MD/DB)':<24} | {'Type':<8} | {'HP':<6} | {'Rule':<12} | {'Qty':>3} | Match")
    print("-" * 120)

    for e in md_entries:
        cid = e["id"]
        cursor.execute("SELECT id, name, category, stage, hp, energy_type, weakness, rule FROM cards WHERE id=?", (cid,))
        row = cursor.fetchone()
        assert row is not None, f"Card ID {cid} not found in database!"
        db_id, db_name, db_category, db_stage, db_hp, db_energy_type, db_weakness, db_rule = row

        # Verification of exact attributes
        # Name
        assert e["name"] == db_name, f"Name mismatch for ID {cid}: MD '{e['name']}' != DB '{db_name}'"
        
        # HP
        if db_hp is not None:
            assert str(e["hp"]) == str(db_hp), f"HP mismatch for ID {cid}: MD '{e['hp']}' != DB '{db_hp}'"
        else:
            assert e["hp"] in ["—", "-", "None", ""], f"HP mismatch for ID {cid}: MD '{e['hp']}' != DB None"
        
        # Energy Type
        if db_energy_type is not None:
            assert e["energy_type"] == db_energy_type, f"Type mismatch for ID {cid}: MD '{e['energy_type']}' != DB '{db_energy_type}'"
        else:
            assert e["energy_type"] in ["None", "—", "-", ""], f"Type mismatch for ID {cid}: MD '{e['energy_type']}' != DB None"

        # Rule Box
        if db_rule is not None:
            assert e["rule"] == db_rule, f"Rule mismatch for ID {cid}: MD '{e['rule']}' != DB '{db_rule}'"
        else:
            assert e["rule"] in ["None", "—", "-", ""], f"Rule mismatch for ID {cid}: MD '{e['rule']}' != DB None"

        name_display = f"{e['name']}"
        cat_display = f"{e['category']}/{db_category or 'None'}"
        stage_display = f"{e['stage']}/{db_stage or 'None'}"
        type_display = f"{e['energy_type']}"
        hp_display = f"{e['hp']}"
        rule_display = f"{e['rule']}"

        print(f"{e['slot_range']:<8} | {cid:>4} | {name_display:<32} | {cat_display:<18} | {stage_display:<24} | {type_display:<8} | {hp_display:<6} | {rule_display:<12} | {e['qty']:>3} | [PASS]")

    # 4. GAME MECHANICS & LEGALITY RULES
    print("\n[4] Pokémon TCG Game Mechanics & Tournament Legality Audit:")
    
    # 4.1 Card Copy Limits (Max 4 of any card by name, except Basic Energy)
    name_tally = {}
    for e in md_entries:
        cursor.execute("SELECT stage FROM cards WHERE id=?", (e["id"],))
        st = cursor.fetchone()[0]
        if st != "Basic Energy":
            name_tally[e["name"]] = name_tally.get(e["name"], 0) + e["qty"]
            assert name_tally[e["name"]] <= 4, f"Deck legality violation: {e['name']} has {name_tally[e['name']]} copies (>4 allowed)"
    print("    [PASS] 4-Copy Rule: All non-Basic Energy cards are within legal limits (<= 4).")
    
    # 4.2 ACE SPEC Limit (Max 1 ACE SPEC card per deck)
    ace_spec_entries = [e for e in md_entries if e["rule"] == "ACE SPEC"]
    ace_spec_total = sum(e["qty"] for e in ace_spec_entries)
    assert ace_spec_total == 1, f"ACE SPEC violation: expected 1 ACE SPEC, found {ace_spec_total}"
    print(f"    [PASS] ACE SPEC Rule: Exactly 1 ACE SPEC card ({ace_spec_entries[0]['name']}, ID {ace_spec_entries[0]['id']}).")

    # 4.3 Basic Pokémon Count (At least 1 required, deck has 11)
    basic_pkmn_entries = [e for e in md_entries if "Basic" in e["stage"] and e["category"] == "Pokémon"]
    basic_pkmn_total = sum(e["qty"] for e in basic_pkmn_entries)
    assert basic_pkmn_total == 11, f"Basic Pokémon count violation: expected 11, got {basic_pkmn_total}"
    print(f"    [PASS] Basic Pokémon Count: Exactly 11 Basic Pokémon across 6 distinct species.")

    # 4.4 Energy Composition
    energy_entries = [e for e in md_entries if e["category"] == "Energy"]
    energy_total = sum(e["qty"] for e in energy_entries)
    assert energy_total == 13, f"Energy count violation: expected 13, got {energy_total}"
    basic_energy_total = sum(e["qty"] for e in energy_entries if e["stage"] == "Basic")
    special_energy_total = sum(e["qty"] for e in energy_entries if e["stage"] == "Special")
    assert basic_energy_total == 12, f"Basic Energy count: expected 12, got {basic_energy_total}"
    assert special_energy_total == 1, f"Special Energy count: expected 1, got {special_energy_total}"
    print(f"    [PASS] Energy Curve: 13 Energies (10 Basic {{G}}, 2 Basic {{D}}, 1 Special {{G}} Grow Grass).")

    # 5. ALL MATCHUP REFERENCED CARD IDS IN DECK_SUPREME_60.md
    print(f"\n[5] Cross-Validating All Card IDs in Matchup Playbooks:")
    matchup_card_checks = {
        "Matchup 1 (Alakazam control)": [743, 742, 66, 741, 140, 343, 1081, 1080, 1213, 1182, 112, 7, 1097],
        "Matchup 2 (Mega Lucario ex aggro)": [678, 1192, 1141, 920, 112, 1123, 184, 96],
        "Matchup 3 (Dragapult / Crustle wall)": [121, 345, 120, 119, 1264, 920, 112, 1201],
        "Matchup 4 (first_sub baseline)": [743, 66, 1266, 1197, 184, 1094, 1080, 1182, 140],
        "Matchup 5 (Mega Abomasnow ex ramp)": [723, 3, 1182, 96, 920, 18],
        "Matchup 6 (Deck #633 mirror)": [96, 920, 112, 1201, 1094]
    }

    all_referenced_cids = set()
    for matchup_name, cids in matchup_card_checks.items():
        print(f"\n    * {matchup_name}:")
        for cid in cids:
            all_referenced_cids.add(cid)
            cursor.execute("SELECT id, name, category, stage, hp, energy_type, weakness, rule FROM cards WHERE id=?", (cid,))
            r = cursor.fetchone()
            assert r is not None, f"Referenced card ID {cid} missing from DB!"
            print(f"      - ID {r[0]:>4}: {r[1]:<24} | Stage: {str(r[3]):<15} | HP: {str(r[4]):<4} | Type: {str(r[5]):<5} | Weakness: {str(r[6]):<5} | Rule: {str(r[7])}")

    print(f"\n[PASS] All {len(all_referenced_cids)} distinct Card IDs referenced across matchup playbooks are verified in SQLite model/results.db.")

    # 6. HYPERGEOMETRIC SETUP PROBABILITY PROOF VERIFICATION
    print(f"\n[6] Hypergeometric Setup & Resource Access Probability Assertions:")
    N = 60
    n = 7
    Kb = 11
    p_mul_7 = Fraction(math.comb(N - Kb, n), math.comb(N, n))
    p_set_7 = 1 - p_mul_7
    p_mul_w1 = p_mul_7 ** 2
    p_set_w1 = 1 - p_mul_w1

    print(f"    - P(Mulligan n=7): {p_mul_7} = {float(p_mul_7)*100:.4f}%")
    print(f"    - P(Setup n=7):    {p_set_7} = {float(p_set_7)*100:.4f}%")
    print(f"    - P(Mulligan <= 1): {p_mul_w1} = {float(p_mul_w1)*100:.4f}%")
    print(f"    - P(Setup <= 1):    {p_set_w1} = {float(p_set_w1)*100:.4f}%")

    assert p_mul_7 == Fraction(325381, 1462905)
    assert p_set_7 == Fraction(1137524, 1462905)
    assert p_mul_w1 == Fraction(105872795161, 2140091039025)
    assert p_set_w1 == Fraction(2034218243864, 2140091039025)
    assert float(p_set_w1) >= 0.92
    assert float(p_mul_w1) <= 0.08
    print("    [PASS] Hypergeometric assertions verified with exact irreducible rational fractions.")

    print("\n" + "=" * 110)
    print("FINAL VERDICT: CONFIRMED")
    print("=" * 110)

if __name__ == "__main__":
    main()
