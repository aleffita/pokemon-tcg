"""
Comprehensive SQLite Cross-Validation & Integrity Audit Script
Milestone 1 — Challenger 2
"""
import sys
import json
import sqlite3
from collections import Counter

def run_comprehensive_validation():
    print("=" * 90)
    print("SQLITE CROSS-VALIDATION & CARD INTEGRITY AUDIT — MILESTONE 1 (CHALLENGER 2)")
    print("=" * 90)

    # 1. Load agent/deck.json
    deck_json_path = "agent/deck.json"
    with open(deck_json_path, "r", encoding="utf-8") as f:
        agent_deck = json.load(f)
    print(f"[*] Loaded {deck_json_path} with {len(agent_deck)} card IDs.")

    # 2. Load experiments/decks/deck_supreme_60.json
    supreme_path = "experiments/decks/deck_supreme_60.json"
    with open(supreme_path, "r", encoding="utf-8") as f:
        supreme_data = json.load(f)
    supreme_cards = supreme_data.get("card_list", [])
    print(f"[*] Loaded {supreme_path} with {len(supreme_cards)} unique card definitions.")

    # 3. Read-only SQLite connection
    db_path = "file:model/results.db?mode=ro"
    conn = sqlite3.connect(db_path, uri=True)
    cursor = conn.cursor()
    print(f"[*] SQLite connection established (URI: {db_path})")

    # Fetch all cards from SQLite
    cursor.execute("SELECT id, name, category, stage, hp, energy_type, weakness, rule, metadata_complete FROM cards")
    db_cards = {
        row[0]: {
            "id": row[0],
            "name": row[1],
            "category": row[2],
            "stage": row[3],
            "hp": row[4],
            "energy_type": row[5],
            "weakness": row[6],
            "rule": row[7],
            "metadata_complete": row[8]
        }
        for row in cursor.fetchall()
    }
    print(f"[*] Total cards indexed in SQLite results.db: {len(db_cards)}")

    # -------------------------------------------------------------------------
    # TEST 1: agent/deck.json Exact Size & Integer Type Verification
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("TEST 1: agent/deck.json Count & Type Validation")
    print("=" * 90)
    deck_len = len(agent_deck)
    all_integers = all(isinstance(x, int) for x in agent_deck)
    print(f"Deck card count: {deck_len} (Required: 60)")
    print(f"All IDs integer: {all_integers}")
    assert deck_len == 60, f"Deck size {deck_len} != 60"
    assert all_integers, "Non-integer card ID found in deck.json"
    print("-> TEST 1 PASSED: Exactly 60 integer Card IDs in agent/deck.json.")

    # -------------------------------------------------------------------------
    # TEST 2: Relational Integrity against `cards` table in SQLite
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("TEST 2: Relational Foreign Key Integrity (Every ID exists in `cards` table)")
    print("=" * 90)
    missing_db_ids = [cid for cid in set(agent_deck) if cid not in db_cards]
    print(f"Unique Card IDs in deck: {len(set(agent_deck))}")
    print(f"Missing in SQLite:       {missing_db_ids}")
    assert len(missing_db_ids) == 0, f"Missing Card IDs in SQLite: {missing_db_ids}"
    print("-> TEST 2 PASSED: 100% of Card IDs resolve to valid records in model/results.db.")

    # -------------------------------------------------------------------------
    # TEST 3: Exact Quantity & Card ID Parity (agent/deck.json vs deck_supreme_60.json)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("TEST 3: Quantity & ID Parity between agent/deck.json and deck_supreme_60.json")
    print("=" * 90)
    agent_counter = Counter(agent_deck)
    supreme_counter = {c["id"]: c["quantity"] for c in supreme_cards}
    supreme_sum = sum(c["quantity"] for c in supreme_cards)

    print(f"Summed quantity in deck_supreme_60.json: {supreme_sum} (Required: 60)")
    assert supreme_sum == 60, f"deck_supreme_60.json summed quantity {supreme_sum} != 60"

    print(f"Unique cards in deck_supreme_60.json:   {len(supreme_counter)}")
    print(f"Unique cards in agent/deck.json:        {len(agent_counter)}")

    assert agent_counter == supreme_counter, "Quantity mismatch between agent/deck.json and deck_supreme_60.json"
    print("-> TEST 3 PASSED: Exact 1-to-1 card ID and quantity parity verified across both artifacts.")

    # -------------------------------------------------------------------------
    # TEST 4: 100% Metadata Cross-Validation (Names, HP, Types, Stages, Rules)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("TEST 4: Field-by-Field Metadata Parity (SQLite vs deck_supreme_60.json)")
    print("=" * 90)
    print(f"{'ID':<5} | {'Card Name':<24} | {'DB Stage':<15} | {'DB Type':<7} | {'DB HP':<5} | {'DB Rule':<12} | {'Qty':<3} | {'Parity'}")
    print("-" * 90)

    for c in supreme_cards:
        cid = c["id"]
        db = db_cards[cid]

        # Verify Name
        assert c["name"].strip() == (db["name"] or "").strip(), f"Name mismatch ID {cid}: {c['name']} vs {db['name']}"
        # Verify Stage
        assert c["stage"] == db["stage"], f"Stage mismatch ID {cid}: {c['stage']} vs {db['stage']}"
        # Verify Type (in json 'type', in db 'energy_type')
        assert c["type"] == db["energy_type"], f"Type mismatch ID {cid}: {c['type']} vs {db['energy_type']}"
        # Verify HP
        assert c["hp"] == db["hp"], f"HP mismatch ID {cid}: {c['hp']} vs {db['hp']}"
        # Verify Rule box
        assert c["rule"] == db["rule"], f"Rule mismatch ID {cid}: {c['rule']} vs {db['rule']}"
        # Verify Metadata Complete flag
        assert db["metadata_complete"] == 1, f"Metadata incomplete for ID {cid}"

        print(f"{cid:<5} | {db['name']:<24} | {str(db['stage']):<15} | {str(db['energy_type']):<7} | {str(db['hp']):<5} | {str(db['rule']):<12} | {c['quantity']:<3} | VERIFIED")

    print("\n-> TEST 4 PASSED: 100% metadata parity confirmed across all 24 unique card entries.")

    # -------------------------------------------------------------------------
    # TEST 5: 4-Copy Limit Rule Enforcement (Except Basic Energy)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("TEST 5: Deck Construction Invariant — Max 4 Copies (Except Basic Energy)")
    print("=" * 90)
    for cid, count in agent_counter.items():
        db = db_cards[cid]
        is_basic_energy = (db["stage"] == "Basic Energy")
        if is_basic_energy:
            print(f"ID {cid:>4} | {db['name']:<25} | Copies: {count:<2} (Basic Energy: UNLIMITED) -> OK")
        else:
            print(f"ID {cid:>4} | {db['name']:<25} | Copies: {count:<2} (Limit <= 4)            -> OK")
            assert count <= 4, f"Card {db['name']} (ID {cid}) exceeds 4 copies: {count}"

    print("-> TEST 5 PASSED: Standard format copy rules strictly respected.")

    # -------------------------------------------------------------------------
    # TEST 6: ACE SPEC Invariant Enforcement (Exactly 1, ID 1080 `Unfair Stamp`)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("TEST 6: Deck Construction Invariant — Exactly 1 ACE SPEC Card")
    print("=" * 90)
    ace_specs = [
        (cid, db_cards[cid]["name"], agent_counter[cid])
        for cid in agent_counter
        if db_cards[cid]["rule"] == "ACE SPEC"
    ]
    print(f"ACE SPEC cards detected in deck: {ace_specs}")
    assert len(ace_specs) == 1, f"Expected exactly 1 ACE SPEC card type, found {len(ace_specs)}"
    assert ace_specs[0][0] == 1080, f"Expected ACE SPEC ID 1080 ('Unfair Stamp'), got ID {ace_specs[0][0]}"
    assert ace_specs[0][2] == 1, f"Expected exactly 1 copy of ACE SPEC, got {ace_specs[0][2]}"
    print("-> TEST 6 PASSED: Exactly 1 ACE SPEC card present (Unfair Stamp, ID 1080, quantity 1).")

    # -------------------------------------------------------------------------
    # TEST 7: Macro Category Breakdown & Energy Curve Validation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("TEST 7: Macro Composition & Energy Curve Cross-Validation")
    print("=" * 90)
    pokemon_count = sum(c["quantity"] for c in supreme_cards if "Pokémon" in c["stage"])
    item_count = sum(c["quantity"] for c in supreme_cards if c["stage"] == "Item")
    supporter_count = sum(c["quantity"] for c in supreme_cards if c["stage"] == "Supporter")
    stadium_count = sum(c["quantity"] for c in supreme_cards if c["stage"] == "Stadium")
    energy_count = sum(c["quantity"] for c in supreme_cards if "Energy" in c["stage"])

    print(f"Pokémon count:  {pokemon_count:>2} (11 Basics: 4 Ogerpon ex, 2 Bulu, 2 Munkidori, 1 Fezandipiti ex, 1 Latias ex, 1 Budew)")
    print(f"Item count:     {item_count:>2} (Search, Recovery, Switch, 1 ACE SPEC)")
    print(f"Supporter count:{supporter_count:>2} (Draw & Gust)")
    print(f"Stadium count:  {stadium_count:>2} (Battle Cage)")
    print(f"Energy count:   {energy_count:>2} (10 Grass, 2 Darkness, 1 Grow Grass)")
    print(f"Total Cards:    {pokemon_count + item_count + supporter_count + stadium_count + energy_count:>2}")

    assert pokemon_count == 11, f"Pokemon count {pokemon_count} != 11"
    assert item_count == 24, f"Item count {item_count} != 24"
    assert supporter_count == 10, f"Supporter count {supporter_count} != 10"
    assert stadium_count == 2, f"Stadium count {stadium_count} != 2"
    assert energy_count == 13, f"Energy count {energy_count} != 13"
    assert pokemon_count + item_count + supporter_count + stadium_count + energy_count == 60

    print("-> TEST 7 PASSED: Macro distribution matches tournament specifications.")

    print("\n" + "=" * 90)
    print("FINAL EMPIRICAL VERDICT: CONFIRMED")
    print("100% of structural, relational, and rule-based invariants are verified.")
    print("=" * 90)
    conn.close()

if __name__ == "__main__":
    run_comprehensive_validation()
