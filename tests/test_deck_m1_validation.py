import json
import sqlite3
import math
from fractions import Fraction

def test_deck_validation():
    db_path = "file:model/results.db?mode=ro"
    conn = sqlite3.connect(db_path, uri=True)
    c = conn.cursor()

    # 1. Load agent/deck.json
    with open("agent/deck.json", "r") as f:
        deck_ids = json.load(f)

    assert isinstance(deck_ids, list), "agent/deck.json must be a list"
    assert len(deck_ids) == 60, f"agent/deck.json must contain exactly 60 cards, got {len(deck_ids)}"
    for idx, cid in enumerate(deck_ids):
        assert isinstance(cid, int), f"Element at index {idx} ({cid}) is not an integer"

    # 2. Check all card IDs exist in model/results.db
    card_info = {}
    for cid in set(deck_ids):
        c.execute("SELECT id, name, category, stage, hp, energy_type, rule FROM cards WHERE id = ?", (cid,))
        row = c.fetchone()
        assert row is not None, f"Card ID {cid} does not exist in model/results.db cards table"
        card_info[cid] = {
            "id": row[0],
            "name": row[1],
            "category": row[2],
            "stage": row[3],
            "hp": row[4],
            "energy_type": row[5],
            "rule": row[6]
        }

    # 3. Rule compliance: Max 4 copies per name (except Basic Energy)
    name_counts = {}
    for cid in deck_ids:
        info = card_info[cid]
        name = info["name"]
        stage = info["stage"]
        if stage != "Basic Energy":
            name_counts[name] = name_counts.get(name, 0) + 1
            assert name_counts[name] <= 4, f"Rule violation: {name} appears {name_counts[name]} times (max 4 allowed)"

    # 4. Exactly 1 ACE SPEC card
    ace_spec_count = sum(1 for cid in deck_ids if card_info[cid]["rule"] == "ACE SPEC")
    assert ace_spec_count == 1, f"Deck must contain exactly 1 ACE SPEC, found {ace_spec_count}"

    # 5. At least 10 Basic Pokemon
    basic_pokemon_count = sum(1 for cid in deck_ids if card_info[cid]["stage"] == "Basic Pokémon")
    assert basic_pokemon_count >= 10, f"Deck must have at least 10 Basic Pokémon, found {basic_pokemon_count}"

    # 6. Load and validate experiments/decks/deck_supreme_60.json
    with open("experiments/decks/deck_supreme_60.json", "r") as f:
        capsule = json.load(f)

    assert capsule["deck_name"] == "Deck Supreme 60 — Teal Mask Ogerpon ex / Turbo Acceleration & Psychic Counter Hybrid"
    assert capsule["archetype"] == "Teal Mask Ogerpon ex / Grass Turbo Ramp / Anti-Meta Control"
    assert capsule["card_count"] == 60

    # Validate card_list in capsule
    capsule_card_list = capsule["card_list"]
    capsule_id_expansion = []
    required_fields = ["id", "name", "category", "stage", "type", "hp", "rule", "quantity", "role"]
    for card_obj in capsule_card_list:
        for rf in required_fields:
            assert rf in card_obj, f"Missing field '{rf}' in card_obj: {card_obj}"
        cid = card_obj["id"]
        q = card_obj["quantity"]
        assert cid in card_info, f"Capsule Card ID {cid} not verified in DB"
        assert card_obj["name"] == card_info[cid]["name"], f"Name mismatch: capsule {card_obj['name']} vs db {card_info[cid]['name']}"
        capsule_id_expansion.extend([cid] * q)

    assert len(capsule_id_expansion) == 60, f"Capsule total card count must be 60, got {len(capsule_id_expansion)}"
    assert sorted(capsule_id_expansion) == sorted(deck_ids), "Card IDs in capsule do not match agent/deck.json"

    # Validate energy_curve
    ec = capsule["energy_curve"]
    assert ec["total_energy"] == 13
    assert ec["basic_energy"]["total_basic_energy"] == 12
    assert ec["special_energy"]["total_special_energy"] == 1
    assert "turn_by_turn_attachment_expectations" in ec

    # Validate hypergeometric probabilities
    hg = capsule["hypergeometric_probabilities"]
    assert hg["basic_pokemon_count_Kb"] == basic_pokemon_count
    assert hg["p_setup_within_1_mulligan"]["float"] >= 0.92, f"P(Setup within 1 mulligan) must be >= 0.92, got {hg['p_setup_within_1_mulligan']['float']}"
    assert hg["p_mulligan_within_1_mulligan"]["float"] <= 0.08, f"P(Mulligan within 1 mulligan) must be <= 0.08, got {hg['p_mulligan_within_1_mulligan']['float']}"
    
    # Confirm exact hypergeometric math
    N = 60
    n = 7
    p_mul_7 = Fraction(math.comb(N - basic_pokemon_count, n), math.comb(N, n))
    p_setup_7 = 1 - p_mul_7
    p_mul_within_1 = p_mul_7 ** 2
    p_setup_within_1 = 1 - p_mul_within_1

    assert hg["p_setup_n7"]["rational"] == f"{p_setup_7.numerator}/{p_setup_7.denominator}"
    assert hg["p_mulligan_n7"]["rational"] == f"{p_mul_7.numerator}/{p_mul_7.denominator}"
    assert hg["p_setup_within_1_mulligan"]["rational"] == f"{p_setup_within_1.numerator}/{p_setup_within_1.denominator}"
    assert hg["p_mulligan_within_1_mulligan"]["rational"] == f"{p_mul_within_1.numerator}/{p_mul_within_1.denominator}"

    # Validate matchup_profiles (all 6 panel archetypes present)
    mp = capsule["matchup_profiles"]
    expected_matchups = [
        "lb826_alakazam_seok",
        "lb1009_945_mega_lucario_ex",
        "lb814_600_dragapult_crustle",
        "first_sub_kaggle_2707",
        "lb510_mega_abomasnow",
        "deck_633_baseline_yan"
    ]
    for em in expected_matchups:
        assert em in mp, f"Missing matchup profile for {em}"
        assert "opponent_threat_vector" in mp[em]
        assert "counter_strategy" in mp[em]
        assert "key_tactical_lines" in mp[em]
        assert len(mp[em]["key_tactical_lines"]) >= 3

    print("ALL VERIFICATION CHECKS PASSED PERFECTLY!")

if __name__ == "__main__":
    test_deck_validation()
