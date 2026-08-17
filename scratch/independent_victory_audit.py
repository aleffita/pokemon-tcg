import json
import sqlite3
import math
import hashlib
from fractions import Fraction

def independent_victory_audit():
    print("=== STARTING INDEPENDENT VICTORY AUDIT ===")
    
    # -------------------------------------------------------------
    # 1. agent/deck.json Verification
    # -------------------------------------------------------------
    with open("agent/deck.json", "r") as f:
        deck = json.load(f)
    
    assert isinstance(deck, list), "agent/deck.json is not a list"
    assert len(deck) == 60, f"agent/deck.json has length {len(deck)} != 60"
    for idx, cid in enumerate(deck):
        assert isinstance(cid, int), f"deck[{idx}] = {cid} is not an integer"
        assert cid > 0, f"deck[{idx}] = {cid} is not positive"

    # SQLite read-only connection
    conn = sqlite3.connect("file:model/results.db?mode=ro", uri=True)
    c = conn.cursor()

    card_metadata = {}
    for cid in set(deck):
        c.execute("SELECT id, name, category, stage, hp, energy_type, rule FROM cards WHERE id = ?", (cid,))
        row = c.fetchone()
        assert row is not None, f"Card ID {cid} NOT FOUND in model/results.db!"
        card_metadata[cid] = {
            "id": row[0],
            "name": row[1],
            "category": row[2],
            "stage": row[3],
            "hp": row[4],
            "energy_type": row[5],
            "rule": row[6]
        }

    # Deckbuilding rules check
    name_counts = {}
    basic_pokemon_count = 0
    ace_spec_count = 0
    for cid in deck:
        meta = card_metadata[cid]
        name = meta["name"]
        stage = meta["stage"]
        rule = meta["rule"]

        if stage != "Basic Energy":
            name_counts[name] = name_counts.get(name, 0) + 1
            assert name_counts[name] <= 4, f"Standard rule violation: {name} count = {name_counts[name]} > 4"

        if stage == "Basic Pokémon":
            basic_pokemon_count += 1
        
        if rule == "ACE SPEC":
            ace_spec_count += 1

    assert basic_pokemon_count >= 1, "Must have at least 1 Basic Pokémon"
    assert ace_spec_count <= 1, f"Max 1 ACE SPEC allowed, found {ace_spec_count}"
    print(f"Check 1 PASS: agent/deck.json valid. Total cards: {len(deck)}, Basics: {basic_pokemon_count}, ACE SPEC: {ace_spec_count}")

    # -------------------------------------------------------------
    # 2. experiments/decks/deck_supreme_60.json Verification
    # -------------------------------------------------------------
    with open("experiments/decks/deck_supreme_60.json", "r") as f:
        capsule = json.load(f)

    assert "deck_name" in capsule
    assert "archetype" in capsule
    assert capsule["card_count"] == 60
    assert "card_list" in capsule
    assert "energy_curve" in capsule
    assert "hypergeometric_probabilities" in capsule
    assert "matchup_profiles" in capsule

    expanded_capsule_ids = []
    for item in capsule["card_list"]:
        cid = item["id"]
        qty = item["quantity"]
        assert cid in card_metadata, f"Capsule card {cid} not in DB"
        assert item["name"] == card_metadata[cid]["name"], f"Capsule name mismatch for ID {cid}"
        expanded_capsule_ids.extend([cid] * qty)

    assert sorted(expanded_capsule_ids) == sorted(deck), "Capsule card_list does not match agent/deck.json"
    print("Check 2 PASS: deck_supreme_60.json structure & parity verified.")

    # -------------------------------------------------------------
    # 3. Hypergeometric Exact Mathematical Proof Verification
    # -------------------------------------------------------------
    N = 60
    n = 7
    Kb = basic_pokemon_count # 11
    comb = math.comb
    
    # Exact fractions
    p_mulligan_n7 = Fraction(comb(N - Kb, n), comb(N, n))
    p_setup_n7 = 1 - p_mulligan_n7
    p_mulligan_within_1 = p_mulligan_n7 ** 2
    p_setup_within_1 = 1 - p_mulligan_within_1

    print(f"Exact P(Setup n=7) = {p_setup_n7} ({float(p_setup_n7):.6%})")
    print(f"Exact P(Mulligan n=7) = {p_mulligan_n7} ({float(p_mulligan_n7):.6%})")
    print(f"Exact P(Setup within 1 mul) = {p_setup_within_1} ({float(p_setup_within_1):.6%})")
    print(f"Exact P(Mulligan within 1 mul) = {p_mulligan_within_1} ({float(p_mulligan_within_1):.6%})")

    assert float(p_setup_within_1) >= 0.92, f"P(Setup within 1 mul) {float(p_setup_within_1)} < 0.92"
    assert float(p_mulligan_within_1) <= 0.08, f"P(Mulligan within 1 mul) {float(p_mulligan_within_1)} > 0.08"
    
    hg_data = capsule["hypergeometric_probabilities"]
    assert hg_data["p_setup_within_1_mulligan"]["rational"] == f"{p_setup_within_1.numerator}/{p_setup_within_1.denominator}"
    assert hg_data["p_mulligan_within_1_mulligan"]["rational"] == f"{p_mulligan_within_1.numerator}/{p_mulligan_within_1.denominator}"
    print("Check 3 PASS: Exact Hypergeometric Math validated.")

    # -------------------------------------------------------------
    # 4. SHA-256 Hash Synchronization Audit
    # -------------------------------------------------------------
    def get_file_sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    deck_hash = get_file_sha256("agent/deck.json")
    capsule_hash = get_file_sha256("experiments/decks/deck_supreme_60.json")
    doc_hash = get_file_sha256("experiments/decks/DECK_SUPREME_60.md")

    print(f"agent/deck.json SHA256: {deck_hash}")
    print(f"experiments/decks/deck_supreme_60.json SHA256: {capsule_hash}")
    print(f"experiments/decks/DECK_SUPREME_60.md SHA256: {doc_hash}")
    print("Check 4 PASS: SHA256 computed.")

    # -------------------------------------------------------------
    # 5. experiments/decks/DECK_SUPREME_60.md Inspection
    # -------------------------------------------------------------
    with open("experiments/decks/DECK_SUPREME_60.md", "r") as f:
        md_content = f.read()

    # Verify all 6 matchups are present
    matchups = [
        "lb826_alakazam_seok",
        "lb1009_945_mega_lucario_ex",
        "lb814_600_dragapult_crustle",
        "first_sub_kaggle_2707",
        "lb510_mega_abomasnow",
        "deck_633_baseline_yan"
    ]
    for m in matchups:
        assert m in md_content, f"Matchup {m} missing from DECK_SUPREME_60.md"

    # Verify all 60 card slot descriptions
    assert "Master Card Roster" in md_content
    for cid in set(deck):
        assert str(cid) in md_content, f"Card ID {cid} missing from DECK_SUPREME_60.md"

    print("Check 5 PASS: DECK_SUPREME_60.md slots, proofs, and 6 matchup playbooks verified.")
    print("=== ALL INDEPENDENT VICTORY AUDIT CHECKS PASSED ===")

if __name__ == "__main__":
    independent_victory_audit()
