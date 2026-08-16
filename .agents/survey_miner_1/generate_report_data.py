import sqlite3
import json
from collections import defaultdict
from itertools import combinations

DB_PATH = "file:/Users/alefita/workdir/pokemon-tcg/model/results.db?mode=ro"

def main():
    conn = sqlite3.connect(DB_PATH, uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    report = {}

    # ----------------------------------------------------
    # Task 2: Deck #633 and Deck #251 Compositions
    # ----------------------------------------------------
    print("[1/5] Extracting Decks #633 and #251...", flush=True)
    report["decks"] = {}
    for deck_id in [633, 251]:
        cursor.execute("SELECT * FROM decks WHERE id = ?", (deck_id,))
        deck_meta = dict(cursor.fetchone())

        cursor.execute("""
            SELECT dc.card_id, dc.quantity, c.name, c.category, c.stage, c.hp, c.energy_type, c.weakness, c.rule
            FROM deck_cards dc
            JOIN cards c ON dc.card_id = c.id
            WHERE dc.deck_id = ?
            ORDER BY c.category, c.stage, c.name
        """, (deck_id,))
        cards = [dict(r) for r in cursor.fetchall()]
        total_qty = sum(c["quantity"] for c in cards)

        # Performance across matches
        cursor.execute("""
            SELECT count(*) as total_matches,
                   SUM(CASE WHEN (our_deck_id = ? AND result = 1) OR (opp_deck_id = ? AND result = 0) THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN (our_deck_id = ? AND result = 0) OR (opp_deck_id = ? AND result = 1) THEN 1 ELSE 0 END) as losses,
                   SUM(CASE WHEN result = 2 THEN 1 ELSE 0 END) as draws
            FROM matches
            WHERE our_deck_id = ? OR opp_deck_id = ?
        """, (deck_id, deck_id, deck_id, deck_id, deck_id, deck_id))
        perf = dict(cursor.fetchone())
        perf["win_rate"] = (perf["wins"] / perf["total_matches"] * 100) if perf["total_matches"] > 0 else 0

        # Elo daily
        cursor.execute("""
            SELECT day_id, elo, games_played, wins, losses, draws
            FROM deck_elo_daily
            WHERE deck_id = ?
            ORDER BY day_id DESC
        """, (deck_id,))
        elo_history = [dict(r) for r in cursor.fetchall()]

        report["decks"][str(deck_id)] = {
            "metadata": deck_meta,
            "total_cards": total_qty,
            "performance": perf,
            "elo_history": elo_history,
            "cards": cards
        }

    # ----------------------------------------------------
    # Task 3: Legal Card Catalog
    # ----------------------------------------------------
    print("[2/5] Extracting Legal Card Catalog...", flush=True)
    cursor.execute("""
        SELECT id, name, category, stage, hp, energy_type, weakness, rule, metadata_complete
        FROM cards
        ORDER BY id
    """)
    all_cards = [dict(r) for r in cursor.fetchall()]
    report["catalog_summary"] = {
        "total_cards": len(all_cards),
        "by_category": {},
        "by_stage": {},
        "by_energy_type": {},
        "by_rule": {}
    }
    for c in all_cards:
        cat = c["category"] or "None"
        stg = c["stage"] or "Trainer/Energy"
        etype = c["energy_type"] or "Colorless/None"
        rule = c["rule"] or "Standard"

        report["catalog_summary"]["by_category"][cat] = report["catalog_summary"]["by_category"].get(cat, 0) + 1
        report["catalog_summary"]["by_stage"][stg] = report["catalog_summary"]["by_stage"].get(stg, 0) + 1
        report["catalog_summary"]["by_energy_type"][etype] = report["catalog_summary"]["by_energy_type"].get(etype, 0) + 1
        report["catalog_summary"]["by_rule"][rule] = report["catalog_summary"]["by_rule"].get(rule, 0) + 1

    # ----------------------------------------------------
    # Task 4: High-Elo Cards and Combinations (Elo >= 1100.0)
    # ----------------------------------------------------
    print("[3/5] Mining High-Elo (>=1100.0) Cards and Synergies...", flush=True)
    cursor.execute("""
        SELECT ced.card_id, c.name, c.category, c.stage, c.energy_type, c.hp, c.rule,
               MAX(ced.elo) as max_elo, AVG(ced.elo) as avg_elo,
               SUM(ced.games_played) as total_games, SUM(ced.wins) as total_wins,
               CAST(SUM(ced.wins) AS FLOAT) / SUM(ced.games_played) as win_rate
        FROM card_elo_daily ced
        JOIN cards c ON ced.card_id = c.id
        WHERE ced.elo >= 1100.0
        GROUP BY ced.card_id
        ORDER BY max_elo DESC
    """)
    report["high_elo_cards"] = [dict(r) for r in cursor.fetchall()]

    # High Elo decks
    cursor.execute("""
        SELECT ded.deck_id, MAX(ded.elo) as max_elo, AVG(ded.elo) as avg_elo,
               SUM(ded.games_played) as total_games, SUM(ded.wins) as total_wins,
               CAST(SUM(ded.wins) AS FLOAT) / SUM(ded.games_played) as win_rate
        FROM deck_elo_daily ded
        WHERE ded.elo >= 1100.0
        GROUP BY ded.deck_id
        ORDER BY max_elo DESC
    """)
    high_elo_decks = [dict(r) for r in cursor.fetchall()]
    high_elo_deck_ids = [d["deck_id"] for d in high_elo_decks]
    report["high_elo_decks_count"] = len(high_elo_decks)
    report["high_elo_decks"] = high_elo_decks

    # Co-occurrence analysis in high-Elo decks
    deck_card_map = defaultdict(list)
    for did in high_elo_deck_ids:
        cursor.execute("SELECT card_id, quantity FROM deck_cards WHERE deck_id = ?", (did,))
        for r in cursor.fetchall():
            deck_card_map[did].append(r["card_id"])

    pair_counts = defaultdict(int)
    for did, cids in deck_card_map.items():
        unique_cids = sorted(list(set(cids)))
        for c1, c2 in combinations(unique_cids, 2):
            pair_counts[(c1, c2)] += 1

    card_name_map = {c["id"]: c["name"] for c in all_cards}
    report["top_card_pairs"] = []
    for (c1, c2), cnt in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:50]:
        report["top_card_pairs"].append({
            "card_1_id": c1, "card_1_name": card_name_map.get(c1, "Unknown"),
            "card_2_id": c2, "card_2_name": card_name_map.get(c2, "Unknown"),
            "deck_count": cnt,
            "deck_percentage": cnt / len(high_elo_deck_ids) * 100
        })

    # Triples
    triple_counts = defaultdict(int)
    for did, cids in deck_card_map.items():
        unique_cids = sorted(list(set(cids)))
        for c1, c2, c3 in combinations(unique_cids, 3):
            triple_counts[(c1, c2, c3)] += 1

    report["top_card_triples"] = []
    for (c1, c2, c3), cnt in sorted(triple_counts.items(), key=lambda x: x[1], reverse=True)[:30]:
        report["top_card_triples"].append({
            "card_1_id": c1, "card_1_name": card_name_map.get(c1, "Unknown"),
            "card_2_id": c2, "card_2_name": card_name_map.get(c2, "Unknown"),
            "card_3_id": c3, "card_3_name": card_name_map.get(c3, "Unknown"),
            "deck_count": cnt,
            "deck_percentage": cnt / len(high_elo_deck_ids) * 100
        })

    # ----------------------------------------------------
    # Task 5: Top Engine Cards Analysis
    # ----------------------------------------------------
    print("[4/5] Classifying Top Engine Cards...", flush=True)
    engine_groups = {
        "Draw Supporters": [
            "Professor", "Iono", "Colress", "Arven", "Carmine", "Lillie", "Judge", "Dawn", "Hilda", "Briar", "Canari", "Xerosic", "Wally"
        ],
        "Search Items": [
            "Nest Ball", "Ultra Ball", "Buddy-Buddy", "Pass", "Poffin", "Tera Orb", "Bug Catching", "Dusk Ball", "Trolley", "Poké Pad", "Pokégear"
        ],
        "Energy Acceleration & Recovery": [
            "Teal Mask", "Dark Patch", "Electric Generator", "Energy Switch", "Energy Retrieval", "Crispin", "Powerglass", "Max Rod", "Super Rod", "Night Stretcher", "Sacred Ash", "Lana"
        ],
        "Disruption & Switching": [
            "Boss’s Orders", "Prime Catcher", "Switch", "Switch Cart", "Crushing Hammer", "Enhanced Hammer", "Unfair Stamp", "Hero’s Cape", "Gravity Mountain", "Lively Stadium", "Battle Cage", "Secret Box"
        ]
    }

    report["engine_analysis"] = {}
    for group_name, patterns in engine_groups.items():
        report["engine_analysis"][group_name] = []
        for pat in patterns:
            cursor.execute("""
                SELECT c.id, c.name, c.category, c.stage, c.rule, c.hp, c.energy_type,
                       MAX(ced.elo) as max_elo, AVG(ced.elo) as avg_elo,
                       SUM(ced.games_played) as total_games, SUM(ced.wins) as total_wins,
                       CAST(SUM(ced.wins) AS FLOAT) / NULLIF(SUM(ced.games_played), 0) as win_rate
                FROM cards c
                LEFT JOIN card_elo_daily ced ON c.id = ced.card_id
                WHERE c.name LIKE ?
                GROUP BY c.id
                ORDER BY avg_elo DESC
            """, (f"%{pat}%",))
            for r in cursor.fetchall():
                d = dict(r)
                # Check how many high-Elo decks run this card
                cursor.execute("""
                    SELECT count(DISTINCT deck_id) as high_elo_decks, SUM(quantity) as total_qty
                    FROM deck_cards
                    WHERE card_id = ? AND deck_id IN (SELECT DISTINCT deck_id FROM deck_elo_daily WHERE elo >= 1100.0)
                """, (d["id"],))
                he_info = cursor.fetchone()
                d["high_elo_deck_count"] = he_info["high_elo_decks"]
                d["high_elo_total_qty"] = he_info["total_qty"] or 0
                report["engine_analysis"][group_name].append(d)

    # ----------------------------------------------------
    # Save output to JSON
    # ----------------------------------------------------
    print("[5/5] Saving compilation to mining_data.json...", flush=True)
    out_path = "/Users/alefita/workdir/pokemon-tcg/.agents/survey_miner_1/mining_data.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("Successfully generated mining_data.json!", flush=True)
    conn.close()

if __name__ == "__main__":
    main()
