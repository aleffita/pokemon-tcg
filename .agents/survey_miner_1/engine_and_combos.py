import sqlite3
import json
from collections import defaultdict
from itertools import combinations

DB_PATH = "file:/Users/alefita/workdir/pokemon-tcg/model/results.db?mode=ro"

def main():
    conn = sqlite3.connect(DB_PATH, uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("=== 1. Searching for Engine Cards in `cards` Table ===")
    search_terms = [
        "Professor", "Iono", "Colress", "Arven", "Carmine", "Lillie", "Judge", "Dawn", "Hilda", "Briar", "Canari", "Xerosic",
        "Nest Ball", "Ultra Ball", "Buddy-Buddy", "Pass", "Poffin", "Tera Orb", "Bug Catching", "Dusk Ball", "Trolley", "Pad",
        "Teal Mask", "Dark Patch", "Electric Generator", "Energy Switch", "Energy Retrieval", "Energy Search", "Crispin", "Powerglass",
        "Boss", "Prime Catcher", "Switch", "Super Rod", "Night Stretcher", "Crushing Hammer", "Enhanced Hammer", "Unfair Stamp", "Hero’s Cape", "Max Rod", "Sacred Ash", "Secret Box"
    ]

    found_cards = {}
    for term in search_terms:
        cursor.execute("""
            SELECT id, name, category, stage, hp, energy_type, weakness, rule
            FROM cards
            WHERE name LIKE ?
            ORDER BY id
        """, (f"%{term}%",))
        rows = cursor.fetchall()
        for r in rows:
            found_cards[r['id']] = dict(r)

    print(f"Total matching engine cards found: {len(found_cards)}")
    for cid, c in sorted(found_cards.items()):
        print(f"  ID {cid:4d}: {c['name']} | Stage: {c['stage']} | Category: {c['category']} | Type: {c['energy_type']} | HP: {c['hp']} | Rule: {c['rule']}")

    print("\n=== 2. Detailed Performance of Engine Cards in card_elo_daily & Matches ===")
    card_ids_tuple = tuple(found_cards.keys())
    placeholders = ",".join("?" for _ in card_ids_tuple)

    cursor.execute(f"""
        SELECT ced.card_id, c.name, c.stage, c.category, c.rule,
               MAX(ced.elo) as max_elo, AVG(ced.elo) as avg_elo,
               SUM(ced.games_played) as total_games, SUM(ced.wins) as total_wins,
               CAST(SUM(ced.wins) AS FLOAT) / NULLIF(SUM(ced.games_played), 0) as win_rate
        FROM card_elo_daily ced
        JOIN cards c ON ced.card_id = c.id
        WHERE ced.card_id IN ({placeholders})
        GROUP BY ced.card_id
        ORDER BY avg_elo DESC
    """, card_ids_tuple)
    engine_stats = [dict(r) for r in cursor.fetchall()]

    for es in engine_stats:
        wr_str = f"{es['win_rate']*100:.1f}%" if es['win_rate'] is not None else "N/A"
        print(f"  ID {es['card_id']:4d}: {es['name']:<30} | AvgElo: {es['avg_elo']:6.1f} | MaxElo: {es['max_elo']:6.1f} | WR: {wr_str:<6} | Games: {es['total_games']}")

    print("\n=== 3. Synergistic Card Combinations in High-Elo Decks (Elo >= 1100) ===")
    cursor.execute("""
        SELECT DISTINCT ded.deck_id
        FROM deck_elo_daily ded
        WHERE ded.elo >= 1100.0
    """)
    high_elo_deck_ids = [r['deck_id'] for r in cursor.fetchall()]
    print(f"Found {len(high_elo_deck_ids)} high-Elo deck IDs")

    # For each high-Elo deck, get all cards in it
    deck_card_map = defaultdict(list)
    for did in high_elo_deck_ids:
        cursor.execute("SELECT card_id, quantity FROM deck_cards WHERE deck_id = ?", (did,))
        for r in cursor.fetchall():
            deck_card_map[did].append(r['card_id'])

    # Pairwise co-occurrence in high-Elo decks
    pair_counts = defaultdict(int)
    for did, cids in deck_card_map.items():
        unique_cids = sorted(list(set(cids)))
        for c1, c2 in combinations(unique_cids, 2):
            pair_counts[(c1, c2)] += 1

    # Fetch card names
    cursor.execute("SELECT id, name FROM cards")
    card_name_map = {r['id']: r['name'] for r in cursor.fetchall()}

    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 40 most frequent card pairs in high-Elo (>=1100) decks:")
    for (c1, c2), cnt in sorted_pairs[:40]:
        print(f"  ({c1:4d}) {card_name_map.get(c1, 'Unknown'):<25} + ({c2:4d}) {card_name_map.get(c2, 'Unknown'):<25} : in {cnt}/{len(high_elo_deck_ids)} decks ({cnt/len(high_elo_deck_ids)*100:.1f}%)")

    # Triple combinations
    triple_counts = defaultdict(int)
    for did, cids in deck_card_map.items():
        unique_cids = sorted(list(set(cids)))
        for c1, c2, c3 in combinations(unique_cids, 3):
            triple_counts[(c1, c2, c3)] += 1

    sorted_triples = sorted(triple_counts.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 25 most frequent card triples in high-Elo (>=1100) decks:")
    for (c1, c2, c3), cnt in sorted_triples[:25]:
        print(f"  {card_name_map.get(c1, 'Unknown'):<20} + {card_name_map.get(c2, 'Unknown'):<20} + {card_name_map.get(c3, 'Unknown'):<20} : {cnt} decks ({cnt/len(high_elo_deck_ids)*100:.1f}%)")

    conn.close()

if __name__ == "__main__":
    main()
