import sqlite3
import json
from collections import defaultdict

DB_PATH = "file:/Users/alefita/workdir/pokemon-tcg/model/results.db?mode=ro"

def main():
    conn = sqlite3.connect(DB_PATH, uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Deck Archetypes with Elo >= 1100
    print("=== Archetype Analysis of Top Decks (Elo >= 1100) ===")
    cursor.execute("""
        SELECT ded.deck_id, MAX(ded.elo) as max_elo, AVG(ded.elo) as avg_elo,
               SUM(ded.games_played) as total_games, SUM(ded.wins) as total_wins,
               CAST(SUM(ded.wins) AS FLOAT) / SUM(ded.games_played) as win_rate
        FROM deck_elo_daily ded
        WHERE ded.elo >= 1100.0
        GROUP BY ded.deck_id
        ORDER BY max_elo DESC
    """)
    high_decks = [dict(r) for r in cursor.fetchall()]

    archetypes = defaultdict(list)
    for d in high_decks:
        did = d['deck_id']
        cursor.execute("""
            SELECT dc.card_id, dc.quantity, c.name, c.stage, c.category, c.energy_type, c.rule
            FROM deck_cards dc
            JOIN cards c ON dc.card_id = c.id
            WHERE dc.deck_id = ?
        """, (did,))
        cards = [dict(r) for r in cursor.fetchall()]
        # Identify key Pokémon
        key_pokemon = [c['name'] for c in cards if c['stage'] and ('ex' in c['name'] or 'Mega' in c['name'] or c['stage'] == 'Stage 2 Pokémon' or c['rule'] == 'Pokémon ex')]
        if not key_pokemon:
            key_pokemon = [c['name'] for c in cards if c['stage']]
        arch_label = " / ".join(key_pokemon[:2]) if key_pokemon else "Trainer/Tool heavy"
        archetypes[arch_label].append((d, cards))

    print(f"Discovered {len(archetypes)} distinct high-Elo archetypes:")
    for arch, dlist in sorted(archetypes.items(), key=lambda x: len(x[1]), reverse=True):
        avg_elo = sum(item[0]['avg_elo'] for item in dlist) / len(dlist)
        max_elo = max(item[0]['max_elo'] for item in dlist)
        tot_games = sum(item[0]['total_games'] for item in dlist)
        tot_wins = sum(item[0]['total_wins'] for item in dlist)
        wr = (tot_wins / tot_games * 100) if tot_games > 0 else 0
        print(f"  * Archetype: {arch} ({len(dlist)} decks) | Max Elo: {max_elo:.1f} | Avg Elo: {avg_elo:.1f} | WR: {wr:.1f}% ({tot_wins}/{tot_games})")

    # 2. Engine Classification & Performance
    print("\n=== Engine Classification & Empirical Performance ===")
    
    categories = {
        "Draw Supporters": [
            "Professor's Research", "Iono", "Colress’s Tenacity", "Arven", "Carmine",
            "Lillie's Determination", "Judge", "Dawn", "Hilda", "Canari", "Briar", "N's Plan", "Harlequin", "Wally's Compassion"
        ],
        "Search Items": [
            "Nest Ball", "Ultra Ball", "Buddy-Buddy Poffin", "Tera Orb", "Bug Catching Set",
            "Dusk Ball", "Precious Trolley", "Poké Pad", "Pokégear 3.0", "Energy Search"
        ],
        "Energy Acceleration & Recovery": [
            "Teal Mask Ogerpon ex", "Dark Patch", "Electric Generator", "Energy Switch",
            "Energy Retrieval", "Crispin", "Powerglass", "Max Rod", "Super Rod", "Night Stretcher", "Sacred Ash", "Lana’s Aid"
        ],
        "Disruption & Switching": [
            "Boss’s Orders", "Prime Catcher", "Switch", "Switch Cart", "Crushing Hammer",
            "Enhanced Hammer", "Xerosic’s Machinations", "Team Rocket's Petrel", "Unfair Stamp", "Hero’s Cape", "Gravity Mountain", "Lively Stadium", "Battle Cage"
        ]
    }

    for cat_name, card_names in categories.items():
        print(f"\n--- Category: {cat_name} ---")
        for cname in card_names:
            cursor.execute("""
                SELECT c.id, c.name, c.stage, c.category, c.rule,
                       MAX(ced.elo) as max_elo, AVG(ced.elo) as avg_elo,
                       SUM(ced.games_played) as total_games, SUM(ced.wins) as total_wins,
                       CAST(SUM(ced.wins) AS FLOAT) / NULLIF(SUM(ced.games_played), 0) as win_rate
                FROM cards c
                LEFT JOIN card_elo_daily ced ON c.id = ced.card_id
                WHERE c.name LIKE ?
                GROUP BY c.id
            """, (f"%{cname}%",))
            rows = cursor.fetchall()
            for r in rows:
                wr_val = f"{r['win_rate']*100:.1f}%" if r['win_rate'] is not None else "0 games"
                avg_elo_val = f"{r['avg_elo']:.1f}" if r['avg_elo'] is not None else "N/A"
                max_elo_val = f"{r['max_elo']:.1f}" if r['max_elo'] is not None else "N/A"
                games_val = r['total_games'] if r['total_games'] is not None else 0
                print(f"  ID {r['id']:4d}: {r['name']:<28} | Rule: {str(r['rule']):<12} | AvgElo: {avg_elo_val:<6} | MaxElo: {max_elo_val:<6} | WR: {wr_val:<7} | Games: {games_val}")

    # 3. Match Card Usage Analysis for Elo >= 1100
    print("\n=== Match Card Usage Analysis in High Elo Matches (Elo >= 1100) ===")
    cursor.execute("""
        WITH high_elo_participants AS (
            SELECT mp.id as participant_id, mp.match_id, mp.seat, mp.outcome, mp.deck_id
            FROM match_participants mp
            JOIN deck_elo_daily ded ON mp.deck_id = ded.deck_id
            WHERE ded.elo >= 1100.0
        )
        SELECT mcu.card_id, c.name, c.category, c.stage, c.rule,
               count(DISTINCT hep.participant_id) as games_used,
               SUM(CASE WHEN hep.outcome = 1 THEN 1 ELSE 0 END) as wins_with_card,
               CAST(SUM(CASE WHEN hep.outcome = 1 THEN 1 ELSE 0 END) AS FLOAT) / count(DISTINCT hep.participant_id) as win_rate_in_high_elo
        FROM match_card_usage mcu
        JOIN high_elo_participants hep ON mcu.participant_id = hep.participant_id
        JOIN cards c ON mcu.card_id = c.id
        GROUP BY mcu.card_id
        HAVING games_used >= 500
        ORDER BY win_rate_in_high_elo DESC
        LIMIT 40
    """)
    high_elo_mcu = [dict(r) for r in cursor.fetchall()]
    for r in high_elo_mcu:
        print(f"  ID {r['card_id']:4d}: {r['name']:<28} | HighElo WR: {r['win_rate_in_high_elo']*100:.1f}% ({r['wins_with_card']}/{r['games_used']}) | Rule: {r['rule']}")

    conn.close()

if __name__ == "__main__":
    main()
