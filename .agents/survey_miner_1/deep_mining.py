import sqlite3
import json
import collections

DB_PATH = "file:/Users/alefita/workdir/pokemon-tcg/model/results.db?mode=ro"

def main():
    conn = sqlite3.connect(DB_PATH, uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("=== 1. Performance of Deck #633 and Deck #251 in DB ===")
    for deck_id in [633, 251]:
        cursor.execute("""
            SELECT count(*) as total_matches,
                   SUM(CASE WHEN our_deck_id = ? AND result = 1 THEN 1
                            WHEN opp_deck_id = ? AND result = 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN our_deck_id = ? AND result = 0 THEN 1
                            WHEN opp_deck_id = ? AND result = 1 THEN 1 ELSE 0 END) as losses,
                   SUM(CASE WHEN result = 2 THEN 1 ELSE 0 END) as draws
            FROM matches
            WHERE our_deck_id = ? OR opp_deck_id = ?
        """, (deck_id, deck_id, deck_id, deck_id, deck_id, deck_id))
        row = cursor.fetchone()
        tot = row['total_matches']
        w = row['wins']
        l = row['losses']
        d = row['draws']
        wr = (w / tot * 100) if tot > 0 else 0
        print(f"Deck {deck_id} across all matches: Total={tot}, W={w}, L={l}, D={d}, WinRate={wr:.2f}%")

        # Check in deck_elo_daily
        cursor.execute("""
            SELECT day_id, elo, games_played, wins, losses, draws
            FROM deck_elo_daily
            WHERE deck_id = ?
            ORDER BY day_id DESC
            LIMIT 5
        """, (deck_id,))
        elo_rows = [dict(r) for r in cursor.fetchall()]
        print(f"  Latest deck_elo_daily for Deck {deck_id}: {elo_rows}")

    print("\n=== 2. High-Elo Decks (Elo >= 1100.0) in deck_elo_daily ===")
    cursor.execute("""
        SELECT ded.deck_id, MAX(ded.elo) as max_elo, AVG(ded.elo) as avg_elo,
               SUM(ded.games_played) as total_games, SUM(ded.wins) as total_wins,
               CAST(SUM(ded.wins) AS FLOAT) / SUM(ded.games_played) as win_rate,
               d.name, d.fingerprint
        FROM deck_elo_daily ded
        JOIN decks d ON ded.deck_id = d.id
        WHERE ded.elo >= 1100.0
        GROUP BY ded.deck_id
        ORDER BY max_elo DESC
    """)
    high_elo_decks = [dict(r) for r in cursor.fetchall()]
    print(f"Number of decks with Elo >= 1100: {len(high_elo_decks)}")
    for d in high_elo_decks[:15]:
        print(f"  Deck #{d['deck_id']}: MaxElo={d['max_elo']:.1f}, AvgElo={d['avg_elo']:.1f}, Games={d['total_games']}, WR={d['win_rate']*100:.1f}%, Name={d['name']}")

    print("\n=== 3. High-Elo Cards (Elo >= 1100.0) in card_elo_daily ===")
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
    high_elo_cards = [dict(r) for r in cursor.fetchall()]
    print(f"Number of distinct cards with daily Elo >= 1100.0: {len(high_elo_cards)}")
    for c in high_elo_cards[:40]:
        print(f"  ID {c['card_id']}: {c['name']} [{c['category']}|{c['stage']}|{c['energy_type']}|Rule:{c['rule']}] - MaxElo={c['max_elo']:.1f}, AvgElo={c['avg_elo']:.1f}, Games={c['total_games']}, WR={c['win_rate']*100:.1f}%")

    print("\n=== 4. High-Elo Matches Analysis (Matches where participating deck had Elo >= 1100) ===")
    # Let's see how many matches involve high-Elo decks or agents
    cursor.execute("""
        SELECT count(DISTINCT m.id) as cnt
        FROM matches m
        WHERE m.our_deck_id IN (SELECT deck_id FROM deck_elo_daily WHERE elo >= 1100)
           OR m.opp_deck_id IN (SELECT deck_id FROM deck_elo_daily WHERE elo >= 1100)
    """)
    print(f"Matches involving high-Elo decks: {cursor.fetchone()['cnt']}")

    # Let's analyze card win rates in matches with high-Elo decks
    cursor.execute("""
        WITH high_elo_decks AS (
            SELECT DISTINCT deck_id FROM deck_elo_daily WHERE elo >= 1100
        )
        SELECT dc.card_id, c.name, c.category, c.stage, c.energy_type, c.rule,
               count(DISTINCT hed.deck_id) as n_high_elo_decks,
               SUM(dc.quantity) as total_qty_across_decks
        FROM deck_cards dc
        JOIN high_elo_decks hed ON dc.deck_id = hed.deck_id
        JOIN cards c ON dc.card_id = c.id
        GROUP BY dc.card_id
        ORDER BY n_high_elo_decks DESC, total_qty_across_decks DESC
        LIMIT 50
    """)
    cards_in_high_elo_decks = [dict(r) for r in cursor.fetchall()]
    print("\nTop cards present in high-Elo (>=1100) decks:")
    for c in cards_in_high_elo_decks[:30]:
        print(f"  ID {c['card_id']}: {c['name']} ({c['stage'] or c['category']}) in {c['n_high_elo_decks']} high-Elo decks (total qty: {c['total_qty_across_decks']})")

    conn.close()

if __name__ == "__main__":
    main()
