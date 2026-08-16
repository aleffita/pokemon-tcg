import sqlite3
import json
import os

DB_PATH = "file:/Users/alefita/workdir/pokemon-tcg/model/results.db?mode=ro"

def main():
    conn = sqlite3.connect(DB_PATH, uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("=== 1. Inspecting Decks #633 and #251 ===")
    cursor.execute("SELECT * FROM decks WHERE id IN (633, 251)")
    decks = [dict(row) for row in cursor.fetchall()]
    print(f"Decks found: {decks}")

    for deck_id in [633, 251]:
        print(f"\n--- Deck {deck_id} Composition ---")
        cursor.execute("""
            SELECT dc.card_id, dc.quantity, c.name, c.category, c.stage, c.hp, c.energy_type, c.weakness, c.rule
            FROM deck_cards dc
            JOIN cards c ON dc.card_id = c.id
            WHERE dc.deck_id = ?
            ORDER BY c.category, c.name
        """, (deck_id,))
        cards = [dict(row) for row in cursor.fetchall()]
        total_cards = sum(c['quantity'] for c in cards)
        print(f"Total cards in deck {deck_id}: {total_cards}")
        for c in cards:
            print(f"  ID {c['card_id']}: {c['quantity']}x {c['name']} [{c['category']}|{c['stage']}|{c['energy_type']}|HP:{c['hp']}|Rule:{c['rule']}]")

    print("\n=== 2. Legal Cards Catalog Summary ===")
    cursor.execute("SELECT count(*) as cnt FROM cards")
    total_legal_cards = cursor.fetchone()['cnt']
    print(f"Total legal cards in catalog: {total_legal_cards}")

    cursor.execute("""
        SELECT category, count(*) as cnt
        FROM cards
        GROUP BY category
        ORDER BY cnt DESC
    """)
    for row in cursor.fetchall():
        print(f"  Category: {row['category']} -> {row['cnt']} cards")

    print("\n=== 3. Card Elo Daily & High-Elo Metrics ===")
    cursor.execute("SELECT count(*) as cnt FROM card_elo_daily")
    total_card_elo_rows = cursor.fetchone()['cnt']
    print(f"Total rows in card_elo_daily: {total_card_elo_rows}")

    if total_card_elo_rows > 0:
        cursor.execute("""
            SELECT c.id, c.name, c.category, AVG(ced.elo) as avg_elo, MAX(ced.elo) as max_elo,
                   SUM(ced.games_played) as total_games, SUM(ced.wins) as total_wins,
                   CAST(SUM(ced.wins) AS FLOAT) / NULLIF(SUM(ced.games_played), 0) as win_rate
            FROM card_elo_daily ced
            JOIN cards c ON ced.card_id = c.id
            GROUP BY c.id, c.name, c.category
            HAVING total_games >= 50
            ORDER BY avg_elo DESC
            LIMIT 30
        """)
        top_elo_cards = [dict(row) for row in cursor.fetchall()]
        print("Top 30 cards by avg_elo in card_elo_daily (min 50 games):")
        for c in top_elo_cards:
            print(f"  ID {c['id']}: {c['name']} ({c['category']}) | Avg Elo: {c['avg_elo']:.1f} | Max Elo: {c['max_elo']:.1f} | WR: {c['win_rate']*100:.1f}% ({c['total_wins']}/{c['total_games']})")

    print("\n=== 4. Matches & Decks Stats ===")
    cursor.execute("SELECT count(*) as cnt FROM matches")
    print(f"Total matches in DB: {cursor.fetchone()['cnt']}")

    cursor.execute("SELECT count(*) as cnt FROM match_card_usage")
    print(f"Total match_card_usage rows: {cursor.fetchone()['cnt']}")

    cursor.execute("SELECT count(*) as cnt FROM deck_elo_daily")
    print(f"Total deck_elo_daily rows: {cursor.fetchone()['cnt']}")

    cursor.execute("SELECT count(*) as cnt FROM tournaments")
    print(f"Total tournaments in DB: {cursor.fetchone()['cnt']}")

    conn.close()

if __name__ == "__main__":
    main()
