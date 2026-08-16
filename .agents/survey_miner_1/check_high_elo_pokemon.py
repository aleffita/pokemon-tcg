import sqlite3

DB_PATH = "file:/Users/alefita/workdir/pokemon-tcg/model/results.db?mode=ro"

conn = sqlite3.connect(DB_PATH, uri=True)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("=== ALL POKEMON IN HIGH-ELO DECKS (Elo >= 1100) ===")
cursor.execute("""
    WITH high_elo_decks AS (
        SELECT DISTINCT deck_id FROM deck_elo_daily WHERE elo >= 1100.0
    )
    SELECT c.id, c.name, c.stage, c.energy_type, c.hp, c.weakness, c.rule, c.category,
           count(DISTINCT hed.deck_id) as decks_count,
           SUM(dc.quantity) as total_qty
    FROM cards c
    JOIN deck_cards dc ON c.id = dc.card_id
    JOIN high_elo_decks hed ON dc.deck_id = hed.deck_id
    WHERE c.stage LIKE '%Pokémon%'
    GROUP BY c.id
    ORDER BY decks_count DESC, total_qty DESC
""")
rows = cursor.fetchall()
print(f"Total distinct Pokémon in high-Elo decks: {len(rows)}")
for r in rows:
    print(f"ID {r['id']:4d}: {r['name']:<25} | Stage: {str(r['stage']):<16} | Type: {str(r['energy_type']):<4} | HP: {str(r['hp']):<4} | Rule: {str(r['rule']):<12} | In {r['decks_count']} decks (Qty: {r['total_qty']})")

conn.close()
