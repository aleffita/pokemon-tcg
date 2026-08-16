import sqlite3

DB_PATH = "file:/Users/alefita/workdir/pokemon-tcg/model/results.db?mode=ro"

conn = sqlite3.connect(DB_PATH, uri=True)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("=== ALL TRAINERS IN CARDS TABLE ===")
cursor.execute("""
    SELECT id, name, category, stage, rule
    FROM cards
    WHERE stage IN ('Supporter', 'Item', 'Pokémon Tool', 'Stadium')
       OR category IN ('Item', 'Supporter', 'Stadium', 'Pokémon Tool', 'Technical Machine')
       OR id >= 1070
    ORDER BY stage, name
""")
rows = cursor.fetchall()
print(f"Total trainer cards found: {len(rows)}")
for r in rows:
    print(f"ID {r['id']:4d}: {r['name']:<35} | Stage: {str(r['stage']):<15} | Rule: {str(r['rule']):<12} | Category: {r['category']}")

conn.close()
