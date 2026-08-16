import sqlite3

DB_PATH = "file:/Users/alefita/workdir/pokemon-tcg/model/results.db?mode=ro"

conn = sqlite3.connect(DB_PATH, uri=True)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("=== ALL ITEMS IN CARDS TABLE ===")
cursor.execute("""
    SELECT id, name, category, stage, rule
    FROM cards
    WHERE stage = 'Item'
    ORDER BY name
""")
rows = cursor.fetchall()
print(f"Total Item cards found: {len(rows)}")
for r in rows:
    print(f"ID {r['id']:4d}: {r['name']:<35} | Rule: {str(r['rule']):<12} | Category: {r['category']}")

conn.close()
