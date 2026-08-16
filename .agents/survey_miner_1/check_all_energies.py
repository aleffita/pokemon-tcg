import sqlite3

DB_PATH = "file:/Users/alefita/workdir/pokemon-tcg/model/results.db?mode=ro"

conn = sqlite3.connect(DB_PATH, uri=True)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("=== ALL ENERGIES IN CARDS TABLE ===")
cursor.execute("""
    SELECT id, name, category, stage, energy_type, rule
    FROM cards
    WHERE stage LIKE '%Energy%' OR name LIKE '%Energy%'
    ORDER BY id
""")
rows = cursor.fetchall()
print(f"Total energy cards found: {len(rows)}")
for r in rows:
    print(f"ID {r['id']:4d}: {r['name']:<30} | Stage: {str(r['stage']):<15} | Type: {str(r['energy_type']):<6} | Rule: {str(r['rule'])}")

conn.close()
