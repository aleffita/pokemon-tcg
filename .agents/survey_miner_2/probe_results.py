import sqlite3
import json

db_path = "file:model/results.db?mode=ro"
conn = sqlite3.connect(db_path, uri=True)
cursor = conn.cursor()

print("--- TABLES ---")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print(tables)

print("\n--- DECKS SUMMARY ---")
cursor.execute("""
    SELECT id, name, archetype, source, card_count, created_at 
    FROM decks 
    ORDER BY id
""")
decks = cursor.fetchall()
for d in decks:
    print(f"Deck ID {d[0]}: Name='{d[1]}', Archetype='{d[2]}', Source='{d[3]}', Cards={d[4]}")

print(f"\nTotal decks count: {len(decks)}")

print("\n--- TOP OPPONENTS IN MATCHES ---")
cursor.execute("""
    SELECT opp_agent, COUNT(*) as cnt, 
           SUM(CASE WHEN result=1 THEN 1 ELSE 0 END) as our_w,
           SUM(CASE WHEN result=-1 THEN 1 ELSE 0 END) as our_l,
           SUM(CASE WHEN result=0 THEN 1 ELSE 0 END) as our_d
    FROM matches
    GROUP BY opp_agent
    ORDER BY cnt DESC
    LIMIT 25
""")
for row in cursor.fetchall():
    print(row)

print("\n--- MATCHUPS TABLE ---")
cursor.execute("""
    SELECT opponent, COUNT(*) as n_entries, SUM(w) as total_w, SUM(l) as total_l, SUM(d) as total_d, AVG(win_rate) as avg_wr, AVG(lb_score) as avg_lb
    FROM matchups
    GROUP BY opponent
    ORDER BY n_entries DESC
""")
for row in cursor.fetchall():
    print(row)

print("\n--- AGENTS TABLE ---")
cursor.execute("SELECT id, name, kaggle_username, submission_ref, is_self FROM agents LIMIT 30")
for row in cursor.fetchall():
    print(row)

conn.close()
