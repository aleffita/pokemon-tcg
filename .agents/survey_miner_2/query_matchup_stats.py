import sqlite3
import json
from collections import defaultdict

db_path = "file:model/results.db?mode=ro"
conn = sqlite3.connect(db_path, uri=True)
cursor = conn.cursor()

print("=== DECK STATS & ELO IN SQLITE ===")
cursor.execute("""
    SELECT d.id, d.name, d.archetype, d.card_count,
           COUNT(m.id) as total_matches,
           SUM(CASE WHEN m.result = 1 THEN 1 ELSE 0 END) as wins,
           SUM(CASE WHEN m.result = -1 THEN 1 ELSE 0 END) as losses,
           SUM(CASE WHEN m.result = 0 THEN 1 ELSE 0 END) as draws
    FROM decks d
    LEFT JOIN matches m ON (d.id = m.our_deck_id OR d.id = m.opp_deck_id)
    GROUP BY d.id
    ORDER BY total_matches DESC
    LIMIT 30
""")
deck_stats = cursor.fetchall()
for ds in deck_stats:
    tot = ds[4]
    wr = (ds[5] / tot * 100) if tot > 0 else 0
    print(f"Deck ID={ds[0]}: Name='{ds[1]}', Archetype='{ds[2]}', Matches={tot}, W={ds[5]}, L={ds[6]}, D={ds[7]}, WR={wr:.1f}%")

print("\n=== MATCHUPS BETWEEN POPULAR DECKS ===")
cursor.execute("""
    SELECT our_deck_id, opp_deck_id, 
           COUNT(*) as cnt,
           SUM(CASE WHEN result = 1 THEN 1 ELSE 0 END) as our_w,
           SUM(CASE WHEN result = -1 THEN 1 ELSE 0 END) as our_l,
           SUM(CASE WHEN result = 0 THEN 1 ELSE 0 END) as our_d
    FROM matches
    WHERE our_deck_id IS NOT NULL AND opp_deck_id IS NOT NULL
    GROUP BY our_deck_id, opp_deck_id
    HAVING cnt >= 10
    ORDER BY cnt DESC
    LIMIT 40
""")
for row in cursor.fetchall():
    wr = (row[3] / row[2] * 100) if row[2] > 0 else 0
    print(f"Our Deck {row[0]} vs Opp Deck {row[1]}: Matches={row[2]}, W={row[3]}, L={row[4]}, D={row[5]}, Our WR={wr:.1f}%")

print("\n=== TOURNAMENTS SUMMARY ===")
cursor.execute("""
    SELECT id, timestamp, agent, games_per_opp, total_w, total_l, total_d, win_rate, note
    FROM tournaments
    ORDER BY id DESC
    LIMIT 25
""")
tourneys = cursor.fetchall()
for t in tourneys:
    print(f"Tourney #{t[0]} ({t[1]}): Agent='{t[2]}', Note='{t[8]}', Games/Opp={t[3]}, W={t[4]}, L={t[5]}, D={t[6]}, WR={t[7]}%")

print("\n=== MATCHUPS TABLE DETAIL ===")
cursor.execute("""
    SELECT opponent, COUNT(*) as cnt, SUM(w) as w, SUM(l) as l, SUM(d) as d, AVG(win_rate) as avg_wr, AVG(lb_score) as avg_lb
    FROM matchups
    GROUP BY opponent
    ORDER BY cnt DESC
""")
for m in cursor.fetchall():
    print(f"Opponent: '{m[0]}', Entries={m[1]}, W={m[2]}, L={m[3]}, D={m[4]}, AvgWR={m[5]:.1f}%, AvgLB={m[6]}")

print("\n=== TOP CARDS IN HIGH ELO GAMES (Elo >= 1000) ===")
cursor.execute("""
    SELECT c.id, c.name, c.category, c.stage, c.hp, 
           COUNT(mcu.match_id) as usage_cnt,
           SUM(CASE WHEN m.result = 1 AND mcu.player_side = m.our_side THEN 1 
                    WHEN m.result = -1 AND mcu.player_side != m.our_side THEN 1 ELSE 0 END) as wins_with_card
    FROM match_card_usage mcu
    JOIN cards c ON mcu.card_id = c.id
    JOIN matches m ON mcu.match_id = m.id
    GROUP BY c.id
    ORDER BY usage_cnt DESC
    LIMIT 30
""")
card_usages = cursor.fetchall()
for cu in card_usages:
    c_wr = (cu[6] / cu[5] * 100) if cu[5] > 0 else 0
    print(f"Card [{cu[0]}] {cu[1]} ({cu[2]} {cu[3]} HP:{cu[4]}): Usage={cu[5]}, Wins={cu[6]}, WinRate={c_wr:.1f}%")

conn.close()
