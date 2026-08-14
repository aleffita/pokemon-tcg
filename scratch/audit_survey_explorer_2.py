import sqlite3
from collections import Counter

DB_PATH = "model/results.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    print("=== FOREIGN KEY AUDIT ===")
    fk_errors = c.execute("PRAGMA foreign_key_check").fetchall()
    print(f"Total FK Error Rows: {len(fk_errors)}")
    
    error_tables = Counter(r[0] for r in fk_errors)
    for table, cnt in error_tables.most_common():
        print(f"  Table: {table:25s} | Error count: {cnt:,d}")
        
    print("\nSample FK errors per table:")
    for table in error_tables.keys():
        samples = [r for r in fk_errors if r[0] == table][:3]
        for s in samples:
            print(f"  Table: {s[0]}, RowId: {s[1]}, TargetTable: {s[2]}, FkIdx: {s[3]}")
            
    print("\n=== DECK #633 CARD COMPOSITION ===")
    deck_cards = c.execute("""
        SELECT dc.card_id, dc.quantity, c.name, c.category, c.stage, c.hp, c.energy_type
        FROM deck_cards dc
        JOIN cards c ON dc.card_id = c.id
        WHERE dc.deck_id = 633
        ORDER BY dc.quantity DESC, c.name
    """).fetchall()
    
    total_cards = sum(r['quantity'] for r in deck_cards)
    print(f"Deck 633 Total Cards: {total_cards} across {len(deck_cards)} unique cards:")
    for r in deck_cards:
        print(f"  - {r['quantity']}x {r['name']} (ID: {r['card_id']}, {r['category'] or ''}, Stage: {r['stage'] or ''}, HP: {r['hp'] or ''}, Type: {r['energy_type'] or ''})")

    print("\n=== TOURNAMENT BENCHMARK RUNS (OVERALL VS FIRST_SUB) ===")
    runs = c.execute("""
        SELECT t.id, t.timestamp, t.agent, t.win_rate, t.total_w, t.total_l, t.total_d, t.total_time_s
        FROM tournaments t
        ORDER BY t.id DESC LIMIT 15
    """).fetchall()
    for r in runs:
        print(f"  ID {r['id']:3d} | {r['timestamp']} | Agent: {r['agent'][:50]:50s} | WR: {r['win_rate']:6.2f}% ({r['total_w']}W/{r['total_l']}L/{r['total_d']}D)")

    conn.close()

if __name__ == "__main__":
    main()
