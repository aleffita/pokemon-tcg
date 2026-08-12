import sqlite3

def print_deck(cursor, deck_id):
    cursor.execute('''
        SELECT c.name, dc.quantity 
        FROM deck_cards dc
        JOIN cards c ON dc.card_id = c.id
        WHERE dc.deck_id = ?
        ORDER BY c.name
    ''', (deck_id,))
    rows = cursor.fetchall()
    print(f"--- DECK {deck_id} ---")
    if not rows:
        print("No cards found.")
    for r in rows:
        print(f"{r[1]}x {r[0]}")
    print("")

with sqlite3.connect('model/results.db') as conn:
    c = conn.cursor()
    # Let's check tables just in case
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    print(f"Tables: {[r[0] for r in c.fetchall()]}")
    
    print_deck(c, 633)
    print_deck(c, 21)
    
    # Also find variant #440 which deck 21 beat heavily
    print_deck(c, 440)
