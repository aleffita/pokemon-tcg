import sqlite3

def main():
    conn = sqlite3.connect("model/results.db")
    c = conn.cursor()
    c.execute("SELECT card_id, quantity FROM deck_cards WHERE deck_id = 251 ORDER BY card_id")
    rows = c.fetchall()
    
    cards = []
    for card_id, qty in rows:
        for _ in range(qty):
            cards.append(str(card_id))
            
    assert len(cards) == 60, f"Deck has {len(cards)} cards instead of 60!"
    
    out_path = "scratch/hero_staging/deck.csv"
    with open(out_path, "w") as f:
        f.write("\n".join(cards) + "\n")
        
    print(f"Success! Wrote {len(cards)} cards to {out_path}.")

if __name__ == "__main__":
    main()
