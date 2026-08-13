import sqlite3
import pandas as pd

def main():
    conn = sqlite3.connect("model/results.db")
    
    # Top Meta Decks identified earlier
    target_decks = (251, 440, 633, 484, 582, 23, 177)
    
    query = f"""
    SELECT 
        d1.id as Deck_A, 
        d2.id as Deck_B, 
        SUM(CASE WHEN m.result = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as WinRate,
        COUNT(*) as Games
    FROM matches m
    JOIN decks d1 ON m.our_deck_id = d1.id
    JOIN decks d2 ON m.opp_deck_id = d2.id
    WHERE d1.id IN {target_decks} AND d2.id IN {target_decks}
    AND d1.id != d2.id
    AND m.source = 'remote'
    GROUP BY d1.id, d2.id
    HAVING Games > 10
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        if df.empty:
            print("No head-to-head matches found among top decks with > 10 games.")
            return
            
        # Pivot table for matrix view
        matrix = df.pivot(index='Deck_A', columns='Deck_B', values='WinRate').round(3)
        print("=== ORTHOGONAL META MATRIX (WIN RATE OF DECK A vs DECK B) ===")
        print(matrix.fillna("-"))
        print("\n=== MATCH VOLUME ===")
        vol_matrix = df.pivot(index='Deck_A', columns='Deck_B', values='Games')
        print(vol_matrix.fillna("-"))
        
    except Exception as e:
        print(f"Probe Error: {e}")

if __name__ == "__main__":
    main()
