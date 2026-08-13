import sqlite3
import pandas as pd

def main():
    conn = sqlite3.connect("model/results.db")
    target_decks = (251, 440, 633, 484, 582, 23, 177)
    
    # 1. Total wins by top decks (Potential Dataset Size)
    q_wins = f"""
    SELECT 
        CASE WHEN result = 1 THEN our_deck_id ELSE opp_deck_id END as winning_deck,
        COUNT(*) as wins
    FROM matches
    WHERE source = 'remote'
    AND (
        (our_deck_id IN {target_decks} AND result = 1) OR
        (opp_deck_id IN {target_decks} AND result = -1)
    )
    GROUP BY 1 ORDER BY wins DESC
    """
    df_wins = pd.read_sql_query(q_wins, conn)
    
    print("=== POTENTIAL TRAINING SAMPLES (WINS BY TOP DECKS) ===")
    print(df_wins)
    print(f"--> Total Clean Samples: {df_wins['wins'].sum()}\n")
    
    # 2. Reconstruct Elo Hierarchy (Opponent Bands)
    q_matchups = f"""
    WITH deck_bands AS (
        SELECT deck_id, elo,
            CASE 
                WHEN elo >= 1200 THEN '1_Gold (1200+)'
                WHEN elo >= 1100 THEN '2_Silver (1100-1199)'
                WHEN elo >= 1000 THEN '3_Bronze (1000-1099)'
                ELSE '4_Wood (<1000)'
            END as band
        FROM deck_elo WHERE source = 'remote'
    )
    SELECT 
        t.our_deck as Top_Deck,
        db.band as Opp_Elo_Band,
        SUM(t.win) * 1.0 / COUNT(*) as WinRate,
        COUNT(*) as Games
    FROM (
        -- As "our_deck"
        SELECT our_deck_id as our_deck, opp_deck_id as opp_deck, CASE WHEN result = 1 THEN 1 ELSE 0 END as win
        FROM matches WHERE source = 'remote' AND our_deck_id IN {target_decks}
        UNION ALL
        -- As "opp_deck"
        SELECT opp_deck_id as our_deck, our_deck_id as opp_deck, CASE WHEN result = -1 THEN 1 ELSE 0 END as win
        FROM matches WHERE source = 'remote' AND opp_deck_id IN {target_decks}
    ) t
    JOIN deck_bands db ON t.opp_deck = db.deck_id
    GROUP BY 1, 2
    """
    try:
        df_matchups = pd.read_sql_query(q_matchups, conn)
        print("=== WIN RATE OF TOP DECKS vs ELO HIERARCHY MURALHAS ===")
        wr_matrix = df_matchups.pivot(index='Top_Deck', columns='Opp_Elo_Band', values='WinRate').round(3)
        print(wr_matrix.fillna("-"))
        print("\n=== MATCH VOLUME (TOP DECKS vs ELO HIERARCHY) ===")
        vol_matrix = df_matchups.pivot(index='Top_Deck', columns='Opp_Elo_Band', values='Games')
        print(vol_matrix.fillna(0).astype(int))
    except Exception as e:
        print(f"Hierarchy Probe Error: {e}")

if __name__ == "__main__":
    main()
