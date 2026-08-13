import sqlite3
import pandas as pd

def main():
    conn = sqlite3.connect("model/results.db")
    
    # 1. Temporal Drift of Top Decks (Last 7 Days)
    q_time = """
    SELECT 
        d.id as deck_id, 
        dy.date as date,
        SUM(CASE WHEN m.result = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_rate,
        COUNT(*) as games
    FROM matches m
    JOIN decks d ON m.our_deck_id = d.id
    JOIN days dy ON m.day_id = dy.id
    WHERE m.source = 'remote' AND d.id IN (251, 23, 633)
    GROUP BY 1, 2
    ORDER BY dy.date ASC
    """
    try:
        df_time = pd.read_sql_query(q_time, conn)
        print("=== TEMPORAL DRIFT (WIN RATE EVOLUTION BY DAY) ===")
        print(df_time.pivot(index='date', columns='deck_id', values='win_rate').round(3).fillna("-"))
        print("\n=== MATCH VOLUME EVOLUTION ===")
        print(df_time.pivot(index='date', columns='deck_id', values='games').fillna("-"))
    except Exception as e:
        print(f"Time Probe Error: {e}")

    # 2. Match Length Distribution (Pacing) by Elo Tier
    q_length = """
    WITH elo_tiers AS (
        SELECT deck_id,
            CASE 
                WHEN elo >= 1100 THEN '1_Elite (1100+)'
                WHEN elo >= 1000 THEN '2_Mid (1000-1099)'
                ELSE '3_Low (<1000)'
            END as tier
        FROM deck_elo WHERE source = 'remote'
    )
    SELECT 
        et.tier,
        ROUND(AVG(m.n_steps), 1) as avg_steps,
        MIN(m.n_steps) as min_steps,
        MAX(m.n_steps) as max_steps,
        COUNT(*) as total_matches
    FROM matches m
    JOIN elo_tiers et ON m.our_deck_id = et.deck_id
    WHERE m.source = 'remote' AND m.n_steps > 0
    GROUP BY 1 ORDER BY 1
    """
    try:
        df_len = pd.read_sql_query(q_length, conn)
        print("\n=== MATCH PACING BY ELO TIER (N_STEPS) ===")
        print(df_len.to_string(index=False))
    except Exception as e:
        print(f"Pacing Probe Error: {e}")

    # 3. Elite Pool Size (Filtering Out the Garbage)
    q_size = """
    SELECT 
        COUNT(*) as total_elite_matches,
        ROUND(AVG(n_steps), 0) as avg_elite_steps
    FROM matches m
    JOIN deck_elo de ON m.our_deck_id = de.deck_id
    WHERE m.source = 'remote' AND de.elo >= 1100
    """
    try:
        df_size = pd.read_sql_query(q_size, conn)
        print("\n=== THE ELITE POOL (ELO >= 1100) ===")
        print(df_size.to_string(index=False))
    except Exception as e:
        print(f"Pool Probe Error: {e}")

if __name__ == "__main__":
    main()
