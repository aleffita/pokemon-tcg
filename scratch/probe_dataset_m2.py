"""Fast In-Memory Analysis of Top-N and Elo Threshold Episode Counts."""
import sqlite3
from collections import defaultdict
from pathlib import Path

def main():
    db_path = Path("model/results.db")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    print("Loading agent_elo_daily...")
    aed_rows = cur.execute("SELECT agent_id, day_id, elo FROM agent_elo_daily WHERE source = 'remote'").fetchall()
    
    # elo[(day_id, agent_id)] -> elo
    agent_elo = {}
    day_agents = defaultdict(list)
    for r in aed_rows:
        agent_elo[(r['day_id'], r['agent_id'])] = r['elo']
        day_agents[r['day_id']].append((r['elo'], r['agent_id']))
        
    # Precompute top-N sets per day
    for day_id in day_agents:
        day_agents[day_id].sort(key=lambda x: x[0], reverse=True)
    
    def get_top_n_set(n):
        top_set = set()
        for day_id, ag_list in day_agents.items():
            for elo_val, ag_id in ag_list[:n]:
                top_set.add((day_id, ag_id))
        return top_set

    print("Loading remote matches...")
    matches = cur.execute("""
        SELECT id, day_id, our_agent_id, opp_agent_id, result
        FROM matches
        WHERE source = 'remote'
    """).fetchall()
    total_remote = len(matches)
    print(f"Loaded {total_remote:,} matches and {len(aed_rows):,} agent_elo_daily records.")
    
    print("\n=== 1. TOP-N RANK METRICS (BOTH vs ANY) ===")
    for top_n in [10, 20, 50, 100, 200]:
        top_set = get_top_n_set(top_n)
        both_c = 0
        any_c = 0
        for m in matches:
            d_id = m['day_id']
            our_in = (d_id, m['our_agent_id']) in top_set
            opp_in = (d_id, m['opp_agent_id']) in top_set
            if our_in and opp_in:
                both_c += 1
            if our_in or opp_in:
                any_c += 1
        print(f"Top-{top_n:3d} agents: both_sides = {both_c:6,d} ({both_c/total_remote*100:5.2f}%), any_side = {any_c:6,d} ({any_c/total_remote*100:5.2f}%)")
    
    print("\n=== 2. ABSOLUTE ELO THRESHOLD METRICS ===")
    for elo_th in [900.0, 1000.0, 1050.0, 1100.0, 1150.0, 1200.0]:
        both_c = 0
        win_c = 0
        any_c = 0
        for m in matches:
            d_id = m['day_id']
            our_elo = agent_elo.get((d_id, m['our_agent_id']), 0.0)
            opp_elo = agent_elo.get((d_id, m['opp_agent_id']), 0.0)
            our_ok = our_elo >= elo_th
            opp_ok = opp_elo >= elo_th
            if our_ok and opp_ok:
                both_c += 1
            if our_ok or opp_ok:
                any_c += 1
            res = m['result']
            if (res == 1 and our_ok) or (res == -1 and opp_ok):
                win_c += 1
        print(f"Elo >= {elo_th:6.1f}: both = {both_c:6,d} ({both_c/total_remote*100:5.2f}%), win = {win_c:6,d} ({win_c/total_remote*100:5.2f}%), any = {any_c:6,d} ({any_c/total_remote*100:5.2f}%)")
    
    conn.close()

if __name__ == "__main__":
    main()
