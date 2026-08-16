import json

with open(".agents/survey_miner_2/panels_compiled.json", "r") as f:
    panels = json.load(f)

for p_title in [
    "Panel 1: Control & Energy Punishment (Alakazam / Hand Disruption)",
    "Panel 2: Top Leaderboard Fast Aggro (Mega Lucario & Multipliers)",
    "Panel 3: Spread Damage & Bench Snipes (Crustle, Lucario, Dragapult ex)"
]:
    p_data = panels[p_title]
    print(f"\n================================================================================")
    print(f" {p_title}")
    print(f"================================================================================")
    print("POKEMON:")
    for p in p_data.get("pokemon", []):
        print(f"  [{p['qty']}x] (ID {p['id']}) {p['name']} | {p['stage']} | HP: {p['hp']} | Type: {p['type']} | Weak: {p['weakness']} | Ret: {p['retreat']} | Rule: {p['rule']}")
        for m in p.get("moves", []):
            cost_str = f" [Cost: {m['cost']}]" if m.get('cost') else ""
            dmg_str = f" [Dmg: {m['damage']}]" if m.get('damage') else ""
            eff_str = f" - {m['effect']}" if m.get('effect') else ""
            print(f"      -> {m['name']}{cost_str}{dmg_str}{eff_str}")
