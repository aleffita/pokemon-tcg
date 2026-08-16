import json

with open(".agents/survey_miner_2/panels_compiled.json", "r") as f:
    panels = json.load(f)

for p_title, p_data in panels.items():
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
    print("\nTRAINERS:")
    for t in p_data.get("trainers", []):
        effect_summary = ""
        if t.get("moves"):
            effect_summary = " : " + " | ".join(m.get("effect", "") for m in t["moves"] if m.get("effect"))
        print(f"  [{t['qty']}x] (ID {t['id']}) {t['name']} ({t['category']}){effect_summary[:120]}")
    print("\nENERGIES:")
    for e in p_data.get("energies", []):
        effect_summary = ""
        if e.get("moves"):
            effect_summary = " : " + " | ".join(m.get("effect", "") for m in e["moves"] if m.get("effect"))
        print(f"  [{e['qty']}x] (ID {e['id']}) {e['name']} ({e['type']}){effect_summary[:120]}")
