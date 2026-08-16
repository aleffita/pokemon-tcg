import json

with open(".agents/survey_miner_2/card_catalog.json", "r") as f:
    catalog = json.load(f)

with open(".agents/survey_miner_2/archetypes_full.json", "r") as f:
    data = json.load(f)

agent_info = data["public_agents"].get("lb826_alakazam_seok", {})
print("=======================================================")
print(" TARGET: lb826_alakazam_seok")
print("=======================================================")
for p in agent_info.get("pokemon", []):
    cid = p["id"]
    c_info = catalog.get(str(cid), {})
    print(f"  [{p['qty']}x] (ID {cid}) {c_info.get('name')} | {c_info.get('stage')} | HP: {c_info.get('hp')} | Type: {c_info.get('type')} | Weak: {c_info.get('weakness')} | Ret: {c_info.get('retreat')} | Rule: {c_info.get('rule')}")
    for m in c_info.get("moves", []):
        cost_str = f" [Cost: {m['cost']}]" if m.get('cost') else ""
        dmg_str = f" [Dmg: {m['damage']}]" if m.get('damage') else ""
        eff_str = f" - {m['effect']}" if m.get('effect') else ""
        print(f"      -> {m['name']}{cost_str}{dmg_str}{eff_str}")
        
for t in agent_info.get("trainers", []):
    cid = t["id"]
    c_info = catalog.get(str(cid), {})
    eff = " | ".join(m.get("effect", "") for m in c_info.get("moves", []) if m.get("effect"))
    print(f"  [{t['qty']}x] (ID {cid}) {c_info.get('name')} ({c_info.get('stage')}) : {eff}")
    
for e in agent_info.get("energies", []):
    cid = e["id"]
    c_info = catalog.get(str(cid), {})
    eff = " | ".join(m.get("effect", "") for m in c_info.get("moves", []) if m.get("effect"))
    print(f"  [{e['qty']}x] (ID {cid}) {c_info.get('name')} ({c_info.get('type')}) : {eff}")
