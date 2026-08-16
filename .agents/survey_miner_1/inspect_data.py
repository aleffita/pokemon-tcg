import json

with open("/Users/alefita/workdir/pokemon-tcg/.agents/survey_miner_1/mining_data.json") as f:
    data = json.load(f)

print("=== DECK 633 PERFORMANCE & SUMMARY ===")
d633 = data["decks"]["633"]
print(f"Total cards: {d633['total_cards']}")
print(f"Performance: {d633['performance']}")
print(f"Card count distinct: {len(d633['cards'])}")
pokemon_count_633 = sum(c['quantity'] for c in d633['cards'] if c['stage'] and 'Pokémon' in c['stage'])
energy_count_633 = sum(c['quantity'] for c in d633['cards'] if c['stage'] and 'Energy' in c['stage'])
trainer_count_633 = sum(c['quantity'] for c in d633['cards'] if c['stage'] and ('Item' in c['stage'] or 'Supporter' in c['stage'] or 'Stadium' in c['stage'] or 'Tool' in c['stage']))
print(f"Breakdown: {pokemon_count_633} Pokémon, {trainer_count_633} Trainers, {energy_count_633} Energies")

print("\n=== DECK 251 PERFORMANCE & SUMMARY ===")
d251 = data["decks"]["251"]
print(f"Total cards: {d251['total_cards']}")
print(f"Performance: {d251['performance']}")
print(f"Card count distinct: {len(d251['cards'])}")
pokemon_count_251 = sum(c['quantity'] for c in d251['cards'] if c['stage'] and 'Pokémon' in c['stage'])
energy_count_251 = sum(c['quantity'] for c in d251['cards'] if c['stage'] and 'Energy' in c['stage'])
trainer_count_251 = sum(c['quantity'] for c in d251['cards'] if c['stage'] and ('Item' in c['stage'] or 'Supporter' in c['stage'] or 'Stadium' in c['stage'] or 'Tool' in c['stage']))
print(f"Breakdown: {pokemon_count_251} Pokémon, {trainer_count_251} Trainers, {energy_count_251} Energies")

print("\n=== TOP 20 HIGH ELO CARDS (MAX ELO >= 1100) ===")
for c in data["high_elo_cards"][:25]:
    print(f"ID {c['card_id']:4d}: {c['name']:<25} | MaxElo: {c['max_elo']:.1f} | AvgElo: {c['avg_elo']:.1f} | WR: {c['win_rate']*100:.1f}% | Games: {c['total_games']}")

print("\n=== TOP 15 CARD PAIRS IN HIGH ELO DECKS ===")
for p in data["top_card_pairs"][:15]:
    print(f"({p['card_1_id']:4d}) {p['card_1_name']:<22} + ({p['card_2_id']:4d}) {p['card_2_name']:<22} : {p['deck_count']} decks ({p['deck_percentage']:.1f}%)")

print("\n=== TOP 10 CARD TRIPLES IN HIGH ELO DECKS ===")
for t in data["top_card_triples"][:10]:
    print(f"{t['card_1_name']:<18} + {t['card_2_name']:<18} + {t['card_3_name']:<18} : {t['deck_count']} decks ({t['deck_percentage']:.1f}%)")
