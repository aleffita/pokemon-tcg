import csv
from pathlib import Path

en_card_data_path = Path("public_agents/submissions/latest-submission-300elo/EN_Card_Data.csv")
with open(en_card_data_path, mode='r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    print("EN_Card_Data.csv Headers:")
    print(header)
    first_row = next(reader)
    print("\nFirst row sample:")
    for h, v in zip(header, first_row):
        if v:
            print(f"  {h}: {v}")

# Search specific cards
target_ids = [678, 677, 121, 120, 743, 742, 96, 723, 269, 140, 1182, 1086, 1231, 1225, 1197, 1213, 1081]
with open(en_card_data_path, mode='r', encoding='utf-8') as f:
    dict_reader = csv.DictReader(f)
    for row in dict_reader:
        cid = int(row.get("id", -1))
        if cid in target_ids:
            print(f"\n--- Card ID {cid}: {row.get('name')} ---")
            for k, v in row.items():
                if v and k not in ['id', 'name']:
                    print(f"  {k}: {v}")
