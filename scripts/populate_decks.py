"""Populate decks and deck_cards tables from all deck sources.

Reads deck definitions from:
  - agent/deck.csv           (our current agent)
  - public_agents/*/deck.csv (public competition agents)
  - rl/deck/decks.py         (official starter decks)
  - rl/deck/decks_kaggle.py  (Kaggle-mined decks)
  - rl/deck/decks_meta.py    (Champions League Aichi + live-ladder meta)
  - rl/deck/decks_generated.py (auto-generated archetypes)

Skips decks with != 60 cards. Uses INSERT OR IGNORE so re-running is safe.

Usage:
    uv run python scripts/populate_decks.py
"""
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rl.results_db import ResultsDB

ROOT = Path(__file__).resolve().parent.parent


def read_csv_deck(path):
    """Read a deck.csv file (one card ID per line), return list of card IDs."""
    ids = []
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(",")
            if line and line.isdigit():
                ids.append(int(line))
    return ids


def count_cards(card_ids):
    """Count occurrences of each card. Returns list of (card_id, quantity)."""
    return list(Counter(card_ids).items())


def add_deck_from_flat(db, name, source, card_ids, archetype=None):
    """Add a deck from a flat list of 60 card IDs. Returns True if added."""
    if len(card_ids) != 60:
        return False
    deck_id = db.add_deck(name, source, archetype=archetype)
    if deck_id is None:
        return False
    db.add_deck_cards(deck_id, count_cards(card_ids))
    return True


def main():
    db = ResultsDB()

    # Idempotency: skip if already populated
    count = db.conn.execute("SELECT COUNT(*) FROM decks").fetchone()[0]
    if count > 0:
        sample = db.conn.execute(
            "SELECT id, name, source, archetype FROM decks LIMIT 5"
        ).fetchall()
        print(f"Decks table already has {count} entries. Skipping.")
        for r in sample:
            print(f"  {r[0]}: {r[1]} ({r[2]}) -- {r[3]}")
        db.close()
        return

    decks_added = 0

    # ---- 1. Our agent's deck (CSV) ----
    agent_deck = ROOT / "agent" / "deck.csv"
    if agent_deck.exists():
        card_ids = read_csv_deck(agent_deck)
        if add_deck_from_flat(db, "agent_current", "agent", card_ids):
            decks_added += 1
            print(f"  agent_current: {len(card_ids)} cards")
        else:
            print(f"  SKIP agent_current: {len(card_ids)} cards (need 60)")

    # ---- 2. Public agents (CSV) ----
    public_dir = ROOT / "public_agents"
    if public_dir.exists():
        for deck_csv in sorted(public_dir.rglob("deck.csv")):
            parent = deck_csv.parent
            # Build name: prefix with parent category for uniqueness
            if parent.parent.name == "starters":
                name = f"starter_{parent.name}"
                source = "starter"
            elif parent.parent.name == "submissions":
                name = f"sub_{parent.name}"
                source = "submission"
            else:
                name = parent.name
                source = "public_agent"

            card_ids = read_csv_deck(deck_csv)
            if add_deck_from_flat(db, name, source, card_ids):
                decks_added += 1
                print(f"  {name}: {len(card_ids)} cards")
            else:
                print(f"  SKIP {name}: {len(card_ids)} cards (need 60)")

    # ---- 3. Official starter decks (rl/deck/decks.py) ----
    try:
        from rl.deck.decks import DECKS, DECK_NAMES
        for deck_name in DECK_NAMES:
            card_ids = DECKS[deck_name]
            name = f"starter_{deck_name}"
            if add_deck_from_flat(db, name, "starter", card_ids, archetype=deck_name):
                decks_added += 1
                print(f"  {name}: {len(card_ids)} cards")
            else:
                print(f"  SKIP {name}: {len(card_ids)} cards (need 60)")
    except Exception as e:
        print(f"  Warning: could not load rl/deck/decks.py: {e}")

    # ---- 4. Kaggle-mined decks (rl/deck/decks_kaggle.py) ----
    # Import before decks_meta because decks_meta merges KAGGLE_DECKS into META
    # at import time. We want the raw kaggle entries with a distinct source tag.
    try:
        from rl.deck.decks_kaggle import KAGGLE_DECKS
        for archetype, card_ids in KAGGLE_DECKS.items():
            if add_deck_from_flat(db, archetype, "kaggle", card_ids, archetype=archetype):
                decks_added += 1
                print(f"  {archetype}: {len(card_ids)} cards")
            else:
                print(f"  SKIP {archetype}: {len(card_ids)} cards (need 60)")
    except Exception as e:
        print(f"  Warning: could not load rl/deck/decks_kaggle.py: {e}")

    # ---- 5. Meta decks (rl/deck/decks_meta.py) ----
    # META already includes KAGGLE_DECKS via self-merge at module level.
    # Archetype keys are already prefixed (meta_aichi_*, meta2_*, meta3_*, meta4_*)
    # so we use them directly as names. Duplicate names are silently skipped by
    # INSERT OR IGNORE.
    try:
        from rl.deck.decks_meta import META, META2, META3, META4
        for _meta_label, meta_dict in [
            ("aichi", META),
            ("meta2", META2),
            ("meta3", META3),
            ("meta4", META4),
        ]:
            for archetype, card_ids in meta_dict.items():
                if add_deck_from_flat(db, archetype, "meta", card_ids, archetype=archetype):
                    decks_added += 1
                    print(f"  {archetype}: {len(card_ids)} cards")
                else:
                    print(f"  SKIP {archetype}: {len(card_ids)} cards (need 60)")
    except Exception as e:
        print(f"  Warning: could not load rl/deck/decks_meta.py: {e}")

    # ---- 6. Auto-generated archetypes (rl/deck/decks_generated.py) ----
    try:
        from rl.deck.decks_generated import GENERATED
        for archetype, card_ids in GENERATED.items():
            name = f"generated_{archetype}"
            if add_deck_from_flat(db, name, "generated", card_ids, archetype=archetype):
                decks_added += 1
                print(f"  {name}: {len(card_ids)} cards")
            else:
                print(f"  SKIP {name}: {len(card_ids)} cards (need 60)")
    except Exception as e:
        print(f"  Warning: could not load rl/deck/decks_generated.py: {e}")

    # ---- 7. Training deck sets (rl/deck/decks_train.py) ----
    # These overlap heavily with META/META2/META3/META4 (already added above).
    # Only add the k-series entries (k01..k50) which are unique to this module.
    try:
        from rl.deck.decks_train import TRAIN_TOP50
        for archetype, card_ids in TRAIN_TOP50.items():
            if archetype.startswith("k") and archetype[1:3].isdigit():
                name = f"train_{archetype}"
                if add_deck_from_flat(db, name, "train", card_ids, archetype=archetype):
                    decks_added += 1
                    print(f"  {name}: {len(card_ids)} cards")
                else:
                    print(f"  SKIP {name}: {len(card_ids)} cards (need 60)")
    except Exception as e:
        print(f"  Warning: could not load rl/deck/decks_train.py: {e}")

    # ---- Summary ----
    total = db.conn.execute("SELECT COUNT(*) FROM decks").fetchone()[0]
    by_source = db.conn.execute(
        "SELECT source, COUNT(*) FROM decks GROUP BY source ORDER BY source"
    ).fetchall()

    print(f"\n=== Summary ===")
    print(f"Decks added this run: {decks_added}")
    print(f"Total decks in DB:    {total}")
    print(f"By source:")
    for source, cnt in by_source:
        print(f"  {source}: {cnt}")

    db.close()


if __name__ == "__main__":
    main()
