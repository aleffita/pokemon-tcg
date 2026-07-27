"""Build card/deck stats from Kaggle replay zips.

Extracts deck choices + results from replay zips, computes card/deck Elo.
Stores only match-level data (source='remote'), NOT full replay steps.

Usage:
    uv run tcg-build-card-stats --date 2026-07-25
    uv run tcg-build-card-stats --range 2026-07-20 2026-07-25
"""
import argparse
import json
import zipfile
from pathlib import Path
from collections import Counter
from rl.results_db import ResultsDB

ROOT = Path(__file__).resolve().parent.parent

def identify_or_create_deck(db, card_ids):
    """Find matching deck or create new one. Returns deck_id."""
    card_set = Counter(card_ids)
    # Try to match against known decks
    decks = db.conn.execute("SELECT id FROM decks").fetchall()
    for (deck_id,) in decks:
        known = db.conn.execute("SELECT card_id, quantity FROM deck_cards WHERE deck_id = ?", (deck_id,)).fetchall()
        known_set = Counter({cid: qty for cid, qty in known})
        # Check overlap
        overlap = sum((card_set & known_set).values())
        total = max(sum(card_set.values()), sum(known_set.values()))
        if total > 0 and overlap / total >= 0.9:
            return deck_id
    # Create new deck
    name = f"replay_deck_{hash(frozenset(card_set.items())) % 100000}"
    deck_id = db.add_deck(name, "replay", archetype=None)
    db.add_deck_cards(deck_id, list(card_set.items()))
    return deck_id

def process_zip(db, zip_path):
    """Process a single replay zip."""
    processed = 0
    with zipfile.ZipFile(zip_path) as z:
        episodes = [n for n in z.namelist() if n.endswith('.json')]
        for ep_name in episodes:
            ep_id = Path(ep_name).stem
            # Check if already processed
            existing = db.conn.execute("SELECT id FROM matches WHERE source = 'remote' AND our_agent = ?", (ep_id,)).fetchone()
            if existing:
                continue
            try:
                data = json.loads(z.read(ep_name))
                steps = data.get('steps', [])
                rewards = data.get('rewards', [0, 0])
                if len(steps) < 2:
                    continue
                # Extract deck choices (60-card action) — find the step with deck selection
                decks = [None, None]
                for step in steps:
                    for side in (0, 1):
                        if decks[side] is not None:
                            continue
                        action = step[side].get('action', [])
                        if len(action) == 60:
                            decks[side] = [int(c) for c in action]
                if not decks[0] or not decks[1]:
                    continue
                # Determine result from our perspective (side 0)
                result = 1 if rewards[0] > rewards[1] else (-1 if rewards[0] < rewards[1] else 0)
                # Create match
                deck0_id = identify_or_create_deck(db, decks[0])
                deck1_id = identify_or_create_deck(db, decks[1])
                cur = db.conn.execute(
                    "INSERT INTO matches (source, game_index, our_agent, our_deck_id, opp_agent, opp_deck_id, our_side, result, n_steps) VALUES (?, 0, ?, ?, ?, ?, 0, ?, ?)",
                    ('remote', ep_id, deck0_id, 'opponent', deck1_id, result, len(steps)))
                match_id = cur.lastrowid
                # Record card usage
                for side, deck in enumerate(decks):
                    for card_id, qty in Counter(deck).items():
                        db.conn.execute(
                            "INSERT OR IGNORE INTO match_card_usage (match_id, card_id, player_side, quantity) VALUES (?, ?, ?, ?)",
                            (match_id, card_id, side, qty))
                processed += 1
            except Exception:
                continue
    db.conn.commit()
    return processed

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", help="Single date (YYYY-MM-DD)")
    p.add_argument("--range", nargs=2, metavar=("START", "END"), help="Date range")
    args = p.parse_args()

    zip_dir = ROOT / "data" / "bc_replay_zip"
    dates = []
    if args.date:
        dates = [args.date]
    elif args.range:
        dates = [f"2026-{m:02d}-{d:02d}" for m in range(7, 8) for d in range(1, 32)
                 if f"2026-{m:02d}-{d:02d}" >= args.range[0] and f"2026-{m:02d}-{d:02d}" <= args.range[1]]
    else:
        dates = sorted([f.stem for f in zip_dir.glob("*.zip")])

    db = ResultsDB()
    total = 0
    for date in dates:
        zip_path = zip_dir / f"{date}.zip"
        if not zip_path.exists():
            print(f"  {date}: not found, skipping")
            continue
        n = process_zip(db, zip_path)
        total += n
        print(f"  {date}: {n} episodes processed")

    print(f"\nTotal: {total} episodes processed")

    if total > 0:
        print("Computing card Elo...")
        db.compute_card_elo(source='remote')
        print("Computing deck Elo...")
        db.compute_deck_elo(source='remote')

        top = db.get_top_cards(10)
        print("\nTop 10 cards by Elo:")
        for c in top:
            print(f"  {c['name']}: {c['elo']:.0f}")

    db.close()

if __name__ == "__main__":
    main()
