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
import sys
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


def add_deck_from_flat(db, name, source, card_ids, archetype=None):
    """Add a deck from a flat list of 60 card IDs. Returns True if added."""
    if len(card_ids) != 60:
        return False
    source_code = {
        "agent": "custom",
        "generated": "custom",
        "kaggle": "custom",
        "meta": "custom",
        "train": "custom",
        "remote": "replay",
    }.get(source, source)
    db.get_or_create_deck(
        card_ids,
        source=source_code,
        name=name,
        archetype=archetype,
    )
    return True


def populate_decks(
    db_or_path: ResultsDB | str | Path | None = None,
    *,
    root: str | Path = ROOT,
    skip_existing: bool = True,
    strict: bool = False,
    output=print,
) -> int:
    """Populate every canonical deck source into a database or database path."""

    owns_db = not isinstance(db_or_path, ResultsDB)
    db = (
        db_or_path
        if isinstance(db_or_path, ResultsDB)
        else ResultsDB(db_or_path)
    )
    source_root = Path(root)

    # Idempotency: skip if already populated
    count = db.conn.execute("SELECT COUNT(*) FROM decks").fetchone()[0]
    if count > 0 and skip_existing:
        sample = db.conn.execute(
            "SELECT id, name, source, archetype FROM decks LIMIT 5"
        ).fetchall()
        output(f"Decks table already has {count} entries. Skipping.")
        for r in sample:
            output(f"  {r[0]}: {r[1]} ({r[2]}) -- {r[3]}")
        if owns_db:
            db.close()
        return 0
    if count > 0:
        if owns_db:
            db.close()
        raise ValueError(
            "decks table is not empty; refusing non-idempotent population"
        )

    decks_added = 0
    from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn
    from rich.console import Console
    console = Console()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_agent = progress.add_task("[cyan]Agent Deck", total=1)
        task_public = progress.add_task("[cyan]Public Agents", total=None)
        task_starters = progress.add_task("[cyan]Starters", total=None)
        task_kaggle = progress.add_task("[cyan]Kaggle", total=None)
        task_meta = progress.add_task("[cyan]Meta", total=None)
        task_gen = progress.add_task("[cyan]Generated", total=None)
        task_train = progress.add_task("[cyan]Train", total=None)

        # ---- 1. Our agent's deck (CSV) ----
        agent_deck = source_root / "agent" / "deck.csv"
        if agent_deck.exists():
            card_ids = read_csv_deck(agent_deck)
            if add_deck_from_flat(db, "agent_current", "agent", card_ids):
                decks_added += 1
        progress.update(task_agent, advance=1)

        # ---- 2. Public agents (CSV) ----
        public_dir = source_root / "public_agents"
        if public_dir.exists():
            csvs = list(public_dir.rglob("deck.csv"))
            progress.update(task_public, total=len(csvs))
            for deck_csv in csvs:
                parent = deck_csv.parent
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
                progress.update(task_public, advance=1)
        else:
            progress.update(task_public, total=0)

        # ---- 3. Official starter decks (rl/deck/decks.py) ----
        try:
            from rl.deck.decks import DECKS, DECK_NAMES
            progress.update(task_starters, total=len(DECK_NAMES))
            for deck_name in sorted(DECK_NAMES):
                card_ids = DECKS[deck_name]
                name = f"starter_{deck_name}"
                if add_deck_from_flat(db, name, "starter", card_ids, archetype=deck_name):
                    decks_added += 1
                progress.update(task_starters, advance=1)
        except Exception as e:
            if strict:
                raise
            output(f"  Warning: could not load rl/deck/decks.py: {e}")
            progress.update(task_starters, total=0)

        # ---- 4. Kaggle-mined decks (rl/deck/decks_kaggle.py) ----
        try:
            from rl.deck.decks_kaggle import KAGGLE_DECKS
            progress.update(task_kaggle, total=len(KAGGLE_DECKS))
            for archetype, card_ids in sorted(KAGGLE_DECKS.items()):
                if add_deck_from_flat(db, archetype, "kaggle", card_ids, archetype=archetype):
                    decks_added += 1
                progress.update(task_kaggle, advance=1)
        except Exception as e:
            if strict:
                raise
            output(f"  Warning: could not load rl/deck/decks_kaggle.py: {e}")
            progress.update(task_kaggle, total=0)

        # ---- 5. Meta decks (rl/deck/decks_meta.py) ----
        try:
            from rl.deck.decks_meta import META, META2, META3, META4
            total_meta = len(META) + len(META2) + len(META3) + len(META4)
            progress.update(task_meta, total=total_meta)
            for _meta_label, meta_dict in [
                ("aichi", META),
                ("meta2", META2),
                ("meta3", META3),
                ("meta4", META4),
            ]:
                for archetype, card_ids in sorted(meta_dict.items()):
                    if add_deck_from_flat(db, archetype, "meta", card_ids, archetype=archetype):
                        decks_added += 1
                    progress.update(task_meta, advance=1)
        except Exception as e:
            if strict:
                raise
            output(f"  Warning: could not load rl/deck/decks_meta.py: {e}")
            progress.update(task_meta, total=0)

        # ---- 6. Auto-generated archetypes (rl/deck/decks_generated.py) ----
        try:
            from rl.deck.decks_generated import GENERATED
            progress.update(task_gen, total=len(GENERATED))
            for archetype, card_ids in sorted(GENERATED.items()):
                name = f"generated_{archetype}"
                if add_deck_from_flat(db, name, "generated", card_ids, archetype=archetype):
                    decks_added += 1
                progress.update(task_gen, advance=1)
        except Exception as e:
            if strict:
                raise
            output(f"  Warning: could not load rl/deck/decks_generated.py: {e}")
            progress.update(task_gen, total=0)

        # ---- 7. Training deck sets (rl/deck/decks_train.py) ----
        try:
            from rl.deck.decks_train import TRAIN_TOP50
            targets = [a for a in TRAIN_TOP50 if a.startswith("k") and a[1:3].isdigit()]
            progress.update(task_train, total=len(targets))
            for archetype in sorted(targets):
                card_ids = TRAIN_TOP50[archetype]
                name = f"train_{archetype}"
                if add_deck_from_flat(db, name, "train", card_ids, archetype=archetype):
                    decks_added += 1
                progress.update(task_train, advance=1)
        except Exception as e:
            if strict:
                raise
            output(f"  Warning: could not load rl/deck/decks_train.py: {e}")
            progress.update(task_train, total=0)

    # ---- Summary ----
    total = db.conn.execute("SELECT COUNT(*) FROM decks").fetchone()[0]
    by_source = db.conn.execute(
        "SELECT source, COUNT(*) FROM decks GROUP BY source ORDER BY source"
    ).fetchall()

    console.print("\n[bold]=== Summary ===[/]")
    console.print(f"Decks added this run: {decks_added}")
    console.print(f"Total decks in DB:    {total}")
    console.print("By source:")
    for source_val, cnt in by_source:
        console.print(f"  {source_val}: {cnt}")

    if owns_db:
        db.close()
    return decks_added


def main():
    populate_decks()


if __name__ == "__main__":
    main()
