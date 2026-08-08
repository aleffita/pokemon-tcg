"""Build exact deck/card statistics from Kaggle replay archives.

Remote archives contribute match-level observations only. Replay payloads are
not persisted. Official episode ids are retained when explicitly exposed;
otherwise the source byte digest is the idempotency identity.

Usage:
    uv run tcg-build-card-stats --date 2026-07-25
    uv run tcg-build-card-stats --range 2026-07-20 2026-07-25
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence
import zipfile

from rl.results_db import (
    IdempotencyConflict,
    MatchParticipant,
    ResultsDB,
    deck_fingerprint,
)


ROOT = Path(__file__).resolve().parent.parent

# Names under which our own submissions appear in Kaggle replay metadata.
# Mirrors scripts/bc/build_bc_dataset.py:SELF_ALIASES.
SELF_AGENT_ALIASES = frozenset({"FitaLabs", "Alef Oliveira"})


def identify_or_create_deck(db: ResultsDB, card_ids: list[int]) -> int:
    """Resolve only an exactly equal deterministic deck composition."""

    fingerprint = deck_fingerprint(card_ids)
    return db.get_or_create_deck(
        card_ids,
        source="replay",
        name=f"replay_deck_{fingerprint[:16]}",
    )


def _external_episode_id(payload: dict) -> str | None:
    """Read only source fields whose semantics explicitly identify an episode."""

    info = payload.get("info")
    if isinstance(info, dict):
        value = info.get("EpisodeId")
        if value is not None and str(value).strip():
            return str(value)
    for key in ("external_episode_id", "episode_id"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _reported_team_names(payload: dict) -> tuple[str | None, str | None]:
    info = payload.get("info")
    if not isinstance(info, dict):
        return None, None
    team_names = info.get("TeamNames")
    if isinstance(team_names, list) and len(team_names) >= 2:
        return (
            str(team_names[0]) if team_names[0] else None,
            str(team_names[1]) if team_names[1] else None,
        )
    agents = info.get("Agents")
    if isinstance(agents, list) and len(agents) >= 2:
        names = []
        for agent in agents[:2]:
            name = agent.get("Name") if isinstance(agent, dict) else None
            names.append(str(name) if name else None)
        return names[0], names[1]
    return None, None


def _extract_decks(steps: list) -> tuple[list[int], list[int]] | None:
    decks: list[list[int] | None] = [None, None]
    for step in steps:
        if not isinstance(step, list) or len(step) < 2:
            continue
        for side in (0, 1):
            if decks[side] is not None or not isinstance(step[side], dict):
                continue
            action = step[side].get("action")
            if isinstance(action, list) and len(action) == 60:
                decks[side] = [int(card_id) for card_id in action]
    if decks[0] is None or decks[1] is None:
        return None
    return decks[0], decks[1]


def _outcomes(rewards: Sequence[float]) -> tuple[int, int]:
    if rewards[0] > rewards[1]:
        return 1, -1
    if rewards[0] < rewards[1]:
        return -1, 1
    return 0, 0


def process_zip(
    db: ResultsDB,
    zip_path: str | Path,
    *,
    archive_date: str | None = None,
) -> int:
    """Idempotently ingest match-level observations from one replay archive."""

    archive = Path(zip_path)
    observed_date = archive_date or archive.stem
    processed = 0
    with zipfile.ZipFile(archive) as replay_zip:
        episodes = sorted(
            name for name in replay_zip.namelist() if name.endswith(".json")
        )
        for member_name in episodes:
            raw_payload = replay_zip.read(member_name)
            source_digest = hashlib.sha256(raw_payload).hexdigest()
            payload = json.loads(raw_payload)
            if not isinstance(payload, dict):
                continue
            external_episode_id = _external_episode_id(payload)
            if external_episode_id is not None:
                existing = db.conn.execute(
                    """
                    SELECT id, source_observation_digest FROM matches
                    WHERE source = 'remote' AND external_episode_id = ?
                    """,
                    (external_episode_id,),
                ).fetchone()
            else:
                existing = db.conn.execute(
                    """
                    SELECT id FROM matches
                    WHERE source = 'remote'
                      AND source_observation_digest = ?
                    """,
                    (source_digest,),
                ).fetchone()
            if existing is not None:
                if (
                    external_episode_id is not None
                    and existing["source_observation_digest"] != source_digest
                ):
                    raise IdempotencyConflict(
                        "official episode id was reused for different replay bytes"
                    )
                continue

            steps = payload.get("steps")
            rewards = payload.get("rewards")
            if (
                not isinstance(steps, list)
                or len(steps) < 2
                or not isinstance(rewards, list)
                or len(rewards) < 2
                or not all(
                    isinstance(reward, (int, float))
                    for reward in rewards[:2]
                )
            ):
                continue
            decks = _extract_decks(steps)
            if decks is None:
                continue

            team_names = _reported_team_names(payload)
            outcomes = _outcomes(rewards)
            participants = []
            agent_ids: list[int | None] = [None, None]
            for side in (0, 1):
                deck_id = identify_or_create_deck(db, decks[side])
                team_id = db.get_or_create_team(
                    team_names[side],
                    observation_key=f"{source_digest}:seat:{side}",
                )
                # Replays do not expose an official submission id. This is an
                # explicitly observed identity, not a claim that every daily
                # observation is one official Kaggle submission lineage.
                submission_id = db.observe_submission(
                    team_id,
                    [deck_id],
                    source="remote",
                    archive_date=observed_date,
                )
                # Only known agent names are registered. An unreported name
                # would otherwise collapse unrelated opponents into a single
                # "Unidentified team" agent row.
                agent_name = team_names[side]
                if agent_name:
                    agent_ids[side] = db.upsert_agent(
                        name=agent_name,
                        team_id=team_id,
                        is_self=agent_name in SELF_AGENT_ALIASES,
                    )
                participants.append(
                    MatchParticipant(
                        seat=side,
                        team_id=team_id,
                        submission_id=submission_id,
                        deck_id=deck_id,
                        reward=float(rewards[side]),
                        outcome=outcomes[side],
                    )
                )

            day_id = db.register_day(observed_date)
            db.record_match(
                source="remote",
                external_episode_id=external_episode_id,
                source_observation_digest=source_digest,
                archive_date=observed_date,
                archive_member=member_name,
                n_steps=len(steps),
                participants=participants,
                our_agent_id=agent_ids[0],
                opp_agent_id=agent_ids[1],
            )
            for agent_id in agent_ids:
                if agent_id is not None:
                    db.touch_agent_seen(agent_id, day_id)
            processed += 1
    return processed


def populate_replays(
    db_or_path: ResultsDB | str | Path | None = None,
    *,
    zip_paths: Sequence[str | Path] | None = None,
    replay_zip_dir: str | Path | None = None,
    output=print,
) -> int:
    """Populate canonical remote replay observations into a DB or DB path."""

    owns_db = not isinstance(db_or_path, ResultsDB)
    db = (
        db_or_path
        if isinstance(db_or_path, ResultsDB)
        else ResultsDB(db_or_path)
    )
    replay_root = (
        Path(replay_zip_dir)
        if replay_zip_dir is not None
        else ROOT / "data" / "bc_replay_zip"
    )
    sources = (
        sorted((Path(path) for path in zip_paths), key=lambda path: str(path))
        if zip_paths is not None
        else sorted(replay_root.glob("*.zip"))
    )
    if not sources:
        if owns_db:
            db.close()
        raise FileNotFoundError(f"no replay ZIPs found under {replay_root}")

    total = 0
    try:
        for archive in sources:
            if not archive.is_file():
                raise FileNotFoundError(f"replay ZIP does not exist: {archive}")
            count = process_zip(db, archive, archive_date=archive.stem)
            total += count
            output(f"  {archive.stem}: {count} episodes processed")
        output(f"\nTotal: {total} episodes processed")
        output("Computing card Elo...")
        db.compute_card_elo(source="remote")
        output("Computing deck Elo...")
        db.compute_deck_elo(source="remote")
        return total
    finally:
        if owns_db:
            db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date", help="Single date (YYYY-MM-DD)")
    parser.add_argument(
        "--range", nargs=2, metavar=("START", "END"), help="Date range"
    )
    args = parser.parse_args()

    zip_dir = ROOT / "data" / "bc_replay_zip"
    if args.date:
        dates = [args.date]
    elif args.range:
        from datetime import date, timedelta
        start = date.fromisoformat(args.range[0])
        end = date.fromisoformat(args.range[1])
        dates = []
        cursor = start
        while cursor <= end:
            dates.append(cursor.isoformat())
            cursor += timedelta(days=1)
    else:
        dates = sorted(file.stem for file in zip_dir.glob("*.zip"))

    selected = []
    for archive_date in dates:
        zip_path = zip_dir / f"{archive_date}.zip"
        if zip_path.exists():
            selected.append(zip_path)
        else:
            print(f"  {archive_date}: not found, skipping")
    if not selected:
        print("No replay ZIPs selected.")
        return
    with ResultsDB() as db:
        total = populate_replays(db, zip_paths=selected)
        if total > 0:
            print("\nTop 10 cards by Elo:")
            for card in db.get_top_cards(10):
                print(f"  {card['name']}: {card['elo']:.0f}")


if __name__ == "__main__":
    main()
