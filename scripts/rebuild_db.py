"""Rebuild model/results.db from canonical cards, decks, and replay ZIPs.

The existing database is never used as an input and no migration is involved.
All work happens in an isolated staging directory beside the target database.
Only a fully validated database is published, using one atomic replacement.

Usage:
    uv run tcg-rebuild-db
    uv run tcg-rebuild-db --dry-run
    uv run tcg-rebuild-db --target model/checkpoint/smoke/results.db
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any

from rl.results_db import DB_PATH, ResultsDB
from scripts.build_card_stats import populate_replays
from scripts.populate_cards import CSV_PATH, populate_cards
from scripts.populate_decks import ROOT, populate_decks


VOLATILE_LOGICAL_COLUMNS = frozenset(
    {"created_at", "updated_at", "first_observed_at"}
)
REQUIRED_NONEMPTY_TABLES = (
    "cards",
    "decks",
    "deck_cards",
    "teams",
    "submissions",
    "submission_decks",
    "matches",
    "match_participants",
    "match_card_usage",
    "card_elo",
    "deck_elo",
)


@dataclass(frozen=True)
class RebuildReport:
    target: str
    published: bool
    cards_populated: int
    decks_populated: int
    replays_populated: int
    counts: dict[str, int]
    logical_fingerprint: str
    integrity_check: tuple[str, ...]
    foreign_key_violations: int
    replay_sources: tuple[str, ...]


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _logical_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    return value


def database_counts(connection: sqlite3.Connection) -> dict[str, int]:
    """Return deterministic counts for every application table and view.

    Views are included because schema 2.0.0 exposes ``card_elo``/``deck_elo``
    as compatibility views over ``card_elo_daily``/``deck_elo_daily`` instead
    of standalone tables; ``REQUIRED_NONEMPTY_TABLES`` still checks them.
    """

    tables = [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
    ]
    return {
        table: int(
            connection.execute(
                f"SELECT COUNT(*) FROM {_quote_identifier(table)}"
            ).fetchone()[0]
        )
        for table in tables
    }


def logical_database_fingerprint(connection: sqlite3.Connection) -> str:
    """Hash schema and logical rows, excluding wall-clock timestamp columns."""

    digest = hashlib.sha256()
    schema_rows = connection.execute(
        "SELECT type, name, tbl_name, COALESCE(sql, '') "
        "FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' AND type IN ('table', 'index') "
        "ORDER BY type, name"
    ).fetchall()
    for row in schema_rows:
        digest.update(
            json.dumps(
                list(row),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")

    for table in database_counts(connection):
        table_info = connection.execute(
            f"PRAGMA table_info({_quote_identifier(table)})"
        ).fetchall()
        columns = [
            str(row[1])
            for row in table_info
            if str(row[1]) not in VOLATILE_LOGICAL_COLUMNS
        ]
        quoted = ", ".join(_quote_identifier(column) for column in columns)
        order = ", ".join(_quote_identifier(column) for column in columns)
        digest.update(
            json.dumps(
                {"table": table, "columns": columns},
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
        query = (
            f"SELECT {quoted} FROM {_quote_identifier(table)}"
            + (f" ORDER BY {order}" if order else "")
        )
        for row in connection.execute(query):
            digest.update(
                json.dumps(
                    [_logical_value(value) for value in row],
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
            digest.update(b"\n")
    return digest.hexdigest()


def validate_rebuilt_database(
    connection: sqlite3.Connection,
    *,
    expected_replays: int,
) -> tuple[dict[str, int], tuple[str, ...], str]:
    """Reject any structurally incomplete or referentially invalid staging DB."""

    integrity = tuple(
        str(row[0]) for row in connection.execute("PRAGMA integrity_check")
    )
    if integrity != ("ok",):
        raise RuntimeError(f"SQLite integrity_check failed: {integrity}")
    foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_key_rows:
        preview = [tuple(row) for row in foreign_key_rows[:10]]
        raise RuntimeError(
            "SQLite foreign_key_check found "
            f"{len(foreign_key_rows)} violation(s): {preview}"
        )

    counts = database_counts(connection)
    empty = [table for table in REQUIRED_NONEMPTY_TABLES if counts.get(table, 0) <= 0]
    if empty:
        raise RuntimeError(f"rebuilt database has empty required tables: {empty}")
    if counts["matches"] != expected_replays:
        raise RuntimeError(
            "rebuilt remote match count disagrees with replay population: "
            f"{counts['matches']} != {expected_replays}"
        )
    non_remote = int(
        connection.execute(
            "SELECT COUNT(*) FROM matches WHERE source != 'remote'"
        ).fetchone()[0]
    )
    if non_remote:
        raise RuntimeError(
            "rebuild imported non-canonical match rows instead of replay sources"
        )
    incomplete_participants = int(
        connection.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT match_id FROM match_participants "
            "  GROUP BY match_id HAVING COUNT(*) != 2"
            ")"
        ).fetchone()[0]
    )
    if incomplete_participants:
        raise RuntimeError(
            f"{incomplete_participants} replay matches lack two participants"
        )
    participants_without_usage = int(
        connection.execute(
            "SELECT COUNT(*) FROM match_participants mp "
            "WHERE NOT EXISTS ("
            "  SELECT 1 FROM match_card_usage mcu "
            "  WHERE mcu.participant_id = mp.id"
            ")"
        ).fetchone()[0]
    )
    if participants_without_usage:
        raise RuntimeError(
            f"{participants_without_usage} participants lack card usage"
        )
    invalid_decks = int(
        connection.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT d.id FROM decks d "
            "  LEFT JOIN deck_cards dc ON dc.deck_id = d.id "
            "  GROUP BY d.id, d.card_count "
            "  HAVING COALESCE(SUM(dc.quantity), 0) != d.card_count"
            ")"
        ).fetchone()[0]
    )
    if invalid_decks:
        raise RuntimeError(
            f"{invalid_decks} decks disagree with their declared card_count"
        )

    fingerprint = logical_database_fingerprint(connection)
    if fingerprint != logical_database_fingerprint(connection):
        raise RuntimeError("logical database fingerprint is not deterministic")
    return counts, integrity, fingerprint


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_target_replaceable(path: Path) -> None:
    if not path.exists():
        return
    connection = sqlite3.connect(str(path), timeout=0)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        connection.rollback()
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            f"target database is busy and cannot be atomically replaced: {path}"
        ) from exc
    finally:
        connection.close()


def rebuild_results_database(
    *,
    target: str | Path = DB_PATH,
    replay_zip_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    publish: bool = True,
) -> RebuildReport:
    """Build, validate, and optionally atomically publish a canonical database."""

    target_path = Path(target).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    replay_sources = tuple(
        str(Path(path).resolve())
        for path in (
            replay_zip_paths
            if replay_zip_paths is not None
            else sorted((ROOT / "data" / "bc_replay_zip").glob("*.zip"))
        )
    )
    if not replay_sources:
        raise FileNotFoundError("no canonical replay ZIPs were found")

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{target_path.stem}-rebuild-",
            dir=target_path.parent,
        )
    )
    staging_db = staging_dir / target_path.name
    database: ResultsDB | None = None
    try:
        from rich.console import Console
        console = Console()
        database = ResultsDB(staging_db)

        console.print("[cyan][rebuild][/] Syncing Kaggle Leaderboard (TTL 28h)...")
        database.sync_kaggle_leaderboard()

        console.print("[cyan][rebuild][/] Populating cards from CSV...")
        cards_populated = populate_cards(
            database,
            csv_path=CSV_PATH,
            skip_existing=False,
        )
        console.print(f"  [green]➔[/] Populated {cards_populated} cards.")

        console.print("[cyan][rebuild][/] Populating decks from canonical definitions...")
        decks_populated = populate_decks(
            database,
            root=ROOT,
            skip_existing=False,
            strict=True,
        )
        console.print(f"  [green]➔[/] Populated {decks_populated} decks.")

        console.print("[cyan][rebuild][/] Populating replays from Kaggle ZIPs (this may take a while)...")
        replays_populated = populate_replays(
            database,
            zip_paths=list(replay_sources),
        )
        console.print(f"  [green]➔[/] Populated {replays_populated} matches.")
        # Elo snapshots and meta features are part of the canonical rebuild.
        # A DB without them is not usable by the trainer / encoder even though
        # matches are populated -- do it here so the rebuild is self-contained.
        # source="remote" is the only source the meta catalog reads today
        # (see rl/encoder/meta_lookup.py); local tournament rebuilds keep
        # their own local elo computation via scripts/tournament.py.
        console.print("[cyan][rebuild][/] Assigning competition_day ordinals...")
        database.refresh_competition_days()
        console.print("[cyan][rebuild][/] Computing rolling-forward daily elos (source=remote)...")
        database.compute_daily_elos(source="remote")
        console.print("[cyan][rebuild][/] Refreshing meta features (source=remote)...")
        database.refresh_meta_features(source="remote")
        counts, integrity, fingerprint = validate_rebuilt_database(
            database.conn,
            expected_replays=replays_populated,
        )
        database.conn.commit()
        database.close()
        database = None
        _fsync_file(staging_db)

        if publish:
            _assert_target_replaceable(target_path)
            os.replace(staging_db, target_path)
            _fsync_directory(target_path.parent)

        return RebuildReport(
            target=str(target_path),
            published=publish,
            cards_populated=cards_populated,
            decks_populated=decks_populated,
            replays_populated=replays_populated,
            counts=counts,
            logical_fingerprint=fingerprint,
            integrity_check=integrity,
            foreign_key_violations=0,
            replay_sources=replay_sources,
        )
    finally:
        if database is not None:
            database.close()
        shutil.rmtree(staging_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default=str(DB_PATH),
        help="Database path to atomically replace after validation",
    )
    parser.add_argument(
        "--replay-zip",
        action="append",
        default=None,
        help="Canonical replay ZIP; repeat to override the default complete set",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate in project staging without replacing the target",
    )
    args = parser.parse_args()

    report = rebuild_results_database(
        target=args.target,
        replay_zip_paths=args.replay_zip,
        publish=not args.dry_run,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
