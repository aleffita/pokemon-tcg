"""Declarative SQLite core for observed teams, submissions, decks, and matches.

The database is disposable.  This module deliberately contains no migration
code: an existing database with another schema is rejected and must be rebuilt
explicitly by its operator.

Schema 2.0.0 replaces the single-snapshot ``card_elo``/``deck_elo`` tables
with day-partitioned ``card_elo_daily``/``deck_elo_daily`` tables (plus a new
``agent_elo_daily`` and ``meta_features_daily``), and introduces ``days``,
``agents``, and ``datasets`` as first-class rows. ``card_elo``/``deck_elo``
survive as read-only compatibility views over the latest day snapshot per
``(entity, source)`` so existing read-only callers keep working unmodified.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
from pathlib import Path
import re
import sqlite3
from typing import Iterable, Iterator, Mapping, Sequence


DB_PATH = Path(__file__).resolve().parent.parent / "model" / "results.db"
SCHEMA_VERSION = "2.0.0"
K = 32
INITIAL_ELO = 600.0


class DatabaseRebuildRequired(RuntimeError):
    """Raised when a non-empty database does not implement this exact schema."""


class IdempotencyConflict(RuntimeError):
    """Raised when one immutable source identity is reused for other evidence."""


def canonical_deck_composition(
    card_ids_or_quantities: Iterable[int] | Mapping[int, int],
) -> tuple[tuple[int, int], ...]:
    """Return a validated, order-independent card composition."""

    if isinstance(card_ids_or_quantities, Mapping):
        counts = Counter(
            {
                int(card_id): int(quantity)
                for card_id, quantity in card_ids_or_quantities.items()
            }
        )
    else:
        counts = Counter(int(card_id) for card_id in card_ids_or_quantities)
    if not counts:
        raise ValueError("a deck composition cannot be empty")
    if any(card_id <= 0 for card_id in counts):
        raise ValueError("card ids must be positive")
    if any(quantity <= 0 for quantity in counts.values()):
        raise ValueError("card quantities must be positive")
    return tuple(sorted(counts.items()))


def deck_fingerprint(
    card_ids_or_quantities: Iterable[int] | Mapping[int, int],
) -> str:
    """Return the full deterministic SHA-256 identity of an exact composition."""

    composition = canonical_deck_composition(card_ids_or_quantities)
    canonical = "deck-v1|" + ";".join(
        f"{card_id}:{quantity}" for card_id, quantity in composition
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _digest(parts: Sequence[str]) -> str:
    framed = "\x1f".join(f"{len(part)}:{part}" for part in parts)
    return hashlib.sha256(framed.encode("utf-8")).hexdigest()


def _extract_lb_score(label: str) -> int | None:
    match = re.search(r"lb(\d+)", label)
    return int(match.group(1)) if match else None


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


SCHEMA_SQL = f"""
CREATE TABLE schema_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
INSERT INTO schema_metadata (singleton, schema_version)
VALUES (1, '{SCHEMA_VERSION}');

CREATE TABLE data_sources (
    code TEXT PRIMARY KEY,
    description TEXT NOT NULL
);
INSERT INTO data_sources (code, description) VALUES
    ('local', 'Local synchronous arena observation'),
    ('remote', 'Remote Kaggle replay observation');

CREATE TABLE deck_sources (
    code TEXT PRIMARY KEY,
    description TEXT NOT NULL
);
INSERT INTO deck_sources (code, description) VALUES
    ('replay', 'Exact composition observed in a remote replay'),
    ('arena', 'Exact composition observed in the local arena'),
    ('starter', 'Bundled starter deck'),
    ('submission', 'Deck extracted from a submission artifact'),
    ('public_agent', 'Deck distributed with a public agent'),
    ('builder', 'Deck authored in the local dashboard'),
    ('custom', 'User-authored deck');

CREATE TABLE teams (
    id INTEGER PRIMARY KEY,
    identity_key TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    identity_kind TEXT NOT NULL
        CHECK (identity_kind IN ('reported_name', 'unidentified_observation')),
    first_observed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE cards (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    stage TEXT,
    hp INTEGER,
    energy_type TEXT,
    weakness TEXT,
    rule TEXT,
    metadata_complete INTEGER NOT NULL DEFAULT 0 CHECK (metadata_complete IN (0, 1))
);

CREATE TABLE decks (
    id INTEGER PRIMARY KEY,
    fingerprint TEXT UNIQUE CHECK (
        fingerprint IS NULL OR length(fingerprint) = 64
    ),
    name TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL REFERENCES deck_sources(code),
    archetype TEXT,
    card_count INTEGER NOT NULL CHECK (card_count > 0),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE deck_cards (
    deck_id INTEGER NOT NULL REFERENCES decks(id) ON DELETE CASCADE,
    card_id INTEGER NOT NULL REFERENCES cards(id),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    PRIMARY KEY (deck_id, card_id)
);

CREATE TABLE submissions (
    id INTEGER PRIMARY KEY,
    team_id INTEGER NOT NULL REFERENCES teams(id),
    source TEXT NOT NULL REFERENCES data_sources(code),
    external_submission_id TEXT,
    identity_kind TEXT NOT NULL
        CHECK (identity_kind IN ('official', 'observed_team_deck_date')),
    observation_fingerprint TEXT NOT NULL UNIQUE CHECK (
        length(observation_fingerprint) = 64
    ),
    first_observed_archive_date TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (id, team_id)
);
CREATE UNIQUE INDEX submissions_external_identity
ON submissions(source, external_submission_id)
WHERE external_submission_id IS NOT NULL;

CREATE TABLE submission_decks (
    submission_id INTEGER NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
    deck_id INTEGER NOT NULL REFERENCES decks(id),
    role TEXT NOT NULL DEFAULT 'primary',
    ordinal INTEGER NOT NULL DEFAULT 0 CHECK (ordinal >= 0),
    PRIMARY KEY (submission_id, deck_id, role),
    UNIQUE (submission_id, role, ordinal),
    UNIQUE (submission_id, deck_id)
);

CREATE TABLE days (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL UNIQUE,
    competition_day INTEGER,
    is_complete INTEGER NOT NULL DEFAULT 0,
    source_zip TEXT,
    source_sha256 TEXT,
    n_matches INTEGER NOT NULL DEFAULT 0,
    imported_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX days_date ON days(date);

CREATE TABLE agents (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    kaggle_username TEXT,
    team_id INTEGER REFERENCES teams(id),
    submission_ref TEXT,
    is_self INTEGER NOT NULL DEFAULT 0,
    first_seen_day INTEGER REFERENCES days(id),
    last_seen_day INTEGER REFERENCES days(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (name, submission_ref)
);
CREATE INDEX agents_team ON agents(team_id);
CREATE INDEX agents_is_self ON agents(is_self);

CREATE TABLE matches (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL REFERENCES data_sources(code),
    day_id INTEGER REFERENCES days(id),
    our_agent_id INTEGER REFERENCES agents(id),
    opp_agent_id INTEGER REFERENCES agents(id),
    matchup_id INTEGER REFERENCES matchups(id),
    game_index INTEGER NOT NULL DEFAULT 0,
    our_agent TEXT,
    our_deck_id INTEGER REFERENCES decks(id),
    opp_agent TEXT,
    opp_deck_id INTEGER REFERENCES decks(id),
    our_side INTEGER CHECK (our_side IS NULL OR our_side IN (0, 1)),
    result INTEGER CHECK (result IS NULL OR result IN (-1, 0, 1)),
    external_episode_id TEXT,
    source_observation_digest TEXT CHECK (
        source_observation_digest IS NULL
        OR length(source_observation_digest) = 64
    ),
    archive_date TEXT,
    archive_member TEXT,
    n_steps INTEGER NOT NULL DEFAULT 0 CHECK (n_steps >= 0),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX matches_local_game
ON matches(matchup_id, game_index)
WHERE matchup_id IS NOT NULL;
CREATE UNIQUE INDEX matches_external_episode
ON matches(source, external_episode_id)
WHERE external_episode_id IS NOT NULL;
CREATE UNIQUE INDEX matches_source_observation
ON matches(source, source_observation_digest);
CREATE INDEX matches_day ON matches(day_id);
CREATE INDEX matches_our_agent_id ON matches(our_agent_id);
CREATE INDEX matches_opp_agent_id ON matches(opp_agent_id);
CREATE INDEX matches_archive_date ON matches(archive_date);

CREATE TABLE match_participants (
    id INTEGER PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    seat INTEGER NOT NULL CHECK (seat IN (0, 1)),
    team_id INTEGER NOT NULL,
    submission_id INTEGER NOT NULL,
    deck_id INTEGER NOT NULL,
    reward REAL NOT NULL,
    outcome INTEGER NOT NULL CHECK (outcome IN (-1, 0, 1)),
    UNIQUE (match_id, seat),
    UNIQUE (id, match_id),
    FOREIGN KEY (submission_id, team_id)
        REFERENCES submissions(id, team_id),
    FOREIGN KEY (submission_id, deck_id)
        REFERENCES submission_decks(submission_id, deck_id)
);

CREATE TABLE match_card_usage (
    participant_id INTEGER REFERENCES match_participants(id) ON DELETE CASCADE,
    match_id INTEGER REFERENCES matches(id) ON DELETE CASCADE,
    player_side INTEGER CHECK (player_side IS NULL OR player_side IN (0, 1)),
    card_id INTEGER NOT NULL REFERENCES cards(id),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    CHECK (
        (participant_id IS NOT NULL AND match_id IS NULL AND player_side IS NULL)
        OR
        (participant_id IS NULL AND match_id IS NOT NULL AND player_side IS NOT NULL)
    )
);
CREATE UNIQUE INDEX match_card_usage_participant
ON match_card_usage(participant_id, card_id)
WHERE participant_id IS NOT NULL;
CREATE UNIQUE INDEX match_card_usage_legacy_seat
ON match_card_usage(match_id, player_side, card_id)
WHERE match_id IS NOT NULL;

CREATE TABLE card_elo_daily (
    card_id INTEGER NOT NULL REFERENCES cards(id),
    day_id INTEGER NOT NULL REFERENCES days(id) ON DELETE CASCADE,
    source TEXT NOT NULL REFERENCES data_sources(code),
    elo REAL NOT NULL,
    games_played INTEGER NOT NULL DEFAULT 0,
    exposure INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    draws INTEGER NOT NULL DEFAULT 0,
    computed_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (card_id, day_id, source)
);
CREATE INDEX card_elo_daily_lookup ON card_elo_daily(day_id, source, card_id);

CREATE TABLE deck_elo_daily (
    deck_id INTEGER NOT NULL REFERENCES decks(id),
    day_id INTEGER NOT NULL REFERENCES days(id) ON DELETE CASCADE,
    source TEXT NOT NULL REFERENCES data_sources(code),
    elo REAL NOT NULL,
    games_played INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    draws INTEGER NOT NULL DEFAULT 0,
    computed_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (deck_id, day_id, source)
);
CREATE INDEX deck_elo_daily_lookup ON deck_elo_daily(day_id, source, deck_id);

CREATE TABLE agent_elo_daily (
    agent_id INTEGER NOT NULL REFERENCES agents(id),
    day_id INTEGER NOT NULL REFERENCES days(id) ON DELETE CASCADE,
    source TEXT NOT NULL REFERENCES data_sources(code),
    elo REAL NOT NULL,
    games_played INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    draws INTEGER NOT NULL DEFAULT 0,
    computed_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (agent_id, day_id, source)
);
CREATE INDEX agent_elo_daily_lookup ON agent_elo_daily(day_id, source, agent_id);

CREATE TABLE meta_features_daily (
    card_id INTEGER NOT NULL REFERENCES cards(id),
    day_id INTEGER NOT NULL REFERENCES days(id) ON DELETE CASCADE,
    source TEXT NOT NULL REFERENCES data_sources(code),
    elo_bucket_10p INTEGER NOT NULL,
    trend_7d REAL NOT NULL DEFAULT 0.0,
    delta_from_yesterday REAL NOT NULL DEFAULT 0.0,
    rank_in_meta INTEGER,
    exposure_pct REAL NOT NULL DEFAULT 0.0,
    computed_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (card_id, day_id, source)
);
CREATE INDEX meta_features_daily_lookup ON meta_features_daily(day_id, card_id);

CREATE TABLE datasets (
    id INTEGER PRIMARY KEY,
    day_id INTEGER NOT NULL REFERENCES days(id) ON DELETE CASCADE,
    path TEXT NOT NULL UNIQUE,
    schema_version INTEGER NOT NULL,
    rows INTEGER NOT NULL DEFAULT 0,
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
    aux_targets INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX datasets_day ON datasets(day_id);

CREATE TABLE operation_receipts (
    idempotency_key TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    payload_digest TEXT NOT NULL CHECK (length(payload_digest) = 64),
    result_table TEXT NOT NULL,
    result_id INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE tournaments (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    agent TEXT NOT NULL,
    games_per_opp INTEGER NOT NULL DEFAULT 20,
    note TEXT NOT NULL DEFAULT '',
    total_w INTEGER NOT NULL DEFAULT 0,
    total_l INTEGER NOT NULL DEFAULT 0,
    total_d INTEGER NOT NULL DEFAULT 0,
    win_rate REAL NOT NULL DEFAULT 0.0,
    total_time_s REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE matchups (
    id INTEGER PRIMARY KEY,
    tournament_id INTEGER NOT NULL
        REFERENCES tournaments(id) ON DELETE CASCADE,
    opponent TEXT NOT NULL,
    w INTEGER NOT NULL DEFAULT 0,
    l INTEGER NOT NULL DEFAULT 0,
    d INTEGER NOT NULL DEFAULT 0,
    win_rate REAL NOT NULL DEFAULT 0.0,
    lb_score INTEGER
);

CREATE TABLE match_steps (
    id INTEGER PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    step_num INTEGER NOT NULL,
    player_idx INTEGER NOT NULL CHECK (player_idx IN (0, 1)),
    turn INTEGER NOT NULL DEFAULT 0,
    select_type INTEGER,
    select_context INTEGER,
    n_options INTEGER NOT NULL DEFAULT 0,
    action TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL,
    reward INTEGER NOT NULL DEFAULT 0,
    UNIQUE (match_id, step_num, player_idx)
);

CREATE TABLE step_options (
    id INTEGER PRIMARY KEY,
    step_id INTEGER NOT NULL REFERENCES match_steps(id) ON DELETE CASCADE,
    option_idx INTEGER NOT NULL,
    option_type INTEGER NOT NULL,
    was_selected INTEGER NOT NULL DEFAULT 0 CHECK (was_selected IN (0, 1)),
    UNIQUE (step_id, option_idx)
);

CREATE TABLE step_events (
    id INTEGER PRIMARY KEY,
    step_id INTEGER NOT NULL REFERENCES match_steps(id) ON DELETE CASCADE,
    event_type INTEGER NOT NULL,
    player_idx INTEGER,
    card_id INTEGER,
    serial INTEGER,
    target_card_id INTEGER,
    target_serial INTEGER,
    value INTEGER
);

CREATE TABLE board_snapshots (
    id INTEGER PRIMARY KEY,
    step_id INTEGER NOT NULL REFERENCES match_steps(id) ON DELETE CASCADE,
    player_idx INTEGER NOT NULL CHECK (player_idx IN (0, 1)),
    turn INTEGER NOT NULL,
    deck_count INTEGER NOT NULL DEFAULT 0,
    hand_count INTEGER NOT NULL DEFAULT 0,
    prize_count INTEGER NOT NULL DEFAULT 0,
    discard_count INTEGER NOT NULL DEFAULT 0,
    poisoned INTEGER NOT NULL DEFAULT 0,
    burned INTEGER NOT NULL DEFAULT 0,
    asleep INTEGER NOT NULL DEFAULT 0,
    paralyzed INTEGER NOT NULL DEFAULT 0,
    confused INTEGER NOT NULL DEFAULT 0,
    UNIQUE (step_id, player_idx)
);

CREATE TABLE pokemon_on_field (
    id INTEGER PRIMARY KEY,
    snapshot_id INTEGER NOT NULL
        REFERENCES board_snapshots(id) ON DELETE CASCADE,
    slot TEXT NOT NULL,
    slot_idx INTEGER NOT NULL,
    card_id INTEGER NOT NULL,
    serial INTEGER NOT NULL,
    hp INTEGER NOT NULL,
    max_hp INTEGER NOT NULL,
    n_energies INTEGER NOT NULL DEFAULT 0,
    n_tools INTEGER NOT NULL DEFAULT 0,
    n_preevo INTEGER NOT NULL DEFAULT 0,
    UNIQUE (snapshot_id, slot, slot_idx)
);

CREATE INDEX deck_cards_card ON deck_cards(card_id);
CREATE INDEX submissions_team ON submissions(team_id);
CREATE INDEX submission_decks_deck ON submission_decks(deck_id);
CREATE INDEX match_participants_team ON match_participants(team_id);
CREATE INDEX match_participants_submission ON match_participants(submission_id);
CREATE INDEX match_participants_deck ON match_participants(deck_id);
CREATE INDEX match_card_usage_card ON match_card_usage(card_id);
CREATE INDEX matchups_tournament ON matchups(tournament_id);
CREATE INDEX match_steps_match ON match_steps(match_id);
CREATE INDEX step_options_step ON step_options(step_id);
CREATE INDEX step_events_step ON step_events(step_id);
CREATE INDEX board_snapshots_step ON board_snapshots(step_id);
CREATE INDEX pokemon_on_field_snapshot ON pokemon_on_field(snapshot_id);

CREATE VIEW card_elo AS
SELECT
    ced.card_id,
    ced.source,
    ced.elo,
    ced.games_played,
    ced.wins,
    ced.losses,
    ced.draws,
    CAST(ced.wins AS REAL) / NULLIF(ced.games_played, 0) AS win_rate,
    ced.computed_at AS updated_at
FROM card_elo_daily ced
INNER JOIN (
    SELECT card_id, source, MAX(day_id) AS max_day_id
    FROM card_elo_daily
    GROUP BY card_id, source
) latest ON latest.card_id = ced.card_id
        AND latest.source = ced.source
        AND latest.max_day_id = ced.day_id;

CREATE VIEW deck_elo AS
SELECT
    ded.deck_id,
    ded.source,
    ded.elo,
    ded.games_played,
    ded.wins,
    ded.losses,
    ded.draws,
    CAST(ded.wins AS REAL) / NULLIF(ded.games_played, 0) AS win_rate,
    ded.computed_at AS updated_at
FROM deck_elo_daily ded
INNER JOIN (
    SELECT deck_id, source, MAX(day_id) AS max_day_id
    FROM deck_elo_daily
    GROUP BY deck_id, source
) latest ON latest.deck_id = ded.deck_id
        AND latest.source = ded.source
        AND latest.max_day_id = ded.day_id;
"""


@dataclass(frozen=True)
class MatchParticipant:
    seat: int
    team_id: int
    submission_id: int
    deck_id: int
    reward: float
    outcome: int


class ResultsDB:
    """One connection to the disposable relational results database."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA busy_timeout = 5000")
        try:
            self._initialize_exact_schema()
        except Exception:
            self.conn.close()
            raise

    def _initialize_exact_schema(self) -> None:
        tables = {
            row[0]
            for row in self.conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if not tables:
            self.conn.executescript(SCHEMA_SQL)
            self.conn.commit()
            return
        if "schema_metadata" not in tables:
            raise DatabaseRebuildRequired(
                f"{self.db_path} uses a legacy schema; rebuild it explicitly"
            )
        row = self.conn.execute(
            "SELECT schema_version FROM schema_metadata WHERE singleton = 1"
        ).fetchone()
        if row is None or str(row["schema_version"]) != SCHEMA_VERSION:
            version = None if row is None else row["schema_version"]
            raise DatabaseRebuildRequired(
                f"{self.db_path} has schema {version}; expected {SCHEMA_VERSION}"
            )

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Run one atomic write transaction."""

        with self.conn:
            yield

    def add_card(
        self,
        card_id: int,
        name: str,
        category: str | None = None,
        stage: str | None = None,
        hp: int | None = None,
        energy_type: str | None = None,
        weakness: str | None = None,
        rule: str | None = None,
    ) -> int:
        with self.transaction():
            self.conn.execute(
                """
                INSERT INTO cards (
                    id, name, category, stage, hp, energy_type, weakness, rule,
                    metadata_complete
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    category = excluded.category,
                    stage = excluded.stage,
                    hp = excluded.hp,
                    energy_type = excluded.energy_type,
                    weakness = excluded.weakness,
                    rule = excluded.rule,
                    metadata_complete = 1
                """,
                (
                    int(card_id),
                    name,
                    category,
                    stage,
                    hp,
                    energy_type,
                    weakness,
                    rule,
                ),
            )
        return int(card_id)

    def add_deck(
        self,
        name: str,
        source: str,
        archetype: str | None = None,
        card_count: int = 60,
    ) -> int:
        """Create a composition-pending catalog deck by its exact name.

        This compatibility API is used by the deck catalog and local arena.
        Identity becomes composition-based when ``add_deck_cards`` is called.
        The row is always re-read by its natural key; ``lastrowid`` is never
        trusted after a conflict-ignoring insert.
        """

        with self.transaction():
            self.conn.execute(
                """
                INSERT INTO decks (
                    fingerprint, name, source, archetype, card_count
                ) VALUES (NULL, ?, ?, ?, ?)
                ON CONFLICT(name) DO NOTHING
                """,
                (name, source, archetype, int(card_count)),
            )
            row = self.conn.execute(
                "SELECT id FROM decks WHERE name = ?", (name,)
            ).fetchone()
        if row is None:
            raise RuntimeError("deck name resolution failed")
        return int(row["id"])

    def add_deck_cards(
        self,
        deck_id: int,
        card_quantities: Sequence[tuple[int, int]],
    ) -> None:
        """Seal a catalog deck with one immutable exact composition."""

        composition = canonical_deck_composition(dict(card_quantities))
        fingerprint = deck_fingerprint(dict(composition))
        with self.transaction():
            deck = self.conn.execute(
                "SELECT fingerprint FROM decks WHERE id = ?", (deck_id,)
            ).fetchone()
            if deck is None:
                raise ValueError("unknown deck id")
            collision = self.conn.execute(
                "SELECT id FROM decks WHERE fingerprint = ? AND id != ?",
                (fingerprint, deck_id),
            ).fetchone()
            if collision is not None:
                raise IdempotencyConflict(
                    "another deck already owns this exact composition"
                )
            existing = tuple(
                (int(card_id), int(quantity))
                for card_id, quantity in self.conn.execute(
                    "SELECT card_id, quantity FROM deck_cards "
                    "WHERE deck_id = ? ORDER BY card_id",
                    (deck_id,),
                )
            )
            if existing and existing != composition:
                raise IdempotencyConflict(
                    "deck revisions are immutable; create another deck"
                )
            if deck["fingerprint"] not in (None, fingerprint):
                raise IdempotencyConflict(
                    "deck fingerprint disagrees with its composition"
                )
            for card_id, _ in composition:
                self.conn.execute(
                    """
                    INSERT INTO cards (id, name, metadata_complete)
                    VALUES (?, ?, 0)
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (card_id, f"Card {card_id}"),
                )
            for card_id, quantity in composition:
                self.conn.execute(
                    """
                    INSERT INTO deck_cards (deck_id, card_id, quantity)
                    VALUES (?, ?, ?)
                    ON CONFLICT(deck_id, card_id) DO NOTHING
                    """,
                    (deck_id, card_id, quantity),
                )
            self.conn.execute(
                """
                UPDATE decks
                SET fingerprint = ?, card_count = ?
                WHERE id = ?
                """,
                (
                    fingerprint,
                    sum(quantity for _, quantity in composition),
                    deck_id,
                ),
            )

    def add_tournament(
        self,
        timestamp: str,
        agent: str,
        games_per_opp: int,
        note: str,
        matchups: list[dict],
        overall: dict,
        total_time: float = 0.0,
    ) -> int:
        """Persist one local tournament and its aggregate matchups."""

        identity = _digest(
            (
                "tournament-v1",
                timestamp,
                agent,
                str(games_per_opp),
                note,
            )
        )
        # The digest is used only as an idempotency receipt; timestamps remain
        # the public tournament label used by the current dashboard.
        receipt_key = f"tournament:{identity}"
        with self.transaction():
            receipt = self.conn.execute(
                "SELECT result_id FROM operation_receipts "
                "WHERE idempotency_key = ?",
                (receipt_key,),
            ).fetchone()
            if receipt is not None:
                return int(receipt["result_id"])
            cursor = self.conn.execute(
                """
                INSERT INTO tournaments (
                    timestamp, agent, games_per_opp, note, total_w, total_l,
                    total_d, win_rate, total_time_s
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    agent,
                    int(games_per_opp),
                    note,
                    int(overall["w"]),
                    int(overall["l"]),
                    int(overall["d"]),
                    float(overall["wr"]),
                    float(total_time),
                ),
            )
            tournament_id = int(cursor.lastrowid)
            for matchup in matchups:
                self.conn.execute(
                    """
                    INSERT INTO matchups (
                        tournament_id, opponent, w, l, d, win_rate, lb_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tournament_id,
                        matchup["opponent"],
                        int(matchup["w"]),
                        int(matchup["l"]),
                        int(matchup["d"]),
                        float(matchup["wr"]),
                        _extract_lb_score(matchup["opponent"]),
                    ),
                )
            self.conn.execute(
                """
                INSERT INTO operation_receipts (
                    idempotency_key, operation, payload_digest,
                    result_table, result_id
                ) VALUES (?, 'add_tournament', ?, 'tournaments', ?)
                """,
                (receipt_key, identity, tournament_id),
            )
        return tournament_id

    def get_all_runs(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM tournaments ORDER BY timestamp, id"
        ).fetchall()
        runs = []
        for row in rows:
            matchups = self.conn.execute(
                "SELECT * FROM matchups "
                "WHERE tournament_id = ? ORDER BY id",
                (row["id"],),
            ).fetchall()
            runs.append(
                {
                    **dict(row),
                    "matchups": [
                        {
                            "opponent": matchup["opponent"],
                            "w": matchup["w"],
                            "l": matchup["l"],
                            "d": matchup["d"],
                            "wr": matchup["win_rate"],
                            "lb_score": matchup["lb_score"],
                        }
                        for matchup in matchups
                    ],
                    "overall": {
                        "w": row["total_w"],
                        "l": row["total_l"],
                        "d": row["total_d"],
                        "wr": row["win_rate"],
                    },
                }
            )
        return runs

    def compute_elos(self) -> dict[str, float]:
        runs = self.get_all_runs()
        elos: dict[str, float] = {}
        for run in runs:
            label = run["timestamp"]
            elos.setdefault(label, INITIAL_ELO)
            for matchup in run["matchups"]:
                opponent = matchup["opponent"]
                elos.setdefault(
                    opponent,
                    float(matchup.get("lb_score") or INITIAL_ELO),
                )
                games = matchup["w"] + matchup["l"] + matchup["d"]
                if not games:
                    continue
                score = (matchup["w"] + 0.5 * matchup["d"]) / games
                expected = 1 / (
                    1
                    + 10
                    ** ((elos[opponent] - elos[label]) / 400)
                )
                delta = K * (score - expected)
                elos[label] += delta
                elos[opponent] -= delta
        return elos

    def get_latest_run(self) -> dict | None:
        runs = self.get_all_runs()
        return runs[-1] if runs else None

    def get_best_run(self) -> dict | None:
        runs = self.get_all_runs()
        return max(runs, key=lambda run: run["win_rate"]) if runs else None

    def get_opponent_history(self, opponent: str) -> list[dict]:
        return [
            dict(row)
            for row in self.conn.execute(
                """
                SELECT m.*, t.timestamp, t.agent, t.note
                FROM matchups m
                JOIN tournaments t ON t.id = m.tournament_id
                WHERE m.opponent = ?
                ORDER BY t.timestamp, m.id
                """,
                (opponent,),
            )
        ]

    def register_day(
        self,
        date: str,
        source_zip: str | Path | None = None,
        competition_day: int | None = None,
    ) -> int:
        """Insert-or-return the ``day_id`` for one calendar date.

        ``source_zip`` sha256 is computed and stored only at first
        registration for this date; later calls never overwrite it.
        """

        source_zip_str = str(source_zip) if source_zip is not None else None
        source_sha256 = (
            _sha256_file(Path(source_zip)) if source_zip is not None else None
        )
        with self.transaction():
            self.conn.execute(
                """
                INSERT INTO days (date, competition_day, source_zip, source_sha256)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(date) DO NOTHING
                """,
                (date, competition_day, source_zip_str, source_sha256),
            )
            row = self.conn.execute(
                "SELECT id FROM days WHERE date = ?", (date,)
            ).fetchone()
        if row is None:
            raise RuntimeError("day natural-key resolution failed")
        return int(row["id"])

    def _bump_day_match_count(self, day_id: int | None) -> None:
        if day_id is None:
            return
        self.conn.execute(
            "UPDATE days SET n_matches = n_matches + 1 WHERE id = ?",
            (day_id,),
        )

    def upsert_agent(
        self,
        name: str,
        submission_ref: str | None = None,
        kaggle_username: str | None = None,
        team_id: int | None = None,
        is_self: bool = False,
    ) -> int:
        """Insert-or-return ``agent_id`` by ``(name, submission_ref)``.

        Looked up with an explicit ``IS`` comparison rather than relying on
        ``ON CONFLICT``: SQLite treats ``NULL`` as distinct in unique
        constraints, so a plain conflict-based upsert would silently create
        one new row per call whenever ``submission_ref`` is ``None``.
        """

        with self.transaction():
            row = self.conn.execute(
                "SELECT id FROM agents WHERE name = ? AND submission_ref IS ?",
                (name, submission_ref),
            ).fetchone()
            if row is None:
                cursor = self.conn.execute(
                    """
                    INSERT INTO agents (
                        name, kaggle_username, team_id, submission_ref, is_self
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (name, kaggle_username, team_id, submission_ref, int(is_self)),
                )
                agent_id = int(cursor.lastrowid)
            else:
                agent_id = int(row["id"])
        return agent_id

    def touch_agent_seen(self, agent_id: int, day_id: int) -> None:
        """Update ``first_seen_day`` (if null) and ``last_seen_day``."""

        with self.transaction():
            self.conn.execute(
                """
                UPDATE agents
                SET first_seen_day = COALESCE(first_seen_day, ?),
                    last_seen_day = ?
                WHERE id = ?
                """,
                (day_id, day_id, agent_id),
            )

    def register_dataset(
        self,
        day_id: int,
        path: str,
        schema_version: int,
        rows: int,
        sha256: str,
        aux_targets: bool = False,
    ) -> int:
        """Insert-or-return ``dataset_id`` by its unique ``path``."""

        with self.transaction():
            self.conn.execute(
                """
                INSERT INTO datasets (
                    day_id, path, schema_version, rows, sha256, aux_targets
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO NOTHING
                """,
                (
                    day_id,
                    str(path),
                    int(schema_version),
                    int(rows),
                    sha256,
                    int(aux_targets),
                ),
            )
            row = self.conn.execute(
                "SELECT id FROM datasets WHERE path = ?", (str(path),)
            ).fetchone()
        if row is None:
            raise RuntimeError("dataset natural-key resolution failed")
        return int(row["id"])

    def list_datasets_by_days(self, day_ids: list[int]) -> list[dict]:
        """Return dataset rows joined with ``days``, ordered by day date."""

        if not day_ids:
            return []
        placeholders = ",".join("?" for _ in day_ids)
        rows = self.conn.execute(
            f"""
            SELECT ds.*, d.date AS day_date, d.competition_day AS competition_day
            FROM datasets ds
            JOIN days d ON d.id = ds.day_id
            WHERE ds.day_id IN ({placeholders})
            ORDER BY d.date, ds.id
            """,
            tuple(day_ids),
        ).fetchall()
        return [dict(row) for row in rows]

    def add_match(
        self,
        matchup_id: int,
        game_index: int,
        source: str,
        our_agent: str,
        our_deck_id: int | None,
        opp_agent: str,
        opp_deck_id: int | None,
        our_side: int,
        result: int,
        n_steps: int = 0,
    ) -> int:
        """Idempotently create the current local-arena compatibility row."""

        observation_digest = _digest(
            (
                "local-match-v1",
                str(matchup_id),
                str(game_index),
                source,
                our_agent,
                str(our_deck_id),
                opp_agent,
                str(opp_deck_id),
                str(our_side),
                str(result),
            )
        )
        today = date.today().isoformat()
        with self.transaction():
            day_id = self.register_day(today)
            cursor = self.conn.execute(
                """
                INSERT INTO matches (
                    source, day_id, matchup_id, game_index, our_agent,
                    our_deck_id, opp_agent, opp_deck_id, our_side, result,
                    source_observation_digest, archive_date, archive_member,
                    n_steps
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                (
                    source,
                    day_id,
                    matchup_id,
                    game_index,
                    our_agent,
                    our_deck_id,
                    opp_agent,
                    opp_deck_id,
                    our_side,
                    result,
                    observation_digest,
                    today,
                    f"local:{matchup_id}:{game_index}",
                    n_steps,
                ),
            )
            if cursor.rowcount:
                self._bump_day_match_count(day_id)
            row = self.conn.execute(
                """
                SELECT id, source, our_agent, our_deck_id, opp_agent,
                       opp_deck_id, our_side, result, n_steps
                FROM matches
                WHERE matchup_id = ? AND game_index = ?
                """,
                (matchup_id, game_index),
            ).fetchone()
        if row is None:
            raise RuntimeError("local match natural-key resolution failed")
        actual = (
            row["source"],
            row["our_agent"],
            row["our_deck_id"],
            row["opp_agent"],
            row["opp_deck_id"],
            row["our_side"],
            row["result"],
            row["n_steps"],
        )
        expected = (
            source,
            our_agent,
            our_deck_id,
            opp_agent,
            opp_deck_id,
            our_side,
            result,
            n_steps,
        )
        if actual != expected:
            raise IdempotencyConflict(
                "local matchup/game identity has conflicting evidence"
            )
        return int(row["id"])

    def add_match_steps(self, match_id: int, steps_data: list[dict]) -> None:
        """Compatibility writer for current local replay normalization."""

        with self.transaction():
            for step in steps_data:
                self.conn.execute(
                    """
                    INSERT INTO match_steps (
                        match_id, step_num, player_idx, turn, select_type,
                        select_context, n_options, action, status, reward
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(match_id, step_num, player_idx) DO NOTHING
                    """,
                    (
                        match_id,
                        step["step_num"],
                        step["player_idx"],
                        step.get("turn", 0),
                        step.get("select_type"),
                        step.get("select_context"),
                        step.get("n_options", 0),
                        step.get("action", "[]"),
                        step["status"],
                        step.get("reward", 0),
                    ),
                )

    def get_or_create_team(
        self,
        display_name: str | None,
        *,
        observation_key: str | None = None,
    ) -> int:
        if display_name is not None and display_name.strip():
            canonical_name = display_name.strip()
            identity_kind = "reported_name"
            identity_key = _digest(("reported-team-v1", canonical_name))
        else:
            if not observation_key:
                raise ValueError(
                    "unidentified teams require a source observation key"
                )
            canonical_name = "Unidentified team"
            identity_kind = "unidentified_observation"
            identity_key = _digest(("unidentified-team-v1", observation_key))
        with self.transaction():
            self.conn.execute(
                """
                INSERT INTO teams (identity_key, display_name, identity_kind)
                VALUES (?, ?, ?)
                ON CONFLICT(identity_key) DO NOTHING
                """,
                (identity_key, canonical_name, identity_kind),
            )
            row = self.conn.execute(
                "SELECT id FROM teams WHERE identity_key = ?",
                (identity_key,),
            ).fetchone()
        if row is None:
            raise RuntimeError("team natural-key resolution failed")
        return int(row["id"])

    def get_or_create_deck(
        self,
        card_ids_or_quantities: Iterable[int] | Mapping[int, int],
        *,
        source: str,
        name: str | None = None,
        archetype: str | None = None,
    ) -> int:
        composition = canonical_deck_composition(card_ids_or_quantities)
        fingerprint = deck_fingerprint(dict(composition))
        resolved_name = name or f"replay_deck_{fingerprint[:16]}"
        with self.transaction():
            for card_id, _ in composition:
                self.conn.execute(
                    """
                    INSERT INTO cards (id, name, metadata_complete)
                    VALUES (?, ?, 0)
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (card_id, f"Card {card_id}"),
                )
            name_owner = self.conn.execute(
                "SELECT fingerprint FROM decks WHERE name = ?",
                (resolved_name,),
            ).fetchone()
            if (
                name_owner is not None
                and name_owner["fingerprint"] != fingerprint
            ):
                resolved_name = f"{resolved_name}_{fingerprint[:12]}"
            self.conn.execute(
                """
                INSERT INTO decks (
                    fingerprint, name, source, archetype, card_count
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(fingerprint) DO NOTHING
                """,
                (
                    fingerprint,
                    resolved_name,
                    source,
                    archetype,
                    sum(quantity for _, quantity in composition),
                ),
            )
            # Never trust lastrowid after ON CONFLICT DO NOTHING. SQLite may
            # retain the id of an unrelated prior insertion on this cursor.
            row = self.conn.execute(
                "SELECT id FROM decks WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
            if row is None:
                raise RuntimeError("deck fingerprint resolution failed")
            deck_id = int(row["id"])
            existing = tuple(
                (int(card_id), int(quantity))
                for card_id, quantity in self.conn.execute(
                    "SELECT card_id, quantity FROM deck_cards "
                    "WHERE deck_id = ? ORDER BY card_id",
                    (deck_id,),
                )
            )
            if existing and existing != composition:
                raise IdempotencyConflict(
                    "stored deck composition disagrees with its fingerprint"
                )
            for card_id, quantity in composition:
                self.conn.execute(
                    """
                    INSERT INTO deck_cards (deck_id, card_id, quantity)
                    VALUES (?, ?, ?)
                    ON CONFLICT(deck_id, card_id) DO NOTHING
                    """,
                    (deck_id, card_id, quantity),
                )
        return deck_id

    def observe_submission(
        self,
        team_id: int,
        deck_ids: Sequence[int],
        *,
        source: str,
        archive_date: str,
        external_submission_id: str | None = None,
    ) -> int:
        if not deck_ids:
            raise ValueError("a submission observation requires at least one deck")
        unique_deck_ids = tuple(dict.fromkeys(int(value) for value in deck_ids))
        rows = self.conn.execute(
            "SELECT id, fingerprint FROM decks WHERE id IN "
            f"({','.join('?' for _ in unique_deck_ids)})",
            unique_deck_ids,
        ).fetchall()
        if len(rows) != len(unique_deck_ids):
            raise ValueError("submission references an unknown deck")
        fingerprints = sorted(str(row["fingerprint"]) for row in rows)
        identity_kind = (
            "official" if external_submission_id else "observed_team_deck_date"
        )
        if external_submission_id is not None:
            identity_parts = (
                "official-submission-v1",
                source,
                str(team_id),
                external_submission_id,
                *fingerprints,
            )
        else:
            identity_parts = (
                "observed-submission-v1",
                source,
                str(team_id),
                archive_date,
                *fingerprints,
            )
        observation_fingerprint = _digest(identity_parts)
        with self.transaction():
            if external_submission_id is not None:
                existing = self.conn.execute(
                    """
                    SELECT id, observation_fingerprint
                    FROM submissions
                    WHERE source = ? AND external_submission_id = ?
                    """,
                    (source, external_submission_id),
                ).fetchone()
                if (
                    existing is not None
                    and existing["observation_fingerprint"]
                    != observation_fingerprint
                ):
                    raise IdempotencyConflict(
                        "official submission id has conflicting observation"
                    )
            self.conn.execute(
                """
                INSERT INTO submissions (
                    team_id, source, external_submission_id, identity_kind,
                    observation_fingerprint, first_observed_archive_date
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(observation_fingerprint) DO NOTHING
                """,
                (
                    team_id,
                    source,
                    external_submission_id,
                    identity_kind,
                    observation_fingerprint,
                    archive_date,
                ),
            )
            row = self.conn.execute(
                "SELECT id, team_id FROM submissions "
                "WHERE observation_fingerprint = ?",
                (observation_fingerprint,),
            ).fetchone()
            if row is None or int(row["team_id"]) != team_id:
                raise RuntimeError("submission natural-key resolution failed")
            submission_id = int(row["id"])
            for ordinal, deck_id in enumerate(unique_deck_ids):
                self.conn.execute(
                    """
                    INSERT INTO submission_decks (
                        submission_id, deck_id, role, ordinal
                    ) VALUES (?, ?, 'primary', ?)
                    ON CONFLICT(submission_id, deck_id) DO NOTHING
                    """,
                    (submission_id, deck_id, ordinal),
                )
        return submission_id

    def record_match(
        self,
        *,
        source: str,
        external_episode_id: str | None,
        source_observation_digest: str,
        archive_date: str,
        archive_member: str,
        n_steps: int,
        participants: Sequence[MatchParticipant],
        our_agent_id: int | None = None,
        opp_agent_id: int | None = None,
    ) -> int:
        if len(source_observation_digest) != 64:
            raise ValueError("source_observation_digest must be full SHA-256")
        if sorted(participant.seat for participant in participants) != [0, 1]:
            raise ValueError("a match requires exactly seats 0 and 1")
        ordered_participants = sorted(participants, key=lambda value: value.seat)
        team_names = []
        for participant in ordered_participants:
            row = self.conn.execute(
                "SELECT display_name FROM teams WHERE id = ?",
                (participant.team_id,),
            ).fetchone()
            if row is None:
                raise ValueError("match participant references an unknown team")
            team_names.append(str(row["display_name"]))
        receipt_key = (
            f"match:{source}:episode:{external_episode_id}"
            if external_episode_id is not None
            else f"match:{source}:observation:{source_observation_digest}"
        )
        with self.transaction():
            receipt = self.conn.execute(
                """
                SELECT payload_digest, result_id
                FROM operation_receipts
                WHERE idempotency_key = ?
                """,
                (receipt_key,),
            ).fetchone()
            if receipt is not None:
                if receipt["payload_digest"] != source_observation_digest:
                    raise IdempotencyConflict(
                        "match idempotency key has conflicting source bytes"
                    )
                return int(receipt["result_id"])

            if external_episode_id is not None:
                existing = self.conn.execute(
                    """
                    SELECT id, source_observation_digest
                    FROM matches
                    WHERE source = ? AND external_episode_id = ?
                    """,
                    (source, external_episode_id),
                ).fetchone()
                if (
                    existing is not None
                    and existing["source_observation_digest"]
                    != source_observation_digest
                ):
                    raise IdempotencyConflict(
                        "external episode id has conflicting source bytes"
                    )

            day_id = self.register_day(archive_date)
            cursor = self.conn.execute(
                """
                INSERT INTO matches (
                    source, day_id, game_index, our_agent, our_deck_id,
                    opp_agent, opp_deck_id, our_side, result,
                    external_episode_id, source_observation_digest,
                    archive_date, archive_member, n_steps,
                    our_agent_id, opp_agent_id
                ) VALUES (?, ?, 0, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, source_observation_digest) DO NOTHING
                """,
                (
                    source,
                    day_id,
                    team_names[0],
                    ordered_participants[0].deck_id,
                    team_names[1],
                    ordered_participants[1].deck_id,
                    ordered_participants[0].outcome,
                    external_episode_id,
                    source_observation_digest,
                    archive_date,
                    archive_member,
                    int(n_steps),
                    our_agent_id,
                    opp_agent_id,
                ),
            )
            if cursor.rowcount:
                self._bump_day_match_count(day_id)
            row = self.conn.execute(
                "SELECT id FROM matches "
                "WHERE source = ? AND source_observation_digest = ?",
                (source, source_observation_digest),
            ).fetchone()
            if row is None:
                raise RuntimeError("match natural-key resolution failed")
            match_id = int(row["id"])

            for participant in participants:
                self.conn.execute(
                    """
                    INSERT INTO match_participants (
                        match_id, seat, team_id, submission_id, deck_id,
                        reward, outcome
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(match_id, seat) DO NOTHING
                    """,
                    (
                        match_id,
                        participant.seat,
                        participant.team_id,
                        participant.submission_id,
                        participant.deck_id,
                        participant.reward,
                        participant.outcome,
                    ),
                )
                participant_row = self.conn.execute(
                    """
                    SELECT id, team_id, submission_id, deck_id, reward, outcome
                    FROM match_participants
                    WHERE match_id = ? AND seat = ?
                    """,
                    (match_id, participant.seat),
                ).fetchone()
                expected = (
                    participant.team_id,
                    participant.submission_id,
                    participant.deck_id,
                    float(participant.reward),
                    participant.outcome,
                )
                actual = (
                    int(participant_row["team_id"]),
                    int(participant_row["submission_id"]),
                    int(participant_row["deck_id"]),
                    float(participant_row["reward"]),
                    int(participant_row["outcome"]),
                )
                if actual != expected:
                    raise IdempotencyConflict(
                        "match participant seat has conflicting evidence"
                    )
                participant_id = int(participant_row["id"])
                self.conn.execute(
                    """
                    INSERT INTO match_card_usage (
                        participant_id, card_id, quantity
                    )
                    SELECT ?, card_id, quantity
                    FROM deck_cards
                    WHERE deck_id = ?
                    ON CONFLICT DO NOTHING
                    """,
                    (participant_id, participant.deck_id),
                )

            self.conn.execute(
                """
                INSERT INTO operation_receipts (
                    idempotency_key, operation, payload_digest,
                    result_table, result_id
                ) VALUES (?, 'record_match', ?, 'matches', ?)
                """,
                (receipt_key, source_observation_digest, match_id),
            )
        return match_id

    def _resolve_latest_day_id(self, source: str) -> int:
        """Return the day_id to stamp a fresh Elo/meta snapshot with.

        Prefers the latest day actually observed among this source's
        matches; falls back to the latest registered day overall when every
        match of this source still has a null ``day_id`` (e.g. rows written
        outside this module's match-creation helpers). Aborts loudly when no
        day has ever been registered — there is no calendar context to stamp
        the snapshot with.
        """

        row = self.conn.execute(
            "SELECT MAX(day_id) AS max_day FROM matches WHERE source = ?",
            (source,),
        ).fetchone()
        day_id = row["max_day"] if row is not None else None
        if day_id is None:
            fallback = self.conn.execute(
                "SELECT MAX(id) AS max_day FROM days"
            ).fetchone()
            day_id = fallback["max_day"] if fallback is not None else None
        if day_id is None:
            raise RuntimeError(
                f"cannot compute a daily snapshot for source={source!r}: no "
                "matches carry a day_id and the days table is empty; call "
                "register_day() before computing Elo"
            )
        return int(day_id)

    def compute_deck_elo(self, source: str = "remote") -> dict[int, float]:
        """Rolling-forward deck Elo: snapshot ratings at each day boundary."""
        ratings: dict[int, float] = {}
        stats: dict[int, list[int]] = {}

        def _snapshot(day_id: int | None) -> None:
            if day_id is None:
                return
            self.conn.execute(
                "DELETE FROM deck_elo_daily WHERE source = ? AND day_id = ?",
                (source, day_id),
            )
            for deck_id, elo in sorted(ratings.items()):
                wins, losses, draws = stats[deck_id]
                games = wins + losses + draws
                self.conn.execute(
                    """
                    INSERT INTO deck_elo_daily (
                        deck_id, day_id, source, elo, games_played, wins,
                        losses, draws
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (deck_id, day_id, source, elo, games, wins, losses, draws),
                )

        matches = self.conn.execute(
            "SELECT id, day_id FROM matches WHERE source = ? ORDER BY day_id, id",
            (source,),
        ).fetchall()

        current_day_id: int | None = None
        with self.transaction():
            for match in matches:
                match_day_id = match["day_id"]
                if current_day_id is not None and match_day_id != current_day_id:
                    _snapshot(current_day_id)
                current_day_id = match_day_id

                players = self.conn.execute(
                    """
                    SELECT deck_id, outcome
                    FROM match_participants
                    WHERE match_id = ?
                    ORDER BY seat
                    """,
                    (match["id"],),
                ).fetchall()
                if len(players) != 2:
                    legacy = self.conn.execute(
                        """
                        SELECT our_deck_id, opp_deck_id, result
                        FROM matches
                        WHERE id = ?
                        """,
                        (match["id"],),
                    ).fetchone()
                    if (
                        legacy is not None
                        and legacy["our_deck_id"] is not None
                        and legacy["opp_deck_id"] is not None
                        and legacy["result"] is not None
                    ):
                        result = int(legacy["result"])
                        players = [
                            {"deck_id": int(legacy["our_deck_id"]), "outcome": result},
                            {"deck_id": int(legacy["opp_deck_id"]), "outcome": -result},
                        ]
                if len(players) != 2:
                    continue
                left, right = players
                for row in players:
                    deck_id = int(row["deck_id"])
                    ratings.setdefault(deck_id, INITIAL_ELO)
                    stats.setdefault(deck_id, [0, 0, 0])
                    stats[deck_id][0 if row["outcome"] > 0 else 1 if row["outcome"] < 0 else 2] += 1
                if int(left["deck_id"]) == int(right["deck_id"]):
                    continue
                score = 1.0 if left["outcome"] > 0 else 0.0 if left["outcome"] < 0 else 0.5
                left_id, right_id = int(left["deck_id"]), int(right["deck_id"])
                expected = 1 / (1 + 10 ** ((ratings[right_id] - ratings[left_id]) / 400))
                delta = K * (score - expected)
                ratings[left_id] += delta
                ratings[right_id] -= delta

            if current_day_id is not None:
                _snapshot(current_day_id)
        return ratings

    def compute_card_elo(self, source: str = "remote") -> dict[int, float]:
        """Rolling-forward card Elo: snapshot ratings at each day boundary."""
        ratings: dict[int, float] = {}
        stats: dict[int, list[int]] = {}

        def _snapshot(day_id: int | None) -> None:
            if day_id is None:
                return
            self.conn.execute(
                "DELETE FROM card_elo_daily WHERE source = ? AND day_id = ?",
                (source, day_id),
            )
            for card_id, elo in sorted(ratings.items()):
                wins, losses, draws = stats[card_id]
                games = wins + losses + draws
                # exposure mirrors games_played 1:1 under this algorithm.
                self.conn.execute(
                    """
                    INSERT INTO card_elo_daily (
                        card_id, day_id, source, elo, games_played, exposure,
                        wins, losses, draws
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (card_id, day_id, source, elo, games, games, wins, losses, draws),
                )

        matches = self.conn.execute(
            """
            SELECT id, day_id, result, our_side
            FROM matches
            WHERE source = ?
            ORDER BY day_id, id
            """,
            (source,),
        ).fetchall()

        current_day_id: int | None = None
        with self.transaction():
            for match in matches:
                match_day_id = match["day_id"]
                if current_day_id is not None and match_day_id != current_day_id:
                    _snapshot(current_day_id)
                current_day_id = match_day_id

                normalized_players = self.conn.execute(
                    """
                    SELECT id, seat, outcome
                    FROM match_participants
                    WHERE match_id = ?
                    ORDER BY seat
                    """,
                    (match["id"],),
                ).fetchall()
                card_sets: list[list[sqlite3.Row]] = []
                outcomes: list[int] = []
                if len(normalized_players) == 2:
                    for player in normalized_players:
                        card_sets.append(
                            self.conn.execute(
                                """
                                SELECT card_id, quantity
                                FROM match_card_usage
                                WHERE participant_id = ?
                                ORDER BY card_id
                                """,
                                (player["id"],),
                            ).fetchall()
                        )
                        outcomes.append(int(player["outcome"]))
                elif match["result"] is not None:
                    our_side = int(match["our_side"]) if match["our_side"] in (0, 1) else 0
                    result = int(match["result"])
                    for side in (0, 1):
                        card_sets.append(
                            self.conn.execute(
                                """
                                SELECT card_id, quantity
                                FROM match_card_usage
                                WHERE match_id = ? AND player_side = ?
                                ORDER BY card_id
                                """,
                                (match["id"], side),
                            ).fetchall()
                        )
                        outcomes.append(result if side == our_side else -result)
                if len(card_sets) != 2:
                    continue
                for cards, outcome in zip(card_sets, outcomes):
                    for card in cards:
                        card_id = int(card["card_id"])
                        ratings.setdefault(card_id, INITIAL_ELO)
                        stats.setdefault(card_id, [0, 0, 0])
                        stats[card_id][0 if outcome > 0 else 1 if outcome < 0 else 2] += 1
                if not card_sets[0] or not card_sets[1]:
                    continue
                left_mean = sum(ratings[int(r["card_id"])] for r in card_sets[0]) / len(card_sets[0])
                right_mean = sum(ratings[int(r["card_id"])] for r in card_sets[1]) / len(card_sets[1])
                score = 1.0 if outcomes[0] > 0 else 0.0 if outcomes[0] < 0 else 0.5
                expected = 1 / (1 + 10 ** ((right_mean - left_mean) / 400))
                delta = K * (score - expected)
                for sign, cards in ((1.0, card_sets[0]), (-1.0, card_sets[1])):
                    for card in cards:
                        ratings[int(card["card_id"])] += sign * delta * int(card["quantity"]) / 4

            if current_day_id is not None:
                _snapshot(current_day_id)
        return ratings

    def compute_agent_elo(self, source: str = "remote") -> dict[int, float]:
        """Compute per-day agent Elo from ``matches.our_agent_id``/``opp_agent_id``.

        Matches created through the current ``add_match``/``record_match``
        helpers do not populate agent linkage on their own; callers that want
        agent Elo must first identify agents with ``upsert_agent`` and stamp
        the corresponding ``matches`` rows themselves.
        """

        ratings: dict[int, float] = {}
        stats: dict[int, list[int]] = {}

        def _snapshot(day_id: int | None) -> None:
            if day_id is None:
                return
            self.conn.execute(
                "DELETE FROM agent_elo_daily WHERE source = ? AND day_id = ?",
                (source, day_id),
            )
            for agent_id, elo in sorted(ratings.items()):
                wins, losses, draws = stats[agent_id]
                games = wins + losses + draws
                self.conn.execute(
                    """
                    INSERT INTO agent_elo_daily (
                        agent_id, day_id, source, elo, games_played, wins,
                        losses, draws
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (agent_id, day_id, source, elo, games, wins, losses, draws),
                )

        matches = self.conn.execute(
            """
            SELECT id, day_id, our_agent_id, opp_agent_id, result
            FROM matches
            WHERE source = ?
              AND our_agent_id IS NOT NULL
              AND opp_agent_id IS NOT NULL
              AND result IS NOT NULL
            ORDER BY day_id, id
            """,
            (source,),
        ).fetchall()

        current_day_id: int | None = None
        with self.transaction():
            for match in matches:
                match_day_id = match["day_id"]
                if current_day_id is not None and match_day_id != current_day_id:
                    _snapshot(current_day_id)
                current_day_id = match_day_id

                left_id = int(match["our_agent_id"])
                right_id = int(match["opp_agent_id"])
                result = int(match["result"])
                for agent_id in (left_id, right_id):
                    ratings.setdefault(agent_id, INITIAL_ELO)
                    stats.setdefault(agent_id, [0, 0, 0])
                if left_id == right_id:
                    continue
                stats[left_id][0 if result > 0 else 1 if result < 0 else 2] += 1
                stats[right_id][0 if -result > 0 else 1 if -result < 0 else 2] += 1
                score = 1.0 if result > 0 else 0.0 if result < 0 else 0.5
                expected = 1 / (1 + 10 ** ((ratings[right_id] - ratings[left_id]) / 400))
                delta = K * (score - expected)
                ratings[left_id] += delta
                ratings[right_id] -= delta

            if current_day_id is not None:
                _snapshot(current_day_id)
        return ratings

    def refresh_competition_days(self) -> None:
        """Assign a monotonic 1-indexed ``competition_day`` to every registered
        day, ordered by calendar date. Idempotent -- resetting the whole column
        each call keeps the assignment consistent when new days are inserted
        out of chronological order (e.g. late remote backfill).
        """
        with self.transaction():
            rows = self.conn.execute(
                "SELECT id FROM days ORDER BY date"
            ).fetchall()
            self.conn.executemany(
                "UPDATE days SET competition_day = ? WHERE id = ?",
                [(rank, int(r["id"])) for rank, r in enumerate(rows, start=1)],
            )

    def compute_daily_elos(self, source: str) -> None:
        """Recompute every ``*_elo_daily`` table for ``source`` in one pass.

        Each sub-computation already deletes its own ``(source, day_id)``
        slice before inserting, so calling this repeatedly for the same day
        is idempotent.
        """

        self.compute_card_elo(source=source)
        self.compute_deck_elo(source=source)
        self.compute_agent_elo(source=source)

    def refresh_meta_features(self, source: str) -> None:
        """Rebuild ``meta_features_daily`` from ``card_elo_daily`` for ``source``.

        ``elo_bucket_10p`` is a 0-based decile (0 = strongest 10% of that
        day's field). ``trend_7d``/``delta_from_yesterday`` compare against
        the same card's elo on an earlier *calendar* day (not the previous
        stored snapshot), so they read 0.0 whenever that earlier day was
        never snapshotted for this source.
        """

        with self.transaction():
            self.conn.execute(
                "DELETE FROM meta_features_daily WHERE source = ?", (source,)
            )
            rows = self.conn.execute(
                """
                SELECT ced.card_id, ced.day_id, ced.elo, ced.exposure,
                       d.date AS day_date
                FROM card_elo_daily ced
                JOIN days d ON d.id = ced.day_id
                WHERE ced.source = ?
                ORDER BY d.date, ced.card_id
                """,
                (source,),
            ).fetchall()
            if not rows:
                return

            by_day: dict[int, list[sqlite3.Row]] = {}
            elo_by_card_date: dict[tuple[int, str], float] = {}
            for row in rows:
                by_day.setdefault(int(row["day_id"]), []).append(row)
                elo_by_card_date[(int(row["card_id"]), str(row["day_date"]))] = (
                    float(row["elo"])
                )

            for day_id, day_rows in by_day.items():
                ordered = sorted(day_rows, key=lambda r: -float(r["elo"]))
                n = len(ordered)
                total_exposure = sum(int(r["exposure"]) for r in day_rows)
                for rank, row in enumerate(ordered, start=1):
                    card_id = int(row["card_id"])
                    day_date_str = str(row["day_date"])
                    elo = float(row["elo"])
                    exposure = int(row["exposure"])
                    decile = min(9, (rank - 1) * 10 // n) if n else 0
                    current_date = date.fromisoformat(day_date_str)
                    prev_key = (
                        card_id,
                        (current_date - timedelta(days=1)).isoformat(),
                    )
                    week_ago_key = (
                        card_id,
                        (current_date - timedelta(days=7)).isoformat(),
                    )
                    prev_elo = elo_by_card_date.get(prev_key)
                    week_ago_elo = elo_by_card_date.get(week_ago_key)
                    delta_from_yesterday = (
                        elo - prev_elo if prev_elo is not None else 0.0
                    )
                    trend_7d = (
                        elo - week_ago_elo if week_ago_elo is not None else 0.0
                    )
                    exposure_pct = (
                        exposure / total_exposure if total_exposure else 0.0
                    )
                    self.conn.execute(
                        """
                        INSERT INTO meta_features_daily (
                            card_id, day_id, source, elo_bucket_10p, trend_7d,
                            delta_from_yesterday, rank_in_meta, exposure_pct
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            card_id,
                            day_id,
                            source,
                            decile,
                            trend_7d,
                            delta_from_yesterday,
                            rank,
                            exposure_pct,
                        ),
                    )

    def get_card_elo(
        self, card_id: int, source: str = "remote"
    ) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM card_elo WHERE card_id = ? AND source = ?",
            (int(card_id), source),
        ).fetchone()
        return dict(row) if row is not None else None

    def get_deck_elo(
        self, deck_id: int, source: str = "remote"
    ) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM deck_elo WHERE deck_id = ? AND source = ?",
            (int(deck_id), source),
        ).fetchone()
        return dict(row) if row is not None else None

    def get_top_cards(
        self, n: int = 50, source: str = "remote"
    ) -> list[dict]:
        return [
            dict(row)
            for row in self.conn.execute(
                """
                SELECT c.id, c.name, c.category, c.energy_type,
                       e.elo, e.games_played, e.win_rate
                FROM card_elo e
                JOIN cards c ON c.id = e.card_id
                WHERE e.source = ?
                ORDER BY e.elo DESC
                LIMIT ?
                """,
                (source, n),
            )
        ]

    def get_top_decks(
        self, n: int = 20, source: str = "remote"
    ) -> list[dict]:
        return [
            dict(row)
            for row in self.conn.execute(
                """
                SELECT d.id, d.name, d.source, d.archetype,
                       e.elo, e.games_played, e.win_rate
                FROM deck_elo e
                JOIN decks d ON d.id = e.deck_id
                WHERE e.source = ?
                ORDER BY e.elo DESC
                LIMIT ?
                """,
                (source, n),
            )
        ]

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ResultsDB":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
