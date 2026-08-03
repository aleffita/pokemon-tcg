"""Load and cache meta-game features from the SQLite catalog for encoder use.

Loads lazily on first use, cached per-process (module-level singleton via
``get_meta_lookup``).

Two failure classes, kept distinct on purpose:

* **Pipeline bug** -> raise ``MetaLookupError``. The catalog file is missing,
  unreadable, or a table the schema promises is absent. Something is wrong
  with the plumbing; silence would hide it.
* **Domain novelty** -> return ``UNKNOWN_BUCKET``. A card/agent/deck the
  catalog has genuinely never observed in **any** day up to the query day
  (newly released cards, first-time opponents). The model learns
  ``UNKNOWN_BUCKET`` as its own embedding row.

Temporal semantics for agent/deck/card buckets: an entity "observed on any
past day" is NEVER unknown. The catalog snapshots elos daily, and today's
elo snapshot for a given day only lands after all matches of that day are
computed -- but the last-known bucket for that entity from a prior day is a
better signal than "unknown" if we ask about it mid-day or on a day whose
snapshot hasn't been run yet. So we forward-fill: for any query
``(entity_id, day_id)``, we return the entity's bucket from the greatest
snapshotted day ``<= day_id``, and UNKNOWN only if no snapshot exists on any
day ``<= day_id`` for that entity.

Schema (see rl/results_db.py):
    days(id, date, competition_day, ...)
    meta_features_daily(card_id, day_id, source, elo_bucket_10p, ...)
    agent_elo_daily(agent_id, day_id, source, elo, ...)
    deck_elo_daily(deck_id, day_id, source, elo, ...)

Deciles (buckets 0..9) are precomputed for cards (``elo_bucket_10p``) but are
derived here for agents/decks by same-day elo rank. 0 == top decile.
"""
from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np

# Domain-legit "no data yet" code. Distinct from every real decile (0..9).
# The model owns an embedding row per real decile + one for UNKNOWN_BUCKET.
UNKNOWN_BUCKET = 10
MAX_COMPETITION_DAYS = 60
NEUTRAL_DAY_INDEX = 0.5   # only used before any day is registered -- see day_index_norm

_SOURCE = "remote"        # only source populated in the catalog today

# Resolve the default DB path against THIS FILE'S location (repo layout is
# stable), so callers who cd into a subdir or run from the Kaggle sandbox
# still resolve the same file. A caller MAY override via get_meta_lookup(...).
_DEFAULT_DB_PATH = str(
    Path(__file__).resolve().parents[2] / "model" / "results.db"
)


class MetaLookupError(RuntimeError):
    """Raised when the meta catalog itself is inconsistent (pipeline bug)."""


class MetaLookup:
    """Cached view over the meta catalog's tables.

    Distinguishes two failure classes:

    * **Pipeline bug** -> raise ``MetaLookupError``. The catalog file is
      missing, unreadable, or a table the schema promises is absent.
    * **Domain novelty** -> return ``UNKNOWN_BUCKET`` (cards/agents/decks not
      yet observed on the requested day). This is a first-class value the
      model consumes via a dedicated embedding row.
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        self.db_path = db_path
        # For each entity_id, keep a list of (day_id, bucket) tuples SORTED
        # ascending by day_id. Forward-fill lookups do a right-bisect on the
        # day_ids list to find the most recent day <= query.
        # Stored as (list[day_id], list[bucket]) rather than list[tuple] so
        # bisect can operate directly on the day list.
        self._card_history: dict[int, tuple[list[int], list[int]]] = {}
        self._agent_history: dict[int, tuple[list[int], list[int]]] = {}
        self._deck_history: dict[int, tuple[list[int], list[int]]] = {}
        # Per-day dense LUT cache for card lookups (perf). Keyed by
        # query day_id; entries are forward-filled at build time.
        self._card_bucket_luts: dict[int, np.ndarray] = {}
        self._day_by_date: dict[str, int] = {}
        self._competition_day_by_id: dict[int, int] = {}
        self._loaded = False

    # ---- loading -----------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not Path(self.db_path).exists():
            raise MetaLookupError(
                f"meta catalog not found at {self.db_path!r} -- "
                "run `uv run tcg-rebuild-db` to build it"
            )
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path)
        except sqlite3.Error as exc:
            raise MetaLookupError(
                f"cannot open meta catalog at {self.db_path!r}: {exc}"
            ) from exc
        conn.row_factory = sqlite3.Row
        try:
            required_tables = {
                "meta_features_daily", "agent_elo_daily", "deck_elo_daily", "days"
            }
            present = {
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type IN ('table','view') AND name IN "
                    "('meta_features_daily','agent_elo_daily','deck_elo_daily','days')"
                )
            }
            missing = required_tables - present
            if missing:
                raise MetaLookupError(
                    f"meta catalog at {self.db_path!r} is missing tables/views "
                    f"{sorted(missing)!r} -- schema is out of date, rebuild it"
                )

            # Cards: bucket is precomputed per (card_id, day_id).
            _card_by_day: dict[int, list[tuple[int, int]]] = {}
            for r in conn.execute(
                "SELECT day_id, card_id, elo_bucket_10p FROM meta_features_daily "
                "WHERE source = ? ORDER BY day_id",
                (_SOURCE,),
            ):
                _card_by_day.setdefault(int(r["card_id"]), []).append(
                    (int(r["day_id"]), int(r["elo_bucket_10p"]))
                )
            for cid, pairs in _card_by_day.items():
                pairs.sort()
                self._card_history[cid] = (
                    [p[0] for p in pairs],
                    [p[1] for p in pairs],
                )

            # Agents & decks: derive deciles per day from elo rank.
            def _load_ranked_history(
                query: str,
                key_col: str,
                dest: dict[int, tuple[list[int], list[int]]],
            ) -> None:
                by_day: dict[int, list[tuple[int, float]]] = {}
                for r in conn.execute(query, (_SOURCE,)):
                    by_day.setdefault(int(r["day_id"]), []).append(
                        (int(r[key_col]), float(r["elo"]))
                    )
                per_entity: dict[int, list[tuple[int, int]]] = {}
                for day_id, elos in by_day.items():
                    ranked = sorted(elos, key=lambda p: -p[1])
                    n = len(ranked)
                    for rank, (entity_id, _elo) in enumerate(ranked):
                        bucket = min(9, rank * 10 // max(n, 1))
                        per_entity.setdefault(entity_id, []).append(
                            (day_id, bucket)
                        )
                for entity_id, pairs in per_entity.items():
                    pairs.sort()
                    dest[entity_id] = (
                        [p[0] for p in pairs],
                        [p[1] for p in pairs],
                    )

            _load_ranked_history(
                "SELECT day_id, agent_id, elo FROM agent_elo_daily WHERE source = ?",
                "agent_id",
                self._agent_history,
            )
            _load_ranked_history(
                "SELECT day_id, deck_id, elo FROM deck_elo_daily WHERE source = ?",
                "deck_id",
                self._deck_history,
            )

            for r in conn.execute(
                "SELECT id, date, competition_day FROM days"
            ):
                self._day_by_date[str(r["date"])] = int(r["id"])
                if r["competition_day"] is not None:
                    self._competition_day_by_id[int(r["id"])] = int(
                        r["competition_day"]
                    )
        finally:
            conn.close()
        self._loaded = True

    # ---- forward-fill temporal lookup --------------------------------------
    @staticmethod
    def _forward_fill(
        history: tuple[list[int], list[int]] | None, query_day_id: int
    ) -> int:
        """Return the bucket from the most recent snapshotted day <= query_day_id
        for a given entity. UNKNOWN_BUCKET if entity has no snapshot in any
        day up to query_day_id (never observed yet)."""
        if history is None:
            return UNKNOWN_BUCKET
        days, buckets = history
        # bisect_right(days, query) - 1 == index of the greatest day <= query.
        idx = bisect.bisect_right(days, query_day_id) - 1
        if idx < 0:
            return UNKNOWN_BUCKET
        return buckets[idx]

    # ---- day resolution ---------------------------------------------------
    def resolve_day_id(self, date_str: str | None) -> int | None:
        """``date_str`` ("YYYY-MM-DD") -> its ``days.id``.

        ``date_str`` is None whenever the observation didn't carry a date
        (legitimate: some obs paths just don't have one) -> returns None.
        A NON-None date that isn't in the catalog is a sync bug -> raises.
        """
        if date_str is None:
            return None
        self._ensure_loaded()
        day_id = self._day_by_date.get(date_str)
        if day_id is None:
            raise MetaLookupError(
                f"date {date_str!r} is not registered in the meta catalog "
                f"({self.db_path!r}) -- run `tcg-data --date {date_str}` "
                "+ `tcg-rebuild-db` to sync"
            )
        return day_id

    def latest_day_id(self) -> int | None:
        """Most recently ingested day, or None if the catalog is empty (fresh install)."""
        self._ensure_loaded()
        if not self._day_by_date:
            return None
        return max(self._day_by_date.values())

    def day_index_norm(self, day_id: int | None) -> float:
        """Normalized competition day in [0, 1]. day_id=None -> neutral 0.5
        (legit -- no day context available yet). day_id present but not in
        the catalog is a bug -> raises."""
        if day_id is None:
            return NEUTRAL_DAY_INDEX
        self._ensure_loaded()
        comp_day = self._competition_day_by_id.get(day_id)
        if comp_day is None:
            raise MetaLookupError(
                f"day_id {day_id} is not registered in the meta catalog "
                f"({self.db_path!r})"
            )
        return max(0.0, min(1.0, (comp_day - 1) / max(1, MAX_COMPETITION_DAYS - 1)))

    # ---- card buckets (vectorized: dense per-query-day LUT with forward fill) ---
    def _card_bucket_lut(self, day_id: int) -> np.ndarray:
        """Dense array indexed by card_id, holding the forward-filled bucket at
        ``day_id`` for every card known up to that day. Cached per query
        day_id (called at most once per day used in a session)."""
        lut = self._card_bucket_luts.get(day_id)
        if lut is not None:
            return lut
        # Collect the set of card_ids that have ANY snapshot <= day_id.
        max_cid = -1
        picks: dict[int, int] = {}
        for cid, (days, buckets) in self._card_history.items():
            idx = bisect.bisect_right(days, day_id) - 1
            if idx < 0:
                continue  # card never seen up to day_id -> stays UNKNOWN in lut
            picks[cid] = buckets[idx]
            if cid > max_cid:
                max_cid = cid
        if max_cid < 0:
            lut = np.empty(0, dtype=np.int32)
        else:
            lut = np.full(max_cid + 1, UNKNOWN_BUCKET, dtype=np.int32)
            for cid, bucket in picks.items():
                lut[cid] = bucket
        self._card_bucket_luts[day_id] = lut
        return lut

    def card_bucket_array(
        self, card_ids: np.ndarray, day_id: int | None
    ) -> np.ndarray:
        """``card_ids`` (any int shape) -> same-shape int32 array of meta
        buckets. Uses forward-fill: for each card, returns the bucket from
        the most recent snapshot day <= ``day_id``. Cards that never appeared
        in any prior day (genuinely new to the meta) get ``UNKNOWN_BUCKET``.
        Card_id 0 (pad) and out-of-range ids also map to UNKNOWN_BUCKET so
        the model gets one consistent "no info" row.
        ``day_id=None`` (obs has no date) fills the whole array with
        UNKNOWN_BUCKET -- no temporal context to forward-fill from.
        """
        card_ids = np.asarray(card_ids)
        if day_id is None:
            return np.full(card_ids.shape, UNKNOWN_BUCKET, dtype=np.int32)
        self._ensure_loaded()
        lut = self._card_bucket_lut(day_id)
        out = np.full(card_ids.shape, UNKNOWN_BUCKET, dtype=np.int32)
        if lut.size:
            flat_ids = card_ids.reshape(-1).astype(np.int64)
            flat_out = out.reshape(-1)
            in_range = (flat_ids >= 0) & (flat_ids < lut.size)
            flat_out[in_range] = lut[flat_ids[in_range]]
        return out

    # ---- opponent buckets --------------------------------------------------
    def agent_bucket(self, agent_id: int | None, day_id: int | None) -> int:
        """Forward-fill bucket lookup for an opponent agent. Returns
        UNKNOWN_BUCKET only when agent has no snapshot in any day up to
        ``day_id`` (first time we ever see this agent). Missing agent_id or
        missing day_id (obs without opponent identity or date) is a legitimate
        "no context" case -> UNKNOWN_BUCKET."""
        if agent_id is None or day_id is None:
            return UNKNOWN_BUCKET
        self._ensure_loaded()
        return self._forward_fill(self._agent_history.get(agent_id), day_id)

    def deck_bucket(self, deck_id: int | None, day_id: int | None) -> int:
        """Forward-fill bucket lookup for an opponent deck. Same semantics as
        ``agent_bucket``: UNKNOWN only if this deck was never seen in any day
        up to ``day_id``."""
        if deck_id is None or day_id is None:
            return UNKNOWN_BUCKET
        self._ensure_loaded()
        return self._forward_fill(self._deck_history.get(deck_id), day_id)


_GLOBAL_LOOKUP: MetaLookup | None = None


def get_meta_lookup(db_path: str | None = None) -> MetaLookup:
    """Process-wide cached MetaLookup singleton.

    First call's ``db_path`` (or the module default) wins for the whole
    process. Callers that need to point at a different catalog must call
    ``reset_meta_lookup()`` first.
    """
    global _GLOBAL_LOOKUP
    if _GLOBAL_LOOKUP is None:
        _GLOBAL_LOOKUP = MetaLookup(db_path or _DEFAULT_DB_PATH)
    return _GLOBAL_LOOKUP


def reset_meta_lookup() -> None:
    """Discard the cached singleton (tests, or after rebuilding the catalog)."""
    global _GLOBAL_LOOKUP
    _GLOBAL_LOOKUP = None
