"""Load and cache meta-game features from the SQLite catalog (``model/results.db``)
for encoder use.

Passive by design: the encoder always calls into this module, but until a real
meta-loop pipeline populates ``obs``/tracker with day/opponent identity, every
lookup here degrades gracefully to a neutral default. Loads lazily on first use
and caches per-process (module-level singleton via ``get_meta_lookup``); if the
DB doesn't exist, or a given day/agent/deck/card is unknown, callers get neutral
defaults instead of an exception.

Schema (see scripts that populate ``model/results.db``):
    days(id, date, competition_day, ...)
    meta_features_daily(card_id, day_id, source, elo_bucket_10p, ...)
    agent_elo_daily(agent_id, day_id, source, elo, ...)
    deck_elo_daily(deck_id, day_id, source, elo, ...)

Deciles (buckets 0..9) are precomputed for cards (``elo_bucket_10p``) but must be
derived here for agents/decks by ranking same-day elo. 0 == top decile (strongest).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

NEUTRAL_CARD_BUCKET = 4
NEUTRAL_AGENT_BUCKET = 4
NEUTRAL_DECK_BUCKET = -1          # "unknown deck" -- encoding.py remaps this to N_DECK_BUCKETS-1
NEUTRAL_DAY_INDEX = 0.5
MAX_COMPETITION_DAYS = 60

_SOURCE = "remote"                # only source populated in the catalog today


class MetaLookup:
    """Cached view over ``model/results.db``'s meta-game tables.

    All DB reads happen once, in ``_ensure_loaded``; nothing here re-queries per
    encode() call. If the DB is missing or empty, every accessor returns its
    neutral default instead of raising.
    """

    def __init__(self, db_path: str = "model/results.db"):
        self.db_path = db_path
        self._card_buckets: dict[int, dict[int, int]] = {}     # day_id -> {card_id -> bucket}
        self._agent_buckets: dict[int, dict[int, int]] = {}    # day_id -> {agent_id -> decile}
        self._deck_buckets: dict[int, dict[int, int]] = {}     # day_id -> {deck_id -> decile}
        self._day_by_date: dict[str, int] = {}
        self._competition_day_by_id: dict[int, int] = {}
        self._card_bucket_luts: dict[int, np.ndarray] = {}     # day_id -> dense lookup array (perf)
        self._loaded = False

    # ---- loading -----------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True             # set first: a load failure must not retry every call
        if not Path(self.db_path).exists():
            return
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            for r in conn.execute(
                "SELECT day_id, card_id, elo_bucket_10p FROM meta_features_daily WHERE source = ?",
                (_SOURCE,),
            ):
                self._card_buckets.setdefault(int(r["day_id"]), {})[int(r["card_id"])] = int(r["elo_bucket_10p"])

            agent_rows_by_day: dict[int, list[tuple[int, float]]] = {}
            for r in conn.execute(
                "SELECT day_id, agent_id, elo FROM agent_elo_daily WHERE source = ?", (_SOURCE,)
            ):
                agent_rows_by_day.setdefault(int(r["day_id"]), []).append((int(r["agent_id"]), float(r["elo"])))
            for day_id, agent_elos in agent_rows_by_day.items():
                ranked = sorted(agent_elos, key=lambda p: -p[1])
                n = len(ranked)
                for rank, (agent_id, _elo) in enumerate(ranked):
                    self._agent_buckets.setdefault(day_id, {})[agent_id] = min(9, rank * 10 // max(n, 1))

            deck_rows_by_day: dict[int, list[tuple[int, float]]] = {}
            for r in conn.execute(
                "SELECT day_id, deck_id, elo FROM deck_elo_daily WHERE source = ?", (_SOURCE,)
            ):
                deck_rows_by_day.setdefault(int(r["day_id"]), []).append((int(r["deck_id"]), float(r["elo"])))
            for day_id, deck_elos in deck_rows_by_day.items():
                ranked = sorted(deck_elos, key=lambda p: -p[1])
                n = len(ranked)
                for rank, (deck_id, _elo) in enumerate(ranked):
                    self._deck_buckets.setdefault(day_id, {})[deck_id] = min(9, rank * 10 // max(n, 1))

            for r in conn.execute("SELECT id, date, competition_day FROM days"):
                self._day_by_date[str(r["date"])] = int(r["id"])
                if r["competition_day"] is not None:
                    self._competition_day_by_id[int(r["id"])] = int(r["competition_day"])
        finally:
            conn.close()

    # ---- day resolution ------------------------------------------------------
    def resolve_day_id(self, date_str: str | None) -> int | None:
        """``date_str`` ("YYYY-MM-DD") -> its ``days.id``, or None if unknown."""
        if date_str is None:
            return None
        self._ensure_loaded()
        return self._day_by_date.get(date_str)

    def latest_day_id(self) -> int | None:
        """Most recently ingested day (max ``days.id``), or None if the catalog is empty."""
        self._ensure_loaded()
        if not self._day_by_date:
            return None
        return max(self._day_by_date.values())

    def day_index_norm(self, day_id: int | None) -> float:
        if day_id is None:
            return NEUTRAL_DAY_INDEX
        self._ensure_loaded()
        comp_day = self._competition_day_by_id.get(day_id)
        if comp_day is None:
            return NEUTRAL_DAY_INDEX
        return max(0.0, min(1.0, (comp_day - 1) / max(1, MAX_COMPETITION_DAYS - 1)))

    # ---- card buckets (vectorized: dense per-day lookup array) ---------------
    def _card_bucket_lut(self, day_id: int) -> np.ndarray:
        lut = self._card_bucket_luts.get(day_id)
        if lut is not None:
            return lut
        table = self._card_buckets.get(day_id, {})
        if table:
            max_id = max(table)
            lut = np.full(max_id + 1, NEUTRAL_CARD_BUCKET, dtype=np.int32)
            for cid, bucket in table.items():
                lut[cid] = bucket
        else:
            lut = np.empty(0, dtype=np.int32)
        self._card_bucket_luts[day_id] = lut
        return lut

    def card_bucket_array(self, card_ids: np.ndarray, day_id: int | None) -> np.ndarray:
        """``card_ids`` (any int shape) -> same-shape int32 array of meta buckets.
        card_id 0 (pad) and any id absent from the day's catalog -> NEUTRAL_CARD_BUCKET."""
        card_ids = np.asarray(card_ids)
        if day_id is None:
            return np.full(card_ids.shape, NEUTRAL_CARD_BUCKET, dtype=np.int32)
        self._ensure_loaded()
        lut = self._card_bucket_lut(day_id)
        out = np.full(card_ids.shape, NEUTRAL_CARD_BUCKET, dtype=np.int32)
        if lut.size:
            flat_ids = card_ids.reshape(-1).astype(np.int64)
            flat_out = out.reshape(-1)
            in_range = (flat_ids >= 0) & (flat_ids < lut.size)
            flat_out[in_range] = lut[flat_ids[in_range]]
        return out

    # ---- opponent buckets ------------------------------------------------------
    def agent_bucket(self, agent_id: int | None, day_id: int | None) -> int:
        if agent_id is None or day_id is None:
            return NEUTRAL_AGENT_BUCKET
        self._ensure_loaded()
        return self._agent_buckets.get(day_id, {}).get(agent_id, NEUTRAL_AGENT_BUCKET)

    def deck_bucket(self, deck_id: int | None, day_id: int | None) -> int:
        if deck_id is None or day_id is None:
            return NEUTRAL_DECK_BUCKET
        self._ensure_loaded()
        return self._deck_buckets.get(day_id, {}).get(deck_id, NEUTRAL_DECK_BUCKET)


_GLOBAL_LOOKUP: MetaLookup | None = None


def get_meta_lookup(db_path: str = "model/results.db") -> MetaLookup:
    """Process-wide cached MetaLookup singleton (first call's ``db_path`` wins)."""
    global _GLOBAL_LOOKUP
    if _GLOBAL_LOOKUP is None:
        _GLOBAL_LOOKUP = MetaLookup(db_path)
    return _GLOBAL_LOOKUP
