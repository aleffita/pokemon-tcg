"""Focused relational and idempotency checks for the disposable results DB."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zipfile

from rl.results_db import (
    DatabaseRebuildRequired,
    IdempotencyConflict,
    MatchParticipant,
    ResultsDB,
    deck_fingerprint,
)
from scripts.build_card_stats import process_zip


ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = ROOT / "model" / "checkpoint" / "smoke"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _deck(card_id: int, *, replacement: int | None = None) -> list[int]:
    cards = [card_id] * 60
    if replacement is not None:
        cards[-6:] = [replacement] * 6
    return cards


def _episode(
    left_deck: list[int],
    right_deck: list[int],
    *,
    episode_id: str | None = None,
    teams: tuple[str, str] = ("FitaLabs", "Opponent"),
) -> dict:
    info: dict = {"TeamNames": list(teams)}
    if episode_id is not None:
        info["EpisodeId"] = episode_id
    return {
        "info": info,
        "steps": [
            [
                {"action": left_deck, "status": "ACTIVE"},
                {"action": right_deck, "status": "ACTIVE"},
            ],
            [
                {"action": [0], "status": "DONE"},
                {"action": [0], "status": "DONE"},
            ],
        ],
        "rewards": [1, -1],
    }


class RelationalResultsDBTests(unittest.TestCase):
    def setUp(self) -> None:
        SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
        self.tempdir = tempfile.TemporaryDirectory(
            dir=SMOKE_ROOT,
            prefix="results-db-relational-",
        )
        self.workdir = Path(self.tempdir.name)
        self.db = ResultsDB(self.workdir / "results.db")

    def tearDown(self) -> None:
        self.db.close()
        self.tempdir.cleanup()

    def _observed_participants(
        self,
        left_deck: int,
        right_deck: int,
        *,
        archive_date: str = "2026-07-28",
    ) -> list[MatchParticipant]:
        result = []
        for seat, (name, deck_id, reward, outcome) in enumerate(
            (
                ("FitaLabs", left_deck, 1.0, 1),
                ("Opponent", right_deck, -1.0, -1),
            )
        ):
            team_id = self.db.get_or_create_team(name)
            submission_id = self.db.observe_submission(
                team_id,
                [deck_id],
                source="remote",
                archive_date=archive_date,
            )
            result.append(
                MatchParticipant(
                    seat=seat,
                    team_id=team_id,
                    submission_id=submission_id,
                    deck_id=deck_id,
                    reward=reward,
                    outcome=outcome,
                )
            )
        return result

    def test_schema_enforces_foreign_keys_and_refuses_legacy_db(self) -> None:
        self.assertEqual(
            self.db.conn.execute("PRAGMA foreign_keys").fetchone()[0],
            1,
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.conn.execute(
                """
                INSERT INTO deck_cards (deck_id, card_id, quantity)
                VALUES (999, 999, 1)
                """
            )
        self.assertEqual(
            self.db.conn.execute("PRAGMA foreign_key_check").fetchall(),
            [],
        )

        legacy_path = self.workdir / "legacy.db"
        legacy = sqlite3.connect(legacy_path)
        legacy.execute("CREATE TABLE legacy_only (id INTEGER PRIMARY KEY)")
        legacy.commit()
        legacy.close()
        with self.assertRaises(DatabaseRebuildRequired):
            ResultsDB(legacy_path)
        tables = {
            row[0]
            for row in sqlite3.connect(legacy_path).execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        self.assertEqual(tables, {"legacy_only"})

    def test_exact_deck_identity_and_stale_lastrowid_regression(self) -> None:
        ninety_percent_same_left = _deck(101, replacement=102)
        ninety_percent_same_right = _deck(101, replacement=103)
        left_id = self.db.get_or_create_deck(
            ninety_percent_same_left,
            source="replay",
            name="near-left",
        )
        right_id = self.db.get_or_create_deck(
            ninety_percent_same_right,
            source="replay",
            name="near-right",
        )
        self.assertNotEqual(left_id, right_id)
        self.assertNotEqual(
            deck_fingerprint(ninety_percent_same_left),
            deck_fingerprint(ninety_percent_same_right),
        )
        self.assertEqual(
            self.db.get_or_create_deck(
                reversed(ninety_percent_same_left),
                source="replay",
            ),
            left_id,
        )

        draft_a = self.db.add_deck("draft-a", "arena")
        self.db.add_deck_cards(draft_a, [(201, 60)])
        draft_b = self.db.add_deck("draft-b", "arena")
        self.db.add_deck_cards(draft_b, [(202, 60)])
        self.assertNotEqual(draft_a, draft_b)
        # The ignored insert occurs after another successful insert. SQLite may
        # retain that unrelated lastrowid, so add_deck must re-select by name.
        self.assertEqual(self.db.add_deck("draft-a", "arena"), draft_a)
        self.assertEqual(
            self.db.conn.execute(
                "SELECT COUNT(*) FROM deck_cards WHERE deck_id = ?",
                (draft_a,),
            ).fetchone()[0],
            1,
        )
        self.assertEqual(
            self.db.conn.execute("PRAGMA foreign_key_check").fetchall(),
            [],
        )

    def test_submission_n_to_n_and_observed_identity_contract(self) -> None:
        deck_a = self.db.get_or_create_deck(_deck(301), source="replay")
        deck_b = self.db.get_or_create_deck(_deck(302), source="replay")
        team_id = self.db.get_or_create_team("FitaLabs")
        first = self.db.observe_submission(
            team_id,
            [deck_a, deck_b],
            source="remote",
            archive_date="2026-07-28",
        )
        repeated = self.db.observe_submission(
            team_id,
            [deck_b, deck_a],
            source="remote",
            archive_date="2026-07-28",
        )
        next_day = self.db.observe_submission(
            team_id,
            [deck_a, deck_b],
            source="remote",
            archive_date="2026-07-29",
        )
        different_deck = self.db.observe_submission(
            team_id,
            [deck_a],
            source="remote",
            archive_date="2026-07-28",
        )
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_day)
        self.assertNotEqual(first, different_deck)
        self.assertEqual(
            self.db.conn.execute(
                "SELECT COUNT(*) FROM submission_decks "
                "WHERE submission_id = ?",
                (first,),
            ).fetchone()[0],
            2,
        )

    def test_official_episode_idempotency_and_conflict_detection(self) -> None:
        deck_a = self.db.get_or_create_deck(_deck(401), source="replay")
        deck_b = self.db.get_or_create_deck(_deck(402), source="replay")
        participants = self._observed_participants(deck_a, deck_b)
        match_id = self.db.record_match(
            source="remote",
            external_episode_id="episode-401",
            source_observation_digest=_sha("same bytes"),
            archive_date="2026-07-28",
            archive_member="401.json",
            n_steps=2,
            participants=participants,
        )
        self.assertEqual(
            self.db.record_match(
                source="remote",
                external_episode_id="episode-401",
                source_observation_digest=_sha("same bytes"),
                archive_date="2026-07-28",
                archive_member="401.json",
                n_steps=2,
                participants=participants,
            ),
            match_id,
        )
        with self.assertRaises(IdempotencyConflict):
            self.db.record_match(
                source="remote",
                external_episode_id="episode-401",
                source_observation_digest=_sha("different bytes"),
                archive_date="2026-07-28",
                archive_member="401.json",
                n_steps=2,
                participants=participants,
            )
        self.assertEqual(
            self.db.conn.execute(
                "SELECT COUNT(*) FROM matches WHERE source = 'remote'"
            ).fetchone()[0],
            1,
        )
        self.assertEqual(
            self.db.conn.execute(
                "SELECT COUNT(*) FROM match_participants WHERE match_id = ?",
                (match_id,),
            ).fetchone()[0],
            2,
        )

    def test_zip_ingestion_is_exact_idempotent_and_does_not_store_replay(self) -> None:
        archive = self.workdir / "2026-07-28.zip"
        official = _episode(
            _deck(501, replacement=502),
            _deck(501, replacement=503),
            episode_id="episode-501",
        )
        fallback = _episode(
            _deck(504),
            _deck(505),
            teams=("FitaLabs", "Other team"),
        )
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr(
                "official.json",
                json.dumps(official, sort_keys=True),
            )
            output.writestr(
                "fallback.json",
                json.dumps(fallback, sort_keys=True),
            )
        self.assertEqual(process_zip(self.db, archive), 2)
        self.assertEqual(process_zip(self.db, archive), 0)
        self.assertEqual(
            self.db.conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0],
            2,
        )
        self.assertEqual(
            self.db.conn.execute(
                "SELECT COUNT(*) FROM match_participants"
            ).fetchone()[0],
            4,
        )
        self.assertEqual(
            self.db.conn.execute("SELECT COUNT(*) FROM match_steps").fetchone()[0],
            0,
        )
        fallback_row = self.db.conn.execute(
            "SELECT external_episode_id, archive_date FROM matches "
            "WHERE archive_member = 'fallback.json'"
        ).fetchone()
        self.assertIsNone(fallback_row["external_episode_id"])
        self.assertEqual(fallback_row["archive_date"], "2026-07-28")
        schema_text = "\n".join(
            str(row[0] or "")
            for row in self.db.conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table'"
            )
        ).lower()
        self.assertNotIn("replay_html", schema_text)
        self.assertNotIn("replay_json", schema_text)

        conflicting_archive = self.workdir / "conflicting.zip"
        changed = _episode(
            _deck(501, replacement=502),
            _deck(501, replacement=503),
            episode_id="episode-501",
            teams=("FitaLabs", "Renamed opponent"),
        )
        with zipfile.ZipFile(conflicting_archive, "w") as output:
            output.writestr(
                "official.json",
                json.dumps(changed, sort_keys=True),
            )
        with self.assertRaises(IdempotencyConflict):
            process_zip(
                self.db,
                conflicting_archive,
                archive_date="2026-07-28",
            )

    def test_current_tournament_and_dashboard_compatibility_surface(self) -> None:
        tournament_id = self.db.add_tournament(
            "2026-07-29T00:00:00",
            "agent.tar.gz",
            1,
            "smoke",
            [
                {
                    "opponent": "public_lb1000",
                    "w": 1,
                    "l": 0,
                    "d": 0,
                    "wr": 1.0,
                }
            ],
            {"w": 1, "l": 0, "d": 0, "wr": 1.0},
        )
        matchup_id = self.db.conn.execute(
            "SELECT id FROM matchups WHERE tournament_id = ?",
            (tournament_id,),
        ).fetchone()[0]
        our_deck = self.db.add_deck("local-us", "arena")
        self.db.add_deck_cards(our_deck, [(601, 60)])
        opp_deck = self.db.add_deck("local-them", "arena")
        self.db.add_deck_cards(opp_deck, [(602, 60)])
        match_id = self.db.add_match(
            matchup_id,
            0,
            "local",
            "agent.tar.gz",
            our_deck,
            "public_lb1000",
            opp_deck,
            0,
            1,
            2,
        )
        self.assertEqual(
            self.db.add_match(
                matchup_id,
                0,
                "local",
                "agent.tar.gz",
                our_deck,
                "public_lb1000",
                opp_deck,
                0,
                1,
                2,
            ),
            match_id,
        )
        with self.assertRaises(IdempotencyConflict):
            self.db.add_match(
                matchup_id,
                0,
                "local",
                "agent.tar.gz",
                our_deck,
                "public_lb1000",
                opp_deck,
                0,
                -1,
                2,
            )
        self.db.conn.executemany(
            """
            INSERT INTO match_card_usage (
                match_id, card_id, player_side, quantity
            ) VALUES (?, ?, ?, ?)
            """,
            (
                (match_id, 601, 0, 60),
                (match_id, 602, 1, 60),
            ),
        )
        self.db.add_match_steps(
            match_id,
            [
                {
                    "step_num": 0,
                    "player_idx": 0,
                    "turn": 1,
                    "action": "[0]",
                    "status": "ACTIVE",
                }
            ],
        )
        self.assertTrue(self.db.compute_card_elo(source="local"))
        self.assertTrue(self.db.compute_deck_elo(source="local"))
        self.assertIsNotNone(self.db.get_card_elo(601, source="local"))
        self.assertIsNotNone(self.db.get_deck_elo(our_deck, source="local"))
        self.assertEqual(len(self.db.get_all_runs()), 1)
        self.assertEqual(len(self.db.compute_elos()), 2)
        self.assertEqual(
            self.db.conn.execute("PRAGMA foreign_key_check").fetchall(),
            [],
        )


if __name__ == "__main__":
    unittest.main()
