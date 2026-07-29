"""Functional tests for would-KO computation and dataset provenance."""

from __future__ import annotations

import dataclasses
import json
import multiprocessing
import random
import tempfile
import unittest
from pathlib import Path
import zipfile

import numpy as np

from cg import api
from rl import search_agent
from scripts.bc import build_bc_dataset as dataset_builder
from scripts.bc import build_bc_from_zips as zip_builder


@dataclasses.dataclass
class _Observation:
    current: dict
    select: dict | None = None


@dataclasses.dataclass
class _SearchState:
    searchId: int
    observation: _Observation


class WouldKOTests(unittest.TestCase):
    def setUp(self):
        self._api = {
            name: getattr(api, name)
            for name in (
                "to_observation_class",
                "search_begin",
                "search_step",
                "search_release",
                "search_end",
            )
        }
        self._determinize = search_agent._determinize
        self._attacks = search_agent._WK_ATTACKS

    def tearDown(self):
        for name, value in self._api.items():
            setattr(api, name, value)
        search_agent._determinize = self._determinize
        search_agent._WK_ATTACKS = self._attacks

    @staticmethod
    def _root():
        return {
            "select": {
                "type": 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [{"attackId": 123}],
            },
            "current": {
                "yourIndex": 0,
                "result": -1,
                "players": [{"prize": [1, 2]}, {"prize": [3, 4]}],
            },
        }

    def test_valid_zero_is_distinct_from_failed_computation(self):
        root = self._root()
        search_agent._WK_ATTACKS = {123: (None, False)}
        search_agent._determinize = lambda *args, **kwargs: {}
        api.to_observation_class = lambda obs: obs
        api.search_begin = lambda *args, **kwargs: _SearchState(
            1, _Observation(current=root["current"], select=root["select"])
        )
        api.search_step = lambda *args, **kwargs: _SearchState(
            2,
            _Observation(
                current={
                    "yourIndex": 1,
                    "result": 2,
                    "players": [{"prize": [1, 2]}, {"prize": [3, 4]}],
                }
            ),
        )
        api.search_release = lambda *args, **kwargs: None
        api.search_end = lambda *args, **kwargs: None

        flags, audit = search_agent.annotate_would_ko_with_audit(
            root, [1] * 60, object(), rng=random.Random(7)
        )
        self.assertEqual(flags[0], (0.0, 0.0, 0.0))
        self.assertTrue(audit["options"][0]["computed"])
        self.assertTrue(audit["options"][0]["zero_result"])
        self.assertEqual(audit["valid_trials"], 1)
        self.assertEqual(audit["failed_trials"], 0)

        # Hidden replay deck identity is provenance only. It is deliberately
        # absent from the feature API, so changing it cannot change annotation.
        annotations = []
        for hidden_deck in ([1] * 60, [2] * 60):
            self.assertTrue(dataset_builder._deck_hash(hidden_deck))
            replay_root = self._root()
            annotations.append(
                search_agent.annotate_would_ko_with_audit(
                    replay_root, [1] * 60, object(), rng=random.Random(7)
                )[0]
            )
        self.assertEqual(annotations[0], annotations[1])

        api.search_begin = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fail"))
        failed_flags, failed_audit = search_agent.annotate_would_ko_with_audit(
            self._root(), [1] * 60, object(), rng=random.Random(7)
        )
        self.assertEqual(failed_flags, {})
        self.assertFalse(failed_audit["options"][0]["computed"])
        self.assertFalse(failed_audit["options"][0]["zero_result"])
        self.assertEqual(failed_audit["valid_trials"], 0)
        self.assertEqual(failed_audit["failed_trials"], 1)

    def test_subselect_resolution_uses_minimum_count_and_seeded_choice(self):
        picks = []

        def step(_sid, selected):
            picks.append(selected)
            return _SearchState(
                2,
                _Observation(
                    current={"yourIndex": 1, "result": -1, "players": [{}, {}]}
                ),
            )

        fake_api = type("FakeAPI", (), {"search_step": staticmethod(step)})
        obs = {
            "current": {"yourIndex": 0, "result": -1, "players": [{}, {}]},
            "select": {
                "minCount": 1,
                "maxCount": 3,
                "option": [{}, {}, {}, {}, {}],
            },
        }
        audit = {}
        search_agent._advance_resolve(
            fake_api, 1, obs, 0, rng=random.Random(11), audit=audit
        )
        self.assertEqual(len(picks), 1)
        self.assertEqual(len(picks[0]), 1)
        self.assertEqual(audit["resolved_subselects"], 1)
        self.assertEqual(audit["ambiguous_subselects"], 1)


class DatasetMetadataTests(unittest.TestCase):
    def test_compact_audit_sidecar_preserves_zero_and_failure(self):
        meta = [
            {
                "episode_id": "ep",
                "side": 0,
                "step_id": 4,
                "new_episode": True,
                "player_name": "FitaLabs",
                "opponent_name": "Opponent",
                "outcome": -1,
                "is_self": True,
                "player_deck_hash": "a" * 64,
                "opponent_deck_hash": "b" * 64,
            },
            {
                "episode_id": "ep",
                "side": 0,
                "step_id": 4,
                "new_episode": False,
                "player_name": "FitaLabs",
                "opponent_name": "Opponent",
                "outcome": -1,
                "is_self": True,
                "player_deck_hash": "a" * 64,
                "opponent_deck_hash": "b" * 64,
            },
        ]
        wk = [{
            "episode_id": "ep",
            "side": 0,
            "step_id": 4,
            "audit": {
                "options": {
                    2: {
                        "requested_trials": 10,
                        "valid_trials": 10,
                        "failed_trials": 0,
                        "computed": True,
                        "zero_result": True,
                    },
                    3: {
                        "requested_trials": 10,
                        "valid_trials": 0,
                        "failed_trials": 10,
                        "computed": False,
                        "zero_result": False,
                    },
                }
            },
        }]
        episode_meta = dataset_builder.episode_meta_array(meta)
        audit_meta = dataset_builder.would_ko_meta_array(meta, wk)
        self.assertEqual(episode_meta.dtype, dataset_builder.EPISODE_META_DTYPE)
        self.assertEqual(episode_meta[0]["player_name"], "FitaLabs")
        self.assertTrue(episode_meta[0]["is_self"])
        self.assertEqual(episode_meta[0]["outcome"], -1)
        self.assertEqual(episode_meta[0]["player_deck_hash"], "a" * 64)
        self.assertEqual(episode_meta[0]["opponent_deck_hash"], "b" * 64)
        self.assertEqual(audit_meta[0]["row_start"], 0)
        self.assertEqual(audit_meta[0]["row_count"], 2)
        self.assertTrue(audit_meta[0]["computed"])
        self.assertTrue(audit_meta[0]["zero_result"])
        self.assertFalse(audit_meta[1]["computed"])
        self.assertEqual(audit_meta[1]["failed_trials"], 10)
        self.assertFalse(np.issubdtype(audit_meta.dtype, np.floating))

    def test_zip_collection_deduplicates_identical_ids_and_rejects_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            z1 = root / "one.zip"
            z2 = root / "two.zip"
            payload = json.dumps({"id": 1})
            with zipfile.ZipFile(z1, "w") as archive:
                archive.writestr("a/shared.json", payload)
            with zipfile.ZipFile(z2, "w") as archive:
                archive.writestr("b/shared.json", payload)
            tasks, sources, dedup = zip_builder._collect_tasks([z1, z2])
            self.assertEqual(len(tasks), 1)
            self.assertEqual(len(sources), 2)
            self.assertEqual(dedup["identical_duplicates_removed"], 1)

            with zipfile.ZipFile(z2, "w") as archive:
                archive.writestr("b/shared.json", json.dumps({"id": 2}))
            with self.assertRaisesRegex(ValueError, "collision"):
                zip_builder._collect_tasks([z1, z2])

    def test_spawn_worker_receives_authoritative_config(self):
        build_config = {
            "bc_would_ko": True,
            "bc_wk_nvar": 17,
            "bc_both_sides": False,
            "seed": 1234,
        }
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(
            1,
            initializer=zip_builder._init_worker,
            initargs=(build_config,),
        ) as pool:
            snapshot = pool.apply(zip_builder._worker_config_snapshot)
        self.assertEqual(snapshot, (True, 17, False, 1234))

    def test_enabled_feature_cannot_validate_without_computation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "no would-KO option"):
                zip_builder._validate_would_ko_dataset(
                    tmp,
                    True,
                    {
                        "eligible_options": 12,
                        "computed_options": 0,
                        "valid_trials": 0,
                    },
                )

    def test_resume_without_matching_contract_is_rejected(self):
        contract = {"build_fingerprint": "one"}
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "shard_0000"
            shard.mkdir()
            (shard / ".done").touch()
            with self.assertRaisesRegex(RuntimeError, "no build contract"):
                zip_builder._prepare_resume_manifest(tmp, contract)

        with tempfile.TemporaryDirectory() as tmp:
            zip_builder._prepare_resume_manifest(tmp, contract)
            with self.assertRaisesRegex(RuntimeError, "different config"):
                zip_builder._prepare_resume_manifest(
                    tmp, {"build_fingerprint": "two"}
                )

    def test_empty_shard_preserves_resume_and_merge_continuity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = root / "shards"
            output = root / "merged"
            self.assertEqual(
                zip_builder._flush_shard(
                    str(shards), 0, [], [], [], set(), stats={}
                ),
                0,
            )
            self.assertEqual(
                zip_builder._flush_shard(
                    str(shards),
                    1,
                    [{"feature": np.array([1.0], dtype=np.float32)}],
                    [0],
                    [False],
                    set(),
                    stats={},
                ),
                1,
            )
            self.assertEqual(zip_builder._discover_shards(str(shards)), 2)
            self.assertEqual(
                zip_builder._merge_shards(
                    str(shards), str(output), n_shards=2
                ),
                1,
            )
            np.testing.assert_array_equal(
                np.load(output / "__labels__.npy"), np.array([0])
            )


if __name__ == "__main__":
    unittest.main()
