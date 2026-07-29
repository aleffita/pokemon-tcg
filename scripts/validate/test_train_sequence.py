"""Regression tests for daily-corpus orchestration contracts."""

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

from rl.train_config import TrainConfig
from scripts.train_sequence import (
    _checkpoint_phase_id,
    _dataset_contract_issues,
    _phase_identity,
)


class TrainSequenceContractTests(unittest.TestCase):
    def test_dataset_manifest_must_match_current_build_contract(self):
        cfg = TrainConfig(
            bc_would_ko=True,
            bc_wk_nvar=10,
            bc_flush=200,
            seed=13,
        )
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            for name in (
                "__labels__.npy",
                "__would_ko_meta__.npy",
                "action_mask.npy",
                "episode_meta.npy",
            ):
                (dataset / name).touch()
            manifest = {
                "build_fingerprint": "dataset-one",
                "config": {
                    "bc_would_ko": True,
                    "bc_wk_nvar": 10,
                    "bc_both_sides": True,
                    "seed": 13,
                    "max_episodes": 0,
                    "bc_flush": 200,
                    "self_aliases": ["Alef Oliveira", "FitaLabs"],
                },
                "would_ko": {
                    "enabled": True,
                    "status": "computed",
                    "n_var": 10,
                },
            }
            (dataset / "dataset_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertEqual(_dataset_contract_issues(dataset, cfg), [])

            manifest["would_ko"]["n_var"] = 9
            (dataset / "dataset_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertTrue(_dataset_contract_issues(dataset, cfg))

    def test_phase_identity_freezes_dataset_and_train_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            config = root / "train.json"
            config.write_text('{"batch_size": 128}\n', encoding="utf-8")
            (dataset / "dataset_manifest.json").write_text(
                '{"build_fingerprint":"one"}\n', encoding="utf-8"
            )
            identity_one = _phase_identity(
                "2026-07-27", dataset, config, dry_run=False
            )[0]
            config.write_text('{"batch_size": 256}\n', encoding="utf-8")
            identity_two = _phase_identity(
                "2026-07-27", dataset, config, dry_run=False
            )[0]
            self.assertNotEqual(identity_one, identity_two)

    def test_checkpoint_phase_identity_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pkl"
            with path.open("wb") as handle:
                pickle.dump(
                    {"phase_id": "phase-27", "epoch": 3, "model": {}},
                    handle,
                )
            self.assertEqual(_checkpoint_phase_id(path), "phase-27")


if __name__ == "__main__":
    unittest.main()
