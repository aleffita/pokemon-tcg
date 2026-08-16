from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rl.encoder.encoding import SUBMIT_ACTION, build_mask
from scripts.rl.trajectory_probe import (
    DateBoundEncoder,
    digest_tensor,
    initial_memory,
    inspect_parquet_provenance,
    load_stage4,
    validate_meta_date,
    validate_rows,
    write_outputs,
)


class _Encoder:
    int_keys: set[str] = set()

    def encode(self, obs, picked=None, **kwargs):
        current = obs["current"]
        assert current["date"] == "2026-08-12"
        select = obs["select"]
        return {
            "action_mask": build_mask(select, picked),
            "feature": np.array([len(picked)], dtype=np.float32),
        }


class _Model:
    def __init__(self):
        self.learned_init = torch.zeros(2, 3)


def test_date_bound_encoder_requires_explicit_complete_date_and_injects_missing_engine_date(monkeypatch):
    class _Lookup:
        def resolve_day_id(self, value):
            assert value == "2026-08-12"
            return 30

        def day_index_norm(self, value):
            assert value == 30
            return 0.5

    monkeypatch.setattr("scripts.rl.trajectory_probe.get_meta_lookup", lambda: _Lookup())
    wrapped = DateBoundEncoder(_Encoder(), "2026-08-12")
    wrapped.encode(
        {"current": {"turn": 1}, "select": {"option": [], "minCount": 0, "maxCount": 1}},
        picked=set(),
    )
    with pytest.raises(ValueError, match="conflicts"):
        wrapped.encode(
            {"current": {"date": "2026-08-11"}, "select": {"option": [], "minCount": 0, "maxCount": 1}},
            picked=set(),
        )


def test_initial_memory_digest_is_stable_and_non_null():
    model = _Model()
    first = initial_memory(model)
    second = initial_memory(model)
    assert first.shape == (1, 2, 3)
    assert digest_tensor(first) == digest_tensor(second)


def test_strict_stage4_loader_does_not_fallback(monkeypatch, tmp_path):
    checkpoint = tmp_path / "stage4_root.pkl"
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    def strict_loader(path, card_table):
        calls.append((path, card_table))
        raise ValueError("strict mismatch")

    monkeypatch.setattr("scripts.rl.trajectory_probe.load_inference_checkpoint", strict_loader)
    with pytest.raises(ValueError, match="strict mismatch"):
        load_stage4(checkpoint, object())
    assert calls == [(checkpoint, object())] or len(calls) == 1


def test_rows_require_ordered_multiselect_substeps_and_terminal_reward():
    rows = [
        {
            "episode_id": "random-000", "decision_index": 0, "substep": 0,
            "terminal": False, "done": False, "legal_action_mask_digest": "m",
            "action_logprob": -0.1, "memory_input_digest": "i", "memory_output_digest": "o",
            "reward": 0.0,
        },
        {
            "episode_id": "random-000", "decision_index": 0, "substep": 1,
            "terminal": True, "done": True, "legal_action_mask_digest": "m2",
            "action_logprob": -0.2, "memory_input_digest": "i", "memory_output_digest": "o2",
            "reward": 1.0,
        },
    ]
    validate_rows(rows)
    rows[1]["substep"] = 2
    with pytest.raises(AssertionError, match="substep order"):
        validate_rows(rows)


def test_output_manifest_contains_exact_jsonl_digest(tmp_path):
    rows = [{"episode_id": "e", "action": 1}]
    manifest = {
        "metadata_date": "2026-08-12",
        "model_sha256": "model",
        "deck_sha256": "deck",
        "counts_by_mode": {"random": {"episodes": 1, "rows": 1, "terminals": 1}},
    }
    write_outputs(tmp_path, rows, manifest)
    jsonl = (tmp_path / "trajectory.jsonl").read_text()
    saved = json.loads((tmp_path / "trajectory.manifest.json").read_text())
    import hashlib
    assert saved["trajectory_sha256"] == hashlib.sha256(jsonl.encode()).hexdigest()
    assert (tmp_path / "trajectory.log").is_file()


def test_parquet_provenance_is_metadata_only():
    provenance = inspect_parquet_provenance(Path("data/bc_data/2026-08-12.parquet"))
    assert provenance["rows"] > 0
    assert provenance["used_for_model_input"] is False
    assert provenance["packed"] is False
