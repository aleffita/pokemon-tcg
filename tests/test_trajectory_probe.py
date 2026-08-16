from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rl.encoder.encoding import SUBMIT_ACTION, build_mask
from rl.encoder.card_features import get_card_table
from rl.policy_infer_torch import load_inference_checkpoint
from scripts.rl.ppo_micro_update import (
    _model_input_digests,
    ppo_micro_update,
    save_candidate_checkpoint,
    validate_bundle,
)
from scripts.rl.trajectory_probe import (
    DateBoundEncoder,
    digest_tensor,
    APPROVED_STAGE4_ROOT_SHA256,
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
    with pytest.raises(ValueError, match="conflict"):
        wrapped.encode(
            {"current": {"date": "2026-08-12", "archive_date": "2026-08-11"}, "select": {"option": [], "minCount": 0, "maxCount": 1}},
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

    monkeypatch.setattr(
        "scripts.rl.trajectory_probe.sha256_file",
        lambda path: APPROVED_STAGE4_ROOT_SHA256,
    )
    monkeypatch.setattr("scripts.rl.trajectory_probe.load_inference_checkpoint", strict_loader)
    with pytest.raises(ValueError, match="strict mismatch"):
        load_stage4(checkpoint, object())
    assert calls == [(checkpoint, object())] or len(calls) == 1


def test_stage4_root_hash_is_rejected_before_loader(monkeypatch, tmp_path):
    checkpoint = tmp_path / "not_the_root.pkl"
    checkpoint.write_bytes(b"not the frozen root")
    calls = []
    monkeypatch.setattr("scripts.rl.trajectory_probe.load_inference_checkpoint", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="approved frozen root"):
        load_stage4(checkpoint, object())
    assert calls == []


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


def _sample_bundle():
    mask = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    memory = torch.zeros(1, 1, 2, dtype=torch.float32)
    samples = []
    rows = []
    for index, (action, reward, done, value) in enumerate(((0, 0.0, False, 0.0), (2, 1.0, True, 0.5))):
        model_input = {
            "action_mask": mask.reshape(1, -1).clone(),
            "feature": torch.tensor([[float(index)]], dtype=torch.float32),
        }
        sample = {
            "sample_index": index,
            "episode_id": "e",
            "env_step": index,
            "decision_index": index,
            "substep": 0,
            "model_input": model_input,
            "action_mask": mask.clone(),
            "memory_input": memory.clone(),
            "action": action,
            "behavior_logprob": float(torch.log_softmax(torch.tensor([0.2, -0.3, 0.1]), 0)[action]),
            "value": value,
            "reward": reward,
            "done": done,
        }
        samples.append(sample)
        rows.append(
            {
                "sample_index": index,
                "episode_id": "e",
                "env_step": index,
                "legal_action_mask_digest": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
                "memory_input_digest": digest_tensor(memory),
                "model_input_digests": _model_input_digests(model_input),
                "done": done,
            }
        )
    return samples, rows


def test_bundle_preserves_shape_order_and_digests():
    bundle, rows = _sample_bundle()
    validate_bundle(bundle, rows)
    rows[1]["legal_action_mask_digest"] = "wrong"
    with pytest.raises(ValueError, match="mask digest"):
        validate_bundle(bundle, rows)


def test_bundle_rejects_model_input_tensor_tamper():
    bundle, rows = _sample_bundle()
    validate_bundle(bundle, rows)
    bundle[1]["model_input"]["feature"][0, 0] = 99.0
    with pytest.raises(ValueError, match="model-input digest"):
        validate_bundle(bundle, rows)


class _ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.2, -0.3, 0.1], dtype=torch.float32))
        self.value = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def logits_value(self, model_input, memory_in=None):
        batch = model_input["action_mask"].shape[0]
        return self.logits.expand(batch, -1), self.value.expand(batch), memory_in


def test_ppo_micro_update_mutates_policy_and_reports_root_reference():
    bundle, rows = _sample_bundle()
    validate_bundle(bundle, rows)
    model = _ToyPolicy()
    root = copy.deepcopy(model)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    metrics = ppo_micro_update(model, root, bundle, learning_rate=1e-2)
    assert metrics["epochs"] == 1
    assert metrics["advantage_std"] == pytest.approx(1.0)
    assert metrics["root_reference_changed_parameters"] > 0
    assert any(not torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_candidate_checkpoint_strict_loads_via_inference_entrypoint(tmp_path):
    card_table = get_card_table()
    model, metadata = load_inference_checkpoint(
        Path("experiments/autoresearch/root/stage4_root.pkl"), card_table
    )
    path = tmp_path / "candidate.pt"
    save_candidate_checkpoint(
        path,
        model,
        metadata,
        root_sha256=APPROVED_STAGE4_ROOT_SHA256,
        sample_manifest_sha256="a" * 64,
        bundle_sha256="b" * 64,
        config={"algorithm": "PPO", "epochs": 1},
        diagnostics={"root_reference_kl_mean": 0.0},
        sample_manifest_content_sha256="c" * 64,
    )
    loaded, loaded_metadata = load_inference_checkpoint(path, card_table)
    assert next(loaded.parameters()).dtype == torch.float32
    assert loaded_metadata["arch_version"] == metadata["arch_version"]
