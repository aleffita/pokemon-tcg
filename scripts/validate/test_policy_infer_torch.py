"""Functional checks for the strict MLX -> PyTorch inference mirror.

Run with::

    uv run python scripts/validate/test_policy_infer_torch.py

These tests deliberately use the checkpoint's embedded architecture metadata;
``configs/train_config.json`` is not involved in inference loading.
"""
from __future__ import annotations

import os
import tempfile
import numpy as np
import torch

from rl.encoder.card_features import get_card_table
from rl.encoder.enc_constants import MAX_OPTIONS, N_ACTIONS, OPT_STRUCT
from rl.policy_infer_torch import (
    checkpoint_arch_config,
    checkpoint_inference_config,
    load_mlx_checkpoint,
    load_torch_inference_checkpoint,
    save_torch_inference_checkpoint,
)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT = os.path.join(ROOT, "model", "bc_model", "bc_best_mlx_final.pkl")
REAL_SMOKE_DATASET = os.path.join(
    ROOT, "data", "bc_data", "bc_smoke_2026_07_28"
)

_INT_KEYS = {
    "select_type", "select_context", "effect_id",
    "self_deck_id", "opp_deck_id", "self_prize_id", "opp_prize_id",
    "self_hand_id", "opp_hand_id", "self_discard_id", "opp_discard_id",
    "stadium_id", "self_unit_top_id", "self_unit_preevo_id",
    "self_unit_tool_id", "self_unit_energy_id", "opp_unit_top_id",
    "opp_unit_preevo_id", "opp_unit_tool_id", "opp_unit_energy_id",
    "opt_src_pos", "opt_tgt_pos", "opt_src_card", "opt_tgt_card",
    "opt_verb", "opt_attack_id",
}


def _real_observation() -> dict[str, torch.Tensor]:
    if not os.path.isfile(os.path.join(REAL_SMOKE_DATASET, "__labels__.npy")):
        raise FileNotFoundError(
            "real smoke dataset is required; build it with "
            "`uv run tcg-build-bc data/bc_data/bc_smoke_2026_07_28 "
            "data/bc_replay_zip/2026-07-28.zip --config configs/smoke.json`"
        )
    arrays = {
        name[:-4]: np.asarray(
            np.load(os.path.join(REAL_SMOKE_DATASET, name), mmap_mode="r")[0:1]
        ).copy()
        for name in os.listdir(REAL_SMOKE_DATASET)
        if name.endswith(".npy")
        and not name.startswith("__")
        and name != "episode_meta.npy"
    }
    return {
        key: torch.as_tensor(value, dtype=torch.int64 if key in _INT_KEYS else torch.float32)
        for key, value in arrays.items()
    }


def test_checkpoint_config_is_authoritative() -> None:
    import pickle

    with open(CHECKPOINT, "rb") as fh:
        state = pickle.load(fh)
    cfg = checkpoint_arch_config(state)
    inference_cfg = checkpoint_inference_config(state)
    assert cfg["nlayers"] == 4
    assert cfg["scratch_registers"] == 16
    assert cfg["has_learned_init"] is True
    assert inference_cfg["bc_would_ko"] is False
    assert inference_cfg["provenance"] == "legacy-checkpoint-default"
    print("  PASS: checkpoint arch_config is loaded independently of JSON")


def test_strict_fp32_load_and_forward() -> None:
    model, cfg = load_mlx_checkpoint(CHECKPOINT, get_card_table())
    assert cfg["nlayers"] == len(model.encoder.layers)
    assert model.scratch.shape == (cfg["scratch_registers"], cfg["d_model"])
    assert next(model.parameters()).dtype == torch.float32

    obs = _real_observation()
    with torch.inference_mode():
        logits, value, memory = model.logits_value(obs)
    assert logits.shape == (1, N_ACTIONS)
    assert value.shape == (1,)
    assert memory.shape == (1, cfg["scratch_registers"], cfg["d_model"])
    assert logits.dtype == torch.float32
    assert memory.dtype == torch.float32
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()
    legal = obs["action_mask"] > 0.5
    assert torch.all(logits[legal] > -65504)
    assert torch.all(logits[~legal] == -65504)
    print("  PASS: strict checkpoint load, FP32 forward, mask and memory shapes")


def test_padding_id_zero_is_zero() -> None:
    model, _ = load_mlx_checkpoint(CHECKPOINT, get_card_table())
    ids = torch.zeros((1, 3), dtype=torch.int64)
    out = model._card_emb(ids)
    assert torch.count_nonzero(out).item() == 0
    print("  PASS: card ID zero contributes no embedding")


def test_memory_is_persistent_and_shape_checked() -> None:
    model, cfg = load_mlx_checkpoint(CHECKPOINT, get_card_table())
    obs = _real_observation()
    with torch.inference_mode():
        _, _, memory1 = model.logits_value(obs)
        _, _, memory2 = model.logits_value(obs, memory_in=memory1)
    assert memory1.shape == memory2.shape
    assert not torch.equal(memory1, memory2)
    try:
        model.logits_value(obs, memory_in=torch.zeros(1, 4, cfg["d_model"], dtype=torch.float32))
    except RuntimeError:
        pass
    else:
        raise AssertionError("wrong scratch-register shape was accepted")
    print("  PASS: memory_out feeds the next step and bad shape fails")


def test_portable_fp32_checkpoint_round_trip_is_strict() -> None:
    card_table = get_card_table()
    with tempfile.TemporaryDirectory(prefix="ptcg_torch_checkpoint_") as out:
        path = os.path.join(out, "model.pt")
        saved_cfg = save_torch_inference_checkpoint(CHECKPOINT, path, card_table)
        model, loaded_cfg = load_torch_inference_checkpoint(path, card_table)
        assert loaded_cfg == saved_cfg
        assert all(
            not value.is_floating_point() or value.dtype == torch.float32
            for value in model.state_dict().values()
        )

        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert "inference_config" in payload
        assert "training_config" in payload
        assert payload["inference_config"] == saved_cfg["inference_config"]
        key = next(iter(payload["state_dict"]))
        payload["state_dict"][f"{key}.unexpected"] = payload["state_dict"][key]
        broken = os.path.join(out, "broken.pt")
        torch.save(payload, broken)
        try:
            load_torch_inference_checkpoint(broken, card_table)
        except ValueError as exc:
            assert "unexpected" in str(exc)
        else:
            raise AssertionError("checkpoint with an unexpected tensor was accepted")
    print("  PASS: portable artifact is all-FP32 and rejects state drift")


def main() -> None:
    print("=== PyTorch inference mirror validation ===")
    test_checkpoint_config_is_authoritative()
    test_strict_fp32_load_and_forward()
    test_padding_id_zero_is_zero()
    test_memory_is_persistent_and_shape_checked()
    test_portable_fp32_checkpoint_round_trip_is_strict()
    print("ALL PASSED")


if __name__ == "__main__":
    main()


__all__ = [
    "test_checkpoint_config_is_authoritative",
    "test_strict_fp32_load_and_forward",
    "test_padding_id_zero_is_zero",
    "test_memory_is_persistent_and_shape_checked",
    "test_portable_fp32_checkpoint_round_trip_is_strict",
]
