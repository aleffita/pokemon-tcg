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
from rl.policy_infer_torch import checkpoint_arch_config, load_mlx_checkpoint
from scripts.validate.make_synthetic_data import make_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT = os.path.join(ROOT, "model", "bc_model", "bc_best_mlx_final.pkl")

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


def _synthetic_observation() -> dict[str, torch.Tensor]:
    with tempfile.TemporaryDirectory(prefix="ptcg_torch_mirror_") as out:
        make_dataset(1, out, seed=7)
        arrays = {
            name[:-4]: np.load(os.path.join(out, name))
            for name in os.listdir(out)
            if name.endswith(".npy") and not name.startswith("__")
        }
    # ``make_synthetic_data`` predates the source-effect mask field.
    arrays["effect_mask"] = np.zeros((1, 2), dtype=np.float32)
    # Keep legal options within the first compiled bucket. This isolates the
    # mask/shape assertion from sparse high-index option truncation.
    arrays["action_mask"][:] = 0.0
    arrays["action_mask"][0, :3] = 1.0
    return {
        key: torch.as_tensor(value, dtype=torch.int64 if key in _INT_KEYS else torch.float16)
        for key, value in arrays.items()
    }


def test_checkpoint_config_is_authoritative() -> None:
    import pickle

    with open(CHECKPOINT, "rb") as fh:
        state = pickle.load(fh)
    cfg = checkpoint_arch_config(state)
    assert cfg["nlayers"] == 4
    assert cfg["scratch_registers"] == 16
    assert cfg["has_learned_init"] is True
    print("  PASS: checkpoint arch_config is loaded independently of JSON")


def test_strict_fp16_load_and_forward() -> None:
    model, cfg = load_mlx_checkpoint(CHECKPOINT, get_card_table())
    assert cfg["nlayers"] == len(model.encoder.layers)
    assert model.scratch.shape == (cfg["scratch_registers"], cfg["d_model"])
    assert next(model.parameters()).dtype == torch.float16

    obs = _synthetic_observation()
    with torch.inference_mode():
        logits, value, memory = model.logits_value(obs)
    assert logits.shape == (1, N_ACTIONS)
    assert value.shape == (1,)
    assert memory.shape == (1, cfg["scratch_registers"], cfg["d_model"])
    assert logits.dtype == torch.float16
    assert memory.dtype == torch.float16
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()
    legal = obs["action_mask"] > 0.5
    assert torch.all(logits[legal] > -65504)
    assert torch.all(logits[~legal] == -65504)
    print("  PASS: strict checkpoint load, FP16 forward, mask and memory shapes")


def test_padding_id_zero_is_zero() -> None:
    model, _ = load_mlx_checkpoint(CHECKPOINT, get_card_table())
    ids = torch.zeros((1, 3), dtype=torch.int64)
    out = model._card_emb(ids)
    assert torch.count_nonzero(out).item() == 0
    print("  PASS: card ID zero contributes no embedding")


def test_memory_is_persistent_and_shape_checked() -> None:
    model, cfg = load_mlx_checkpoint(CHECKPOINT, get_card_table())
    obs = _synthetic_observation()
    with torch.inference_mode():
        _, _, memory1 = model.logits_value(obs)
        _, _, memory2 = model.logits_value(obs, memory_in=memory1)
    assert memory1.shape == memory2.shape
    assert not torch.equal(memory1, memory2)
    try:
        model.logits_value(obs, memory_in=torch.zeros(1, 4, cfg["d_model"], dtype=torch.float16))
    except RuntimeError:
        pass
    else:
        raise AssertionError("wrong scratch-register shape was accepted")
    print("  PASS: memory_out feeds the next step and bad shape fails")


def test_mlx_float32_equivalence() -> None:
    """Compare the same checkpoint and observation before FP16 quantization."""
    import mlx.core as mx
    import mlx.nn as mlx_nn
    from rl.policy_mlx import build_token_net_mlx

    with open(CHECKPOINT, "rb") as fh:
        state = __import__("pickle").load(fh)
    card_table = get_card_table()
    mlx_model = build_token_net_mlx(card_table, state["arch_config"])
    leaves = mlx_nn.utils.tree_flatten(state["model"])
    model_keys = {key for key, _ in mlx_nn.utils.tree_flatten(mlx_model.parameters())}
    mlx_model.update(mlx_nn.utils.tree_unflatten([
        (key, mx.array(value)) for key, value in leaves if key in model_keys
    ]))
    mlx_model.eval()

    obs = _synthetic_observation()
    mlx_obs = {
        key: mx.array(value.numpy().astype(np.int32 if key in _INT_KEYS else np.float32))
        for key, value in obs.items()
    }
    torch_model, _ = load_mlx_checkpoint(CHECKPOINT, card_table, dtype=torch.float32)
    obs_fp32 = {
        key: value.to(dtype=torch.float32) if key not in _INT_KEYS else value
        for key, value in obs.items()
    }
    with torch.inference_mode():
        torch_logits, torch_value, torch_memory = torch_model.logits_value(obs_fp32)
    mlx_logits, mlx_value, mlx_memory = mlx_model.logits_value(mlx_obs)
    mx.eval(mlx_logits, mlx_value, mlx_memory)

    # Only compare legal options directly: masked FP32 sentinel conventions
    # differ (-1e9 in MLX vs finite FP16-safe sentinel in the mirror).
    legal = obs["action_mask"].numpy() > 0.5
    mlx_l = np.asarray(mlx_logits)[legal]
    torch_l = torch_logits.numpy()[legal]
    assert np.max(np.abs(mlx_l - torch_l)) < 1e-3
    assert np.max(np.abs(np.asarray(mlx_value) - torch_value.numpy())) < 1e-3
    assert np.max(np.abs(np.asarray(mlx_memory) - torch_memory.numpy())) < 1e-3
    assert int(np.argmax(np.asarray(mlx_logits))) == int(torch_logits.argmax().item())
    print("  PASS: MLX/PyTorch FP32 logits, value, memory and argmax agree")


def main() -> None:
    print("=== PyTorch inference mirror validation ===")
    test_checkpoint_config_is_authoritative()
    test_strict_fp16_load_and_forward()
    test_padding_id_zero_is_zero()
    test_memory_is_persistent_and_shape_checked()
    test_mlx_float32_equivalence()
    print("ALL PASSED")


if __name__ == "__main__":
    main()


__all__ = [
    "test_checkpoint_config_is_authoritative",
    "test_strict_fp16_load_and_forward",
    "test_padding_id_zero_is_zero",
    "test_memory_is_persistent_and_shape_checked",
    "test_mlx_float32_equivalence",
]
