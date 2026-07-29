"""Validate the lateral PyTorch planner with real smoke data and sidecar trees."""
from __future__ import annotations

import json
import importlib
import os
import tempfile

import numpy as np
import torch

from rl.prospective_input_adapter import (
    PROSPECTIVE_INPUT_ADAPTER_VERSION,
    load_real_prospective_planner_batch,
)
from rl.prospective_planner_mlx import (
    ProspectivePlannerMLX,
    save_prospective_mlx_checkpoint,
)
from rl.prospective_planner_torch import (
    ProspectivePlannerTorch,
    convert_mlx_prospective_checkpoint,
    load_prospective_torch_checkpoint,
    save_prospective_torch_checkpoint,
)
from rl.prospective_schema import (
    BRANCH_POLICY_SCORE,
    BRANCH_VALID,
    EXPECTED_PRIZES,
    KO_PROBABILITY,
    MATCH_TIME_AXIS,
    ROLLOUT_DEPTH_AXIS,
    BRANCH_ACTION_AXIS,
    ENTITY_ZONE_RELATION_AXIS,
    SCALAR_RETURN,
    SCALAR_VALUE,
    TERMINAL_PROBABILITY,
    UNCERTAINTY,
    ProspectivePlannerConfig,
    coordinates_from_episode_meta,
    encode_entity_zone_relation,
)


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SMOKE_DATA = os.path.join(ROOT, "data", "bc_data", "bc_smoke_2026_07_28")
SMOKE_CONFIG = os.path.join(ROOT, "configs", "smoke.json")
BATCH_SIZE = 128


def _real_smoke_coordinate_audit():
    if not os.path.isfile(SMOKE_CONFIG):
        raise AssertionError(f"required smoke config is missing: {SMOKE_CONFIG}")
    with open(SMOKE_CONFIG, encoding="utf-8") as stream:
        smoke_config = json.load(stream)
    assert smoke_config["batch_size"] == BATCH_SIZE
    arrays = {
        name[:-4]: np.load(os.path.join(SMOKE_DATA, name), mmap_mode="r")
        for name in os.listdir(SMOKE_DATA)
        if name.endswith(".npy")
    }
    if len(arrays["__labels__"]) < BATCH_SIZE:
        raise AssertionError("smoke dataset has fewer than 128 real rows")

    action_mask = np.asarray(
        arrays["action_mask"][:BATCH_SIZE, :-1], dtype=np.float32
    )
    branch_valid_full = action_mask > 0.5
    last_valid = np.where(
        branch_valid_full,
        np.arange(branch_valid_full.shape[1])[None, :],
        -1,
    ).max(axis=1)
    branch_count = int(last_valid.max()) + 1
    branch_valid = branch_valid_full[:, :branch_count]

    root_coordinates = coordinates_from_episode_meta(
        arrays["episode_meta"][:BATCH_SIZE]
    )
    branch_coordinates = np.zeros(
        (BATCH_SIZE, branch_count, 4), dtype=np.int32
    )
    branch_coordinates[:, :, MATCH_TIME_AXIS] = root_coordinates[
        :, MATCH_TIME_AXIS
    ][:, None]
    branch_coordinates[:, :, ROLLOUT_DEPTH_AXIS] = 1
    branch_coordinates[:, :, BRANCH_ACTION_AXIS] = np.arange(
        branch_count, dtype=np.int32
    )[None, :]
    branch_coordinates[:, :, ENTITY_ZONE_RELATION_AXIS] = (
        encode_entity_zone_relation(
            arrays["opt_src_pos"][:BATCH_SIZE, :branch_count],
            arrays["opt_tgt_pos"][:BATCH_SIZE, :branch_count],
            arrays["opt_verb"][:BATCH_SIZE, :branch_count],
        )
    )
    return root_coordinates, branch_coordinates, branch_valid


def _real_sidecar_batch():
    config = ProspectivePlannerConfig()
    batch = load_real_prospective_planner_batch(SMOKE_DATA, config=config)
    assert batch.adapter_version == PROSPECTIVE_INPUT_ADAPTER_VERSION
    assert batch.context.shape[0] == 4
    assert np.count_nonzero(batch.branch_valid) > 0
    return config, batch


def test_real_batch_128_coordinate_invariants() -> None:
    roots, branches, branch_valid = _real_smoke_coordinate_audit()
    assert roots.shape == (BATCH_SIZE, 4)
    assert branches.shape[:2] == branch_valid.shape
    assert np.all(roots[:, 1:] == 0)
    assert np.all(branches[:, :, ROLLOUT_DEPTH_AXIS] == 1)
    print("  PASS: real physical batch=128 coordinate invariants")


def test_real_sidecar_forward_and_tree_mask() -> None:
    config, batch = _real_sidecar_batch()
    context = batch.context
    branches = batch.branch_tokens
    coordinates = batch.coordinates
    mask = batch.attention_mask
    branch_valid = batch.branch_valid
    assert np.isfinite(context).all()
    assert np.isfinite(branches).all()

    # Every real valid branch sees context, itself, and only its ancestor chain.
    for batch_index in range(branch_valid.shape[0]):
        for branch_index in np.flatnonzero(branch_valid[batch_index]):
            query = 1 + int(branch_index)
            assert mask[batch_index, 0, query, 0] == 0.0
            assert mask[batch_index, 0, query, query] == 0.0
            parent = int(batch.parent_index[batch_index, branch_index])
            if parent >= 0:
                assert mask[batch_index, 0, query, 1 + parent] == 0.0

    torch.manual_seed(13)
    model = ProspectivePlannerTorch(config).to(torch.float16).eval()
    with torch.inference_mode():
        outputs = model(
            torch.from_numpy(context),
            torch.from_numpy(branches),
            torch.from_numpy(coordinates),
            torch.from_numpy(mask),
            torch.from_numpy(branch_valid),
        )

    expected_shape = branch_valid.shape
    for key, value in outputs.items():
        assert value.shape == expected_shape, (key, value.shape)
        if key != BRANCH_VALID:
            assert value.dtype == torch.float16, (key, value.dtype)
            assert torch.isfinite(value).all(), key
    valid = torch.from_numpy(branch_valid)
    invalid = ~valid
    assert torch.all(outputs[BRANCH_POLICY_SCORE][invalid] == -65504.0)
    for key in (
        SCALAR_RETURN,
        SCALAR_VALUE,
        KO_PROBABILITY,
        EXPECTED_PRIZES,
        TERMINAL_PROBABILITY,
        UNCERTAINTY,
    ):
        assert torch.all(outputs[key][invalid] == 0)
    assert torch.all(outputs[SCALAR_RETURN][valid].abs() <= 1)
    assert torch.all(outputs[SCALAR_VALUE][valid].abs() <= 1)
    assert torch.all((outputs[KO_PROBABILITY][valid] >= 0)
                     & (outputs[KO_PROBABILITY][valid] <= 1))
    assert torch.all((outputs[TERMINAL_PROBABILITY][valid] >= 0)
                     & (outputs[TERMINAL_PROBABILITY][valid] <= 1))
    assert torch.all((outputs[EXPECTED_PRIZES][valid] >= 0)
                     & (outputs[EXPECTED_PRIZES][valid] <= config.max_prizes))
    assert torch.all(outputs[UNCERTAINTY][valid] > 0)
    print(
        "  PASS: real prospective_v1 trees, explicit shared adapter, "
        "four-axis mask, FP16 safe forward"
    )


def test_self_contained_checkpoint_round_trip() -> None:
    config, batch = _real_sidecar_batch()
    torch.manual_seed(13)
    model = ProspectivePlannerTorch(config).to(torch.float16).eval()
    inputs = (
        torch.from_numpy(batch.context),
        torch.from_numpy(batch.branch_tokens),
        torch.from_numpy(batch.coordinates),
        torch.from_numpy(batch.attention_mask),
        torch.from_numpy(batch.branch_valid),
    )
    with torch.inference_mode():
        expected = model(*inputs)
    with tempfile.TemporaryDirectory(prefix="ptcg_prospective_torch_") as out:
        path = os.path.join(out, "planner.pt")
        save_prospective_torch_checkpoint(path, model)
        restored, restored_config = load_prospective_torch_checkpoint(path)
        assert restored_config == config
        with torch.inference_mode():
            actual = restored(*inputs)
    for key in expected:
        assert torch.equal(expected[key], actual[key]), key
    print("  PASS: self-contained versioned FP16 checkpoint round-trip")


def test_strict_mlx_to_torch_conversion_parity() -> None:
    mx = importlib.import_module("mlx.core")
    config, batch = _real_sidecar_batch()
    mlx_model = ProspectivePlannerMLX(config)
    mlx_inputs = (
        mx.array(batch.context),
        mx.array(batch.branch_tokens),
        mx.array(batch.coordinates),
        mx.array(batch.attention_mask),
        mx.array(batch.branch_valid),
    )
    mlx_outputs = mlx_model(*mlx_inputs)
    mx.eval(mlx_outputs)

    with tempfile.TemporaryDirectory(prefix="ptcg_prospective_convert_") as out:
        mlx_path = os.path.join(out, "planner_mlx.pkl")
        torch_path = os.path.join(out, "planner_torch.pt")
        save_prospective_mlx_checkpoint(mlx_path, mlx_model)
        converted_config = convert_mlx_prospective_checkpoint(
            mlx_path, torch_path
        )
        torch_model, loaded_config = load_prospective_torch_checkpoint(torch_path)
        assert converted_config == loaded_config == config
        with torch.inference_mode():
            torch_outputs = torch_model(
                torch.from_numpy(batch.context),
                torch.from_numpy(batch.branch_tokens),
                torch.from_numpy(batch.coordinates),
                torch.from_numpy(batch.attention_mask),
                torch.from_numpy(batch.branch_valid),
            )

    for key, mlx_value in mlx_outputs.items():
        actual = torch_outputs[key].detach().cpu().numpy()
        expected = np.asarray(mlx_value)
        if key == BRANCH_VALID:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=5e-3,
                atol=5e-3,
                err_msg=key,
            )
    print("  PASS: strict MLX checkpoint conversion and cross-backend parity")


def main() -> None:
    print("=== Lateral PyTorch prospective planner validation ===")
    test_real_batch_128_coordinate_invariants()
    test_real_sidecar_forward_and_tree_mask()
    test_self_contained_checkpoint_round_trip()
    test_strict_mlx_to_torch_conversion_parity()
    print("ALL PASSED")


if __name__ == "__main__":
    main()


__all__ = [
    "test_real_batch_128_coordinate_invariants",
    "test_real_sidecar_forward_and_tree_mask",
    "test_self_contained_checkpoint_round_trip",
    "test_strict_mlx_to_torch_conversion_parity",
]
