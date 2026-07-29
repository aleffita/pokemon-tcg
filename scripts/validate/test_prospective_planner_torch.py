"""Validate the lateral PyTorch planner with real smoke data and sidecar trees."""
from __future__ import annotations

import json
import importlib
from functools import lru_cache
import os
import pickle
import tempfile

import numpy as np
import torch

from rl.prospective_input_adapter import (
    PROSPECTIVE_INPUT_ADAPTER_VERSION,
    ProspectivePlannerNumpyBatch,
    load_real_prospective_planner_batch,
    load_real_prospective_planner_index,
)
from rl.prospective_planner_mlx import (
    ProspectivePlannerMLX,
    prospective_mlx_checkpoint_payload,
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
    N_PROSPECTIVE_AXES,
    ProspectivePlannerConfig,
    build_tree_attention_mask,
    coordinates_from_episode_meta,
    encode_entity_zone_relation,
)
from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder
from rl.policy_infer_torch import (
    load_mlx_checkpoint,
    load_torch_inference_checkpoint,
    save_torch_inference_checkpoint,
)


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SMOKE_DATA = os.path.join(ROOT, "data", "bc_data", "bc_smoke_2026_07_28")
SMOKE_CONFIG = os.path.join(ROOT, "configs", "smoke.json")
BATCH_SIZE = 128
LEGACY_TRUNK_CHECKPOINT = os.path.join(
    ROOT, "model", "bc_model", "bc_best_mlx_final.pkl"
)


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


def _real_trunk_context(
    batch: ProspectivePlannerNumpyBatch,
    config: ProspectivePlannerConfig,
    sidecar_index,
) -> np.ndarray:
    card_table = get_card_table()
    trunk, _ = load_mlx_checkpoint(LEGACY_TRUNK_CHECKPOINT, card_table)
    encoder = TokenEncoder(card_table)
    episode_meta = np.load(
        os.path.join(SMOKE_DATA, "episode_meta.npy"), mmap_mode="r"
    )
    arrays = {
        name[:-4]: np.load(os.path.join(SMOKE_DATA, name), mmap_mode="r")
        for name in os.listdir(SMOKE_DATA)
        if name.endswith(".npy")
        and not name.startswith("__")
        and name != "episode_meta.npy"
    }
    cache: dict[tuple[str, int, int], np.ndarray] = {}
    contexts: list[np.ndarray] = []
    for batch_index in range(len(batch.group_keys)):
        target = sidecar_index.group_target_keys[batch_index]
        context = cache.get(target)
        if context is None:
            lane_rows = np.flatnonzero(
                (episode_meta["episode_id"] == target[0])
                & (episode_meta["side"] == target[1])
            )
            memory = None
            for step in dict.fromkeys(
                int(episode_meta["step_id"][row]) for row in lane_rows
            ):
                decision_rows = lane_rows[
                    episode_meta["step_id"][lane_rows] == step
                ]
                observation = {
                    key: torch.as_tensor(
                        np.array(array[decision_rows], copy=True),
                        dtype=(
                            torch.int64
                            if key in encoder.int_keys
                            else torch.float16
                        ),
                    )
                    for key, array in arrays.items()
                }
                repeated_memory = (
                    None
                    if memory is None
                    else memory.expand(len(decision_rows), -1, -1)
                )
                with torch.inference_mode():
                    cls_out, _, pooled, _, memory_out = trunk._encode(
                        observation,
                        memory_in=repeated_memory,
                    )
                if step == target[2]:
                    context = torch.cat(
                        (
                            cls_out[:1, None, :],
                            pooled[:1, None, :],
                            memory_out[:1],
                        ),
                        dim=1,
                    )[0].detach().cpu().numpy().astype(np.float16)
                    cache[target] = context
                # Match the trainer contract: all expanded rows see the same
                # incoming memory, then the last subrow advances the lane once.
                memory = memory_out[-1:]
                if step == target[2]:
                    break
            assert context is not None
        assert context.shape[1] == config.d_model
        contexts.append(context)
    return np.stack(contexts).astype(np.float16)


@lru_cache(maxsize=1)
def _real_sidecar_batch():
    config = ProspectivePlannerConfig()
    sidecar_index = load_real_prospective_planner_index(SMOKE_DATA)
    raw = load_real_prospective_planner_batch(SMOKE_DATA, config=config)
    assert len(sidecar_index) == len(raw.group_keys)
    context = _real_trunk_context(raw, config, sidecar_index)
    context_coordinates = np.zeros(
        (
            context.shape[0],
            context.shape[1],
            N_PROSPECTIVE_AXES,
        ),
        dtype=np.int32,
    )
    context_coordinates[..., MATCH_TIME_AXIS] = raw.coordinates[:, :1, 0]
    coordinates = np.concatenate(
        (context_coordinates, raw.coordinates[:, 1:, :]), axis=1
    )
    attention_mask = build_tree_attention_mask(
        raw.parent_index,
        raw.branch_valid,
        context_length=context.shape[1],
    )
    batch = ProspectivePlannerNumpyBatch(
        context=context,
        branch_tokens=raw.branch_tokens,
        coordinates=coordinates,
        attention_mask=attention_mask,
        branch_valid=raw.branch_valid,
        parent_index=raw.parent_index,
        node_index=raw.node_index,
        group_keys=raw.group_keys,
    )
    batch.validate(config)
    assert batch.adapter_version == PROSPECTIVE_INPUT_ADAPTER_VERSION
    assert batch.context.shape == (4, 18, config.d_model)
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
            context_length = context.shape[1]
            query = context_length + int(branch_index)
            assert mask[batch_index, 0, query, 0] == 0.0
            assert mask[batch_index, 0, query, query] == 0.0
            parent = int(batch.parent_index[batch_index, branch_index])
            if parent >= 0:
                assert (
                    mask[
                        batch_index,
                        0,
                        query,
                        context_length + parent,
                    ]
                    == 0.0
                )

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
        "  PASS: real prospective_v1 trees, 18-token recurrent trunk context, "
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
        nested_mlx_path = os.path.join(out, "trainer_mlx.pkl")
        torch_path = os.path.join(out, "planner_torch.pt")
        save_prospective_mlx_checkpoint(mlx_path, mlx_model)
        with open(mlx_path, "rb") as stream:
            lateral_payload = pickle.load(stream)
        lateral_payload["optimizer_steps"] = 1
        with open(nested_mlx_path, "wb") as stream:
            pickle.dump({"prospective_planner": lateral_payload}, stream)
        converted_config = convert_mlx_prospective_checkpoint(
            nested_mlx_path, torch_path
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
    print(
        "  PASS: strict nested MLX checkpoint conversion and cross-backend parity"
    )


def test_combined_artifact_is_self_contained_and_legacy_gated() -> None:
    config, batch = _real_sidecar_batch()
    mlx_planner = ProspectivePlannerMLX(config)
    planner_payload = prospective_mlx_checkpoint_payload(mlx_planner)
    planner_payload["optimizer_steps"] = 0
    with open(LEGACY_TRUNK_CHECKPOINT, "rb") as stream:
        trainer_checkpoint = pickle.load(stream)
    trainer_checkpoint["prospective_planner"] = planner_payload

    with tempfile.TemporaryDirectory(prefix="ptcg_combined_artifact_") as out:
        mlx_path = os.path.join(out, "trainer.pkl")
        torch_path = os.path.join(out, "artifact.pt")
        with open(mlx_path, "wb") as stream:
            pickle.dump(trainer_checkpoint, stream)
        save_torch_inference_checkpoint(
            mlx_path, torch_path, get_card_table()
        )
        _, metadata = load_torch_inference_checkpoint(
            torch_path, get_card_table()
        )
        restored = metadata["prospective_planner_model"]
        assert restored is not None
        assert (
            metadata["inference_config"]["prospective_planner"]["enabled"]
            is False
        )
        artifact = torch.load(
            torch_path, map_location="cpu", weights_only=True
        )
        assert artifact["prospective_planner"] is not None
        assert artifact["prospective_planner"]["trained_optimizer_steps"] == 0
        with torch.inference_mode():
            outputs = restored(
                torch.from_numpy(batch.context),
                torch.from_numpy(batch.branch_tokens),
                torch.from_numpy(batch.coordinates),
                torch.from_numpy(batch.attention_mask),
                torch.from_numpy(batch.branch_valid),
            )
        assert outputs[BRANCH_POLICY_SCORE].shape == batch.branch_valid.shape
    print(
        "  PASS: combined artifact carries planner but legacy gate remains disabled"
    )


def main() -> None:
    print("=== Lateral PyTorch prospective planner validation ===")
    test_real_batch_128_coordinate_invariants()
    test_real_sidecar_forward_and_tree_mask()
    test_self_contained_checkpoint_round_trip()
    test_strict_mlx_to_torch_conversion_parity()
    test_combined_artifact_is_self_contained_and_legacy_gated()
    print("ALL PASSED")


if __name__ == "__main__":
    main()


__all__ = [
    "test_real_batch_128_coordinate_invariants",
    "test_real_sidecar_forward_and_tree_mask",
    "test_self_contained_checkpoint_round_trip",
    "test_strict_mlx_to_torch_conversion_parity",
    "test_combined_artifact_is_self_contained_and_legacy_gated",
]
