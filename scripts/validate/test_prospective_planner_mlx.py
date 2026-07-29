"""Validate the lateral MLX planner on real replay and rollout-sidecar data.

Coordinate invariants remain independently audited from ``episode_meta``.
Forward validation consumes the causal, target-free shared input adapter over
the real ``prospective_v1`` sidecar; no synthetic rows or hidden states are
created.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from rl.prospective_planner_mlx import (
    ProspectivePlannerMLX,
    group_relative_advantages,
    group_relative_objective,
    rope_nd_angles,
)
from rl.prospective_input_adapter import (
    ProspectivePlannerNumpyBatch,
    load_real_prospective_planner_batch,
)
from rl.prospective_schema import (
    BRANCH_POLICY_SCORE,
    BRANCH_VALID,
    ENTITY_ZONE_RELATION_AXIS,
    EXPECTED_PRIZES,
    KO_PROBABILITY,
    MATCH_TIME_AXIS,
    N_PROSPECTIVE_AXES,
    ProspectivePlannerConfig,
    SCALAR_RETURN,
    SCALAR_VALUE,
    TERMINAL_PROBABILITY,
    UNCERTAINTY,
    coordinates_from_episode_meta,
    decode_entity_zone_relation,
    encode_entity_zone_relation,
    validate_prospective_coordinates,
)


ROOT = Path(__file__).resolve().parents[2]
SMOKE_DATA = ROOT / "data" / "bc_data" / "bc_smoke_2026_07_28"
SMOKE_CONFIG = ROOT / "configs" / "smoke.json"
PHYSICAL_BATCH_SIZE = 128
PROSPECTIVE_DIR = SMOKE_DATA / "prospective_v1"


def _real_smoke_arrays() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with SMOKE_CONFIG.open(encoding="utf-8") as handle:
        config = json.load(handle)
    assert config["batch_size"] == PHYSICAL_BATCH_SIZE
    arrays = {
        name: np.load(SMOKE_DATA / f"{name}.npy", mmap_mode="r")
        for name in (
            "episode_meta",
            "opt_src_pos",
            "opt_tgt_pos",
            "opt_verb",
        )
    }
    assert len(arrays["episode_meta"]) >= PHYSICAL_BATCH_SIZE
    return arrays["episode_meta"][:PHYSICAL_BATCH_SIZE], arrays


def test_real_episode_coordinates_and_rope_invariants() -> None:
    episode_meta, _ = _real_smoke_arrays()
    coordinates = coordinates_from_episode_meta(episode_meta)
    coordinates = validate_prospective_coordinates(coordinates)
    assert coordinates.shape == (PHYSICAL_BATCH_SIZE, N_PROSPECTIVE_AXES)
    np.testing.assert_array_equal(
        coordinates[:, MATCH_TIME_AXIS],
        np.asarray(episode_meta["step_id"], dtype=np.int32),
    )
    # No rollout sidecar exists: only real match time is populated.
    assert np.count_nonzero(coordinates[:, 1:]) == 0

    cosine, sine = rope_nd_angles(
        mx.array(coordinates[None, ...]),
        head_dim=ProspectivePlannerConfig().d_model
        // ProspectivePlannerConfig().nhead,
    )
    mx.eval(cosine, sine)
    cosine_np = np.asarray(cosine)
    sine_np = np.asarray(sine)
    np.testing.assert_allclose(
        cosine_np * cosine_np + sine_np * sine_np,
        np.ones_like(cosine_np),
        rtol=2e-6,
        atol=2e-6,
    )

    # Two real rows with different match times must leave the other three
    # coordinate-axis rotations untouched.
    distinct = np.flatnonzero(
        coordinates[:, MATCH_TIME_AXIS] != coordinates[0, MATCH_TIME_AXIS]
    )
    assert len(distinct) > 0
    other = int(distinct[0])
    np.testing.assert_array_equal(coordinates[0, 1:], coordinates[other, 1:])
    np.testing.assert_allclose(cosine_np[0, 0, 1:], cosine_np[0, other, 1:])
    np.testing.assert_allclose(sine_np[0, 0, 1:], sine_np[0, other, 1:])


def test_real_entity_zone_relation_round_trip() -> None:
    _, arrays = _real_smoke_arrays()
    source = np.asarray(
        arrays["opt_src_pos"][:PHYSICAL_BATCH_SIZE], dtype=np.int32
    )
    target = np.asarray(
        arrays["opt_tgt_pos"][:PHYSICAL_BATCH_SIZE], dtype=np.int32
    )
    verb = np.asarray(
        arrays["opt_verb"][:PHYSICAL_BATCH_SIZE], dtype=np.int32
    )
    coordinate = encode_entity_zone_relation(source, target, verb)
    decoded_source, decoded_target, decoded_verb = (
        decode_entity_zone_relation(coordinate)
    )
    np.testing.assert_array_equal(decoded_source, source)
    np.testing.assert_array_equal(decoded_target, target)
    np.testing.assert_array_equal(decoded_verb, verb)
    assert coordinate.dtype == np.int32
    assert np.all(coordinate >= 0)

    # The encoded relation occupies exactly the declared fourth coordinate
    # axis when a real prospective sidecar eventually supplies branch rows.
    audit_coordinates = np.zeros(
        (*coordinate.shape, N_PROSPECTIVE_AXES), dtype=np.int32
    )
    audit_coordinates[..., ENTITY_ZONE_RELATION_AXIS] = coordinate
    validate_prospective_coordinates(audit_coordinates)


def test_isolated_mlx_parameter_contract() -> None:
    config = ProspectivePlannerConfig()
    config.validate()
    model = ProspectivePlannerMLX(config)
    parameters = dict(nn.utils.tree_flatten(model.parameters()))
    assert parameters
    assert {str(parameter.dtype) for parameter in parameters.values()} == {
        "mlx.core.float16"
    }
    required_paths = {
        "layers.0.norm1.weight",
        "layers.0.attention.q_proj.weight",
        "layers.0.attention.k_proj.weight",
        "layers.0.attention.v_proj.weight",
        "layers.0.attention.out_proj.weight",
        "layers.0.norm2.weight",
        "layers.0.ff1.weight",
        "layers.0.ff2.weight",
        "policy_head.weight",
        "return_head.weight",
        "value_head.weight",
        "ko_head.weight",
        "prize_head.weight",
        "terminal_head.weight",
        "uncertainty_head.weight",
    }
    assert required_paths <= set(parameters)


def _real_sidecar_forward_batch() -> tuple[
    ProspectivePlannerConfig,
    ProspectivePlannerNumpyBatch,
    np.ndarray,
]:
    config = ProspectivePlannerConfig()
    batch = load_real_prospective_planner_batch(
        SMOKE_DATA,
        sidecar_name=PROSPECTIVE_DIR.name,
        config=config,
    )
    nodes = np.load(
        PROSPECTIVE_DIR / "prospective_nodes.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    assert len(nodes) == 24
    assert len(batch.group_keys) == 4
    assert int(np.count_nonzero(batch.node_index >= 0)) == len(nodes)
    assert len(nodes) <= PHYSICAL_BATCH_SIZE
    return config, batch, nodes


def test_real_sidecar_forward_and_group_relative_mode() -> None:
    config, batch, nodes = _real_sidecar_forward_batch()
    branch_valid = batch.branch_valid
    assert int(branch_valid.sum()) == int(nodes["valid"].sum())
    assert int(branch_valid.sum()) <= PHYSICAL_BATCH_SIZE

    model = ProspectivePlannerMLX(config)
    outputs = model(
        mx.array(batch.context),
        mx.array(batch.branch_tokens),
        mx.array(batch.coordinates),
        mx.array(batch.attention_mask),
        mx.array(branch_valid),
    )
    mx.eval(outputs)
    for key, value in outputs.items():
        assert value.shape == branch_valid.shape, (key, value.shape)
        if key != BRANCH_VALID:
            assert value.dtype == mx.float16, (key, value.dtype)
            assert np.isfinite(np.asarray(value)).all(), key
    valid = branch_valid
    for key in (
        SCALAR_RETURN,
        SCALAR_VALUE,
        KO_PROBABILITY,
        EXPECTED_PRIZES,
        TERMINAL_PROBABILITY,
        UNCERTAINTY,
    ):
        values = np.asarray(outputs[key])
        assert np.all(values[~valid] == 0), key
    assert np.all(np.asarray(outputs[BRANCH_POLICY_SCORE])[~valid] == -65_504)
    assert np.all(np.asarray(outputs[UNCERTAINTY])[valid] > 0)

    real_returns: list[float] = []
    group_ids: list[int] = []
    current_logprobs: list[float] = []
    scores = np.asarray(outputs[BRANCH_POLICY_SCORE], dtype=np.float32)
    for group in range(len(batch.group_keys)):
        source_indices = batch.node_index[group]
        occupied = source_indices >= 0
        valid_slots = occupied & branch_valid[group]
        members = nodes[source_indices[valid_slots]]
        group_scores = scores[group, valid_slots]
        shifted = group_scores - group_scores.max()
        log_probabilities = shifted - np.log(np.exp(shifted).sum())
        current_logprobs.extend(log_probabilities.tolist())
        real_returns.extend(float(row["scalar_return"]) for row in members)
        group_ids.extend([group] * len(members))
        assert not any(bool(row["has_behavior_logprob"]) for row in members)
        assert not any(bool(row["has_reference_logprob"]) for row in members)

    advantages = group_relative_advantages(
        mx.array(real_returns, dtype=mx.float32),
        mx.array(group_ids, dtype=mx.int32),
    )
    objective = group_relative_objective(
        mx.array(current_logprobs, dtype=mx.float32),
        advantages,
    )
    mx.eval(
        advantages,
        objective.loss,
        objective.policy_loss,
        objective.kl_loss,
        objective.clip_fraction,
    )
    assert objective.mode == "group_relative_ranking_distillation"
    assert float(objective.kl_loss) == 0.0
    assert float(objective.clip_fraction) == 0.0
    assert np.isfinite(float(objective.loss))


def main() -> None:
    print("=== Lateral MLX prospective planner coordinate validation ===")
    test_real_episode_coordinates_and_rope_invariants()
    print("  PASS: real episode coordinates and RoPE-ND invariants")
    test_real_entity_zone_relation_round_trip()
    print("  PASS: real entity/zone/relation coordinates round-trip")
    test_isolated_mlx_parameter_contract()
    print("  PASS: isolated FP16 parameter contract")
    test_real_sidecar_forward_and_group_relative_mode()
    print(
        "  PASS: real prospective sidecar forward and honest "
        "ranking/distillation mode"
    )
    print("ALL PASSED")


if __name__ == "__main__":
    main()
