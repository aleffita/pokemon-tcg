"""Backend-neutral adapter from real prospective sidecars to planner tensors.

The lateral planner ultimately expects learned trunk/context and branch
representations.  Until that joint trunk interface is trained, this module
provides one explicit, versioned, deterministic projection of *real* dataset
and rollout-sidecar features into the shared ``d_model`` space.  It is an
integration adapter, not a substitute for the learned trunk embedding path.

No targets (returns, KO, prizes, terminal state, outcome, or behavior choice)
are copied into planner inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from rl.prospective_schema import (
    BRANCH_ACTION_AXIS,
    ENTITY_ZONE_RELATION_AXIS,
    MATCH_TIME_AXIS,
    N_PROSPECTIVE_AXES,
    PROSPECTIVE_COORD_SCHEMA_VERSION,
    ROLLOUT_DEPTH_AXIS,
    ProspectivePlannerConfig,
    build_tree_attention_mask,
    validate_prospective_coordinates,
)


PROSPECTIVE_INPUT_ADAPTER_VERSION = "real-sidecar-direct-v1"
_ENUM_RADIX = np.float32(512.0)
_ACTION_RADIX = np.float32(192.0)

# Public layouts make this deterministic adapter auditable across backends.
CONTEXT_FEATURE_LAYOUT = {
    "cls_scalars": slice(0, 19),
    "select_type": 19,
    "select_context": 20,
}
BRANCH_FEATURE_LAYOUT = {
    "depth": 0,
    "branch_order": 1,
    "sibling_index": 2,
    "action_index_plus_one": 3,
    "subselection_count": 4,
    "select_type": 5,
    "select_context": 6,
    "src_pos_plus_one": 7,
    "tgt_pos_plus_one": 8,
    "verb": 9,
    "has_action_index": 10,
}


@dataclass(frozen=True)
class ProspectivePlannerNumpyBatch:
    """Padded real sidecar groups ready for either MLX or PyTorch."""

    context: np.ndarray
    branch_tokens: np.ndarray
    coordinates: np.ndarray
    attention_mask: np.ndarray
    branch_valid: np.ndarray
    parent_index: np.ndarray
    node_index: np.ndarray
    group_keys: tuple[tuple[str, int, str], ...]
    adapter_version: str = PROSPECTIVE_INPUT_ADAPTER_VERSION

    def validate(self, config: ProspectivePlannerConfig) -> None:
        batch, context_length, hidden = self.context.shape
        if context_length != 1 or hidden != config.d_model:
            raise ValueError("context must have shape [B,1,d_model]")
        if self.branch_tokens.ndim != 3:
            raise ValueError("branch_tokens must have shape [B,T,d_model]")
        if self.branch_tokens.shape[0] != batch:
            raise ValueError("context and branch token batch sizes differ")
        if self.branch_tokens.shape[2] != config.d_model:
            raise ValueError("branch token width differs from d_model")
        branch_count = self.branch_tokens.shape[1]
        if self.branch_valid.shape != (batch, branch_count):
            raise ValueError("branch_valid must have shape [B,T]")
        if self.parent_index.shape != (batch, branch_count):
            raise ValueError("parent_index must have shape [B,T]")
        if self.node_index.shape != (batch, branch_count):
            raise ValueError("node_index must have shape [B,T]")
        expected_length = context_length + branch_count
        if self.coordinates.shape != (
            batch,
            expected_length,
            N_PROSPECTIVE_AXES,
        ):
            raise ValueError("coordinates do not cover context and branches")
        if self.attention_mask.shape != (
            batch,
            1,
            expected_length,
            expected_length,
        ):
            raise ValueError("attention mask has the wrong shape")
        if len(self.group_keys) != batch:
            raise ValueError("group_keys length differs from batch size")
        if self.context.dtype != np.float16:
            raise TypeError("context must be FP16")
        if self.branch_tokens.dtype != np.float16:
            raise TypeError("branch_tokens must be FP16")
        if self.coordinates.dtype != np.int32:
            raise TypeError("coordinates must be int32")
        if self.attention_mask.dtype != np.float32:
            raise TypeError("attention_mask must be FP32")
        if self.branch_valid.dtype != np.bool_:
            raise TypeError("branch_valid must be boolean")
        if not np.isfinite(self.context).all():
            raise ValueError("context contains non-finite values")
        if not np.isfinite(self.branch_tokens).all():
            raise ValueError("branch tokens contain non-finite values")
        validate_prospective_coordinates(self.coordinates)


def _load_required_array(dataset_dir: Path, name: str) -> np.ndarray:
    path = dataset_dir / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"required real dataset array is missing: {path}")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _read_manifest(sidecar_dir: Path) -> dict[str, Any]:
    path = sidecar_dir / "prospective_manifest.json"
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (
        manifest.get("prospective_coord_schema_version")
        != PROSPECTIVE_COORD_SCHEMA_VERSION
    ):
        raise ValueError("prospective sidecar coordinate schema is incompatible")
    if manifest.get("semantics", {}).get("synthetic_fill_allowed") is not False:
        raise ValueError("prospective sidecar must forbid synthetic fill")
    return manifest


def _metadata_row_index(metadata: np.ndarray) -> dict[tuple[str, int, int], int]:
    required = {"episode_id", "side", "step_id"}
    if metadata.dtype.names is None or not required <= set(metadata.dtype.names):
        raise ValueError("episode_meta lacks prospective join fields")
    result: dict[tuple[str, int, int], int] = {}
    for index, row in enumerate(metadata):
        key = (
            str(row["episode_id"]),
            int(row["side"]),
            int(row["step_id"]),
        )
        # Multi-select supervision expands one real decision into several
        # target rows with the same observation.  The sidecar joins to that
        # observation, so the first expanded row is the canonical source.
        result.setdefault(key, index)
    return result


def _group_node_indices(
    nodes: np.ndarray,
) -> list[tuple[tuple[str, int, str], list[int]]]:
    required = {
        "episode_id",
        "side",
        "step_id",
        "group_id",
        "branch_id",
        "parent_branch_id",
        "has_parent",
        "depth",
        "trial_index",
        "determination_id",
        "branch_order",
        "sibling_index",
        "action_index",
        "has_action_index",
        "subselection_count",
        "select_type",
        "select_context",
        "src_pos",
        "tgt_pos",
        "verb",
        "entity_zone_relation_id",
        "valid",
    }
    if nodes.dtype.names is None or not required <= set(nodes.dtype.names):
        missing = sorted(required - set(nodes.dtype.names or ()))
        raise ValueError(f"prospective nodes lack required fields: {missing}")
    grouped: dict[tuple[str, int, str], list[int]] = {}
    for index, node in enumerate(nodes):
        key = (
            str(node["group_id"]),
            int(node["trial_index"]),
            str(node["determination_id"]),
        )
        grouped.setdefault(key, []).append(index)
    return list(grouped.items())


def _encode_context(
    destination: np.ndarray,
    *,
    row_index: int,
    cls_scalars: np.ndarray,
    select_type: np.ndarray,
    select_context: np.ndarray,
) -> None:
    scalars = np.asarray(cls_scalars[row_index], dtype=np.float32)
    scalar_slice = CONTEXT_FEATURE_LAYOUT["cls_scalars"]
    if not isinstance(scalar_slice, slice):
        raise AssertionError("invalid context feature layout")
    if scalars.shape != (scalar_slice.stop - scalar_slice.start,):
        raise ValueError("cls_scalars width differs from adapter v1")
    destination[scalar_slice] = scalars
    destination[CONTEXT_FEATURE_LAYOUT["select_type"]] = (
        np.float32(select_type[row_index, 0]) / _ENUM_RADIX
    )
    destination[CONTEXT_FEATURE_LAYOUT["select_context"]] = (
        np.float32(select_context[row_index, 0]) / _ENUM_RADIX
    )


def _encode_branch(destination: np.ndarray, node: np.void) -> None:
    destination[BRANCH_FEATURE_LAYOUT["depth"]] = (
        np.float32(node["depth"]) / _ENUM_RADIX
    )
    destination[BRANCH_FEATURE_LAYOUT["branch_order"]] = (
        np.float32(node["branch_order"]) / _ENUM_RADIX
    )
    destination[BRANCH_FEATURE_LAYOUT["sibling_index"]] = (
        np.float32(node["sibling_index"]) / _ENUM_RADIX
    )
    destination[BRANCH_FEATURE_LAYOUT["action_index_plus_one"]] = (
        np.float32(int(node["action_index"]) + 1) / _ACTION_RADIX
    )
    destination[BRANCH_FEATURE_LAYOUT["subselection_count"]] = (
        np.float32(node["subselection_count"]) / _ACTION_RADIX
    )
    destination[BRANCH_FEATURE_LAYOUT["select_type"]] = (
        np.float32(node["select_type"]) / _ENUM_RADIX
    )
    destination[BRANCH_FEATURE_LAYOUT["select_context"]] = (
        np.float32(node["select_context"]) / _ENUM_RADIX
    )
    destination[BRANCH_FEATURE_LAYOUT["src_pos_plus_one"]] = (
        np.float32(int(node["src_pos"]) + 1) / _ENUM_RADIX
    )
    destination[BRANCH_FEATURE_LAYOUT["tgt_pos_plus_one"]] = (
        np.float32(int(node["tgt_pos"]) + 1) / _ENUM_RADIX
    )
    destination[BRANCH_FEATURE_LAYOUT["verb"]] = (
        np.float32(node["verb"]) / _ENUM_RADIX
    )
    destination[BRANCH_FEATURE_LAYOUT["has_action_index"]] = np.float32(
        node["has_action_index"]
    )


def load_real_prospective_planner_batch(
    dataset_dir: str | Path,
    *,
    sidecar_name: str = "prospective_v1",
    config: ProspectivePlannerConfig | None = None,
) -> ProspectivePlannerNumpyBatch:
    """Load actual replay context and actual simulated tree nodes.

    One planner batch row is one ``(group, trial, determinization)`` tree.
    Trees are padded only with invalid slots; no state, branch, target, or
    embedding is synthesized.
    """

    planner_config = config or ProspectivePlannerConfig()
    planner_config.validate()
    root = Path(dataset_dir)
    sidecar = root / sidecar_name
    manifest = _read_manifest(sidecar)
    nodes = np.load(
        sidecar / manifest["outputs"]["nodes"],
        mmap_mode="r",
        allow_pickle=False,
    )
    if len(nodes) != int(manifest["outputs"]["node_rows"]):
        raise ValueError("prospective sidecar node count disagrees with manifest")
    groups = _group_node_indices(nodes)
    if not groups:
        raise ValueError("prospective sidecar has no real rollout groups")

    episode_meta = _load_required_array(root, "episode_meta")
    cls_scalars = _load_required_array(root, "cls_scalars")
    select_type = _load_required_array(root, "select_type")
    select_context = _load_required_array(root, "select_context")
    decision_index = _metadata_row_index(episode_meta)

    batch_size = len(groups)
    branch_count = max(len(indices) for _, indices in groups)
    context = np.zeros(
        (batch_size, 1, planner_config.d_model), dtype=np.float32
    )
    branch_tokens = np.zeros(
        (batch_size, branch_count, planner_config.d_model), dtype=np.float32
    )
    coordinates = np.zeros(
        (batch_size, 1 + branch_count, N_PROSPECTIVE_AXES), dtype=np.int32
    )
    branch_valid = np.zeros((batch_size, branch_count), dtype=np.bool_)
    parent_index = np.full((batch_size, branch_count), -1, dtype=np.int32)
    node_index = np.full((batch_size, branch_count), -1, dtype=np.int64)

    for batch_index, (_, indices) in enumerate(groups):
        first = nodes[indices[0]]
        decision_key = (
            str(first["episode_id"]),
            int(first["side"]),
            int(first["step_id"]),
        )
        try:
            source_row = decision_index[decision_key]
        except KeyError as error:
            raise ValueError(
                f"prospective group has no real dataset decision {decision_key}"
            ) from error
        _encode_context(
            context[batch_index, 0],
            row_index=source_row,
            cls_scalars=cls_scalars,
            select_type=select_type,
            select_context=select_context,
        )
        coordinates[batch_index, 0, MATCH_TIME_AXIS] = decision_key[2]

        local_by_id: dict[str, int] = {}
        for local_index, source_index in enumerate(indices):
            node = nodes[source_index]
            if (
                str(node["episode_id"]),
                int(node["side"]),
                int(node["step_id"]),
            ) != decision_key:
                raise ValueError("one prospective group spans multiple decisions")
            branch_id = str(node["branch_id"])
            if branch_id in local_by_id:
                raise ValueError("prospective tree contains duplicate branch_id")
            local_by_id[branch_id] = local_index
            node_index[batch_index, local_index] = source_index
            _encode_branch(branch_tokens[batch_index, local_index], node)
            branch_valid[batch_index, local_index] = bool(node["valid"])
            coordinates[
                batch_index, 1 + local_index, MATCH_TIME_AXIS
            ] = decision_key[2]
            coordinates[
                batch_index, 1 + local_index, ROLLOUT_DEPTH_AXIS
            ] = int(node["depth"])
            coordinates[
                batch_index, 1 + local_index, BRANCH_ACTION_AXIS
            ] = int(node["branch_order"])
            coordinates[
                batch_index, 1 + local_index, ENTITY_ZONE_RELATION_AXIS
            ] = int(node["entity_zone_relation_id"])

            if bool(node["has_parent"]):
                parent_id = str(node["parent_branch_id"])
                if parent_id not in local_by_id:
                    raise ValueError(
                        "prospective nodes are not serialized parent-before-child"
                    )
                parent_index[batch_index, local_index] = local_by_id[parent_id]

    coordinates = validate_prospective_coordinates(coordinates)
    attention_mask = build_tree_attention_mask(
        parent_index,
        branch_valid,
        context_length=1,
    )
    batch = ProspectivePlannerNumpyBatch(
        context=context.astype(np.float16),
        branch_tokens=branch_tokens.astype(np.float16),
        coordinates=coordinates,
        attention_mask=attention_mask,
        branch_valid=branch_valid,
        parent_index=parent_index,
        node_index=node_index,
        group_keys=tuple(key for key, _ in groups),
    )
    batch.validate(planner_config)
    return batch


__all__ = [
    "BRANCH_FEATURE_LAYOUT",
    "CONTEXT_FEATURE_LAYOUT",
    "PROSPECTIVE_INPUT_ADAPTER_VERSION",
    "ProspectivePlannerNumpyBatch",
    "load_real_prospective_planner_batch",
]
