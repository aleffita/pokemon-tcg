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

from rl.encoder.enc_constants import MAX_OPTIONS, OPT_STRUCT
from rl.encoder.effect_data import N_ATTACK_FX
from rl.prospective_actions import PROSPECTIVE_ACTION_CANDIDATE_VERSION
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


PROSPECTIVE_INPUT_ADAPTER_VERSION = "real-sidecar-direct-v3"
ACTION_ATTR_AGGREGATE_VERSION = "opt-attr-mean-v1"
ACTION_SET_FEATURE_VERSION = "option-set-moments-fourier-v1"
BRANCH_FEATURE_LAYOUT_VERSION = 3
SUPPORTED_PROSPECTIVE_SIDECAR_SCHEMA_VERSION = 3
SUPPORTED_PROSPECTIVE_SIDECAR_PLANNER_VERSION = 2
ACTION_ATTR_WIDTH = OPT_STRUCT + N_ATTACK_FX
ACTION_SET_FEATURE_WIDTH = 20
ACTION_SET_MOMENT_ORDER = (
    "option_index_min", "option_index_mean", "option_index_max",
    "src_pos_min", "src_pos_mean", "src_pos_max",
    "tgt_pos_min", "tgt_pos_mean", "tgt_pos_max",
    "verb_min", "verb_mean", "verb_max",
)
ACTION_SET_FOURIER_FREQUENCIES = (1, 2, 4, 8)
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
    "action_attr_mean": slice(11, 11 + ACTION_ATTR_WIDTH),
    "action_set_features": slice(
        11 + ACTION_ATTR_WIDTH,
        11 + ACTION_ATTR_WIDTH + ACTION_SET_FEATURE_WIDTH,
    ),
}


def aggregate_action_opt_attr(
    opt_attr: np.ndarray | Any,
    action_indices: tuple[int, ...] | list[int],
) -> np.ndarray:
    """Mean-pool selected real option attributes into one causal action feature.

    The empty legal action maps to the all-zero vector. Indices must be unique;
    ordering therefore cannot change the representation of the same combined
    selection.
    """

    attributes = np.asarray(opt_attr, dtype=np.float32)
    if attributes.ndim != 2 or attributes.shape[1] != ACTION_ATTR_WIDTH:
        raise ValueError(
            "opt_attr must have shape [options,"
            f"{ACTION_ATTR_WIDTH}], got {attributes.shape}"
        )
    indices = tuple(int(index) for index in action_indices)
    if len(set(indices)) != len(indices):
        raise ValueError("combined action contains duplicate option indices")
    if any(index < 0 or index >= len(attributes) for index in indices):
        raise ValueError("combined action option index is outside opt_attr")
    if not indices:
        return np.zeros((ACTION_ATTR_WIDTH,), dtype=np.float32)
    pooled = np.mean(
        attributes[np.asarray(indices, dtype=np.int64)],
        axis=0,
        dtype=np.float32,
    )
    if pooled.shape != (ACTION_ATTR_WIDTH,) or not np.isfinite(pooled).all():
        raise ValueError("combined action aggregate is non-finite or malformed")
    return pooled.astype(np.float32, copy=False)


def aggregate_action_set_features(
    opt_src_pos: np.ndarray | Any,
    opt_tgt_pos: np.ndarray | Any,
    opt_verb: np.ndarray | Any,
    action_indices: tuple[int, ...] | list[int],
) -> np.ndarray:
    """Encode one unordered legal option set using transparent causal statistics."""

    source = np.asarray(opt_src_pos, dtype=np.int32).reshape(-1)
    target = np.asarray(opt_tgt_pos, dtype=np.int32).reshape(-1)
    verb = np.asarray(opt_verb, dtype=np.int32).reshape(-1)
    if not (len(source) == len(target) == len(verb)):
        raise ValueError("option relation arrays must have the same length")
    indices = tuple(int(index) for index in action_indices)
    if len(set(indices)) != len(indices):
        raise ValueError("combined action contains duplicate option indices")
    if any(index < 0 or index >= len(source) for index in indices):
        raise ValueError("combined action option index is outside relation arrays")
    if not indices:
        return np.zeros((ACTION_SET_FEATURE_WIDTH,), dtype=np.float32)

    selected = np.asarray(indices, dtype=np.int64)
    components = (
        (selected.astype(np.float32) + 1.0) / np.float32(MAX_OPTIONS + 1),
        (source[selected].astype(np.float32) + 1.0) / np.float32(512.0),
        (target[selected].astype(np.float32) + 1.0) / np.float32(512.0),
        (verb[selected].astype(np.float32) + 1.0) / np.float32(64.0),
    )
    moments: list[np.float32] = []
    for component in components:
        moments.extend((
            np.float32(np.min(component)),
            np.float32(np.mean(component, dtype=np.float32)),
            np.float32(np.max(component)),
        ))
    phases = (
        np.float32(2.0 * np.pi)
        * (selected.astype(np.float32) + 1.0)
        / np.float32(MAX_OPTIONS + 1)
    )
    fourier: list[np.float32] = []
    for frequency in ACTION_SET_FOURIER_FREQUENCIES:
        angle = np.float32(frequency) * phases
        fourier.extend((
            np.float32(np.mean(np.sin(angle), dtype=np.float32)),
            np.float32(np.mean(np.cos(angle), dtype=np.float32)),
        ))
    result = np.asarray([*moments, *fourier], dtype=np.float32)
    if (
        result.shape != (ACTION_SET_FEATURE_WIDTH,)
        or not np.isfinite(result).all()
    ):
        raise ValueError("combined action set feature is non-finite or malformed")
    return result


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
        if context_length <= 0 or hidden != config.d_model:
            raise ValueError("context must have shape [B,C,d_model] with C > 0")
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


@dataclass(frozen=True)
class RealProspectivePlannerIndex:
    """Memory-mapped compact sidecar catalog; no padded planner tensors."""

    dataset_dir: Path
    sidecar_dir: Path
    manifest: dict[str, Any]
    nodes: np.ndarray
    actions: np.ndarray
    groups: np.ndarray
    group_offsets: np.ndarray
    episode_sides: np.ndarray
    episode_meta: np.ndarray
    cls_scalars: np.ndarray
    select_type: np.ndarray
    select_context: np.ndarray
    group_target_keys: tuple[tuple[str, int, int], ...]
    group_target_rows: np.ndarray
    node_starts: np.ndarray
    node_counts: np.ndarray

    def __len__(self) -> int:
        return len(self.groups)


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
        manifest.get("schema_version")
        != SUPPORTED_PROSPECTIVE_SIDECAR_SCHEMA_VERSION
        or manifest.get("planner_version")
        != SUPPORTED_PROSPECTIVE_SIDECAR_PLANNER_VERSION
        or manifest.get("action_candidate_version")
        != PROSPECTIVE_ACTION_CANDIDATE_VERSION
    ):
        raise ValueError(
            "prospective sidecar action-candidate contract is incompatible"
        )
    if (
        manifest.get("prospective_coord_schema_version")
        != PROSPECTIVE_COORD_SCHEMA_VERSION
    ):
        raise ValueError("prospective sidecar coordinate schema is incompatible")
    if (
        manifest.get("input_adapter_version")
        != PROSPECTIVE_INPUT_ADAPTER_VERSION
    ):
        raise ValueError("prospective sidecar input adapter is incompatible")
    if manifest.get("semantics", {}).get("synthetic_fill_allowed") is not False:
        raise ValueError("prospective sidecar must forbid synthetic fill")
    if manifest.get("storage", {}).get("version") != "compact-sharded-v1":
        raise ValueError("prospective sidecar compact storage is incompatible")
    action_schema = manifest.get("action_feature_schema") or {}
    if (
        action_schema.get("aggregate_version") != ACTION_ATTR_AGGREGATE_VERSION
        or action_schema.get("branch_feature_layout_version")
        != BRANCH_FEATURE_LAYOUT_VERSION
        or action_schema.get("field") != "action_attr_mean"
        or action_schema.get("shape") != [ACTION_ATTR_WIDTH]
        or action_schema.get("action_set_feature_version")
        != ACTION_SET_FEATURE_VERSION
        or action_schema.get("action_set_field") != "action_set_features"
        or action_schema.get("action_set_shape") != [ACTION_SET_FEATURE_WIDTH]
    ):
        raise ValueError("prospective sidecar action feature schema is incompatible")
    return manifest


def decode_prospective_action(
    actions: np.ndarray | Any,
    row: Any,
) -> tuple[int, ...]:
    """Decode one compact action without exposing offset arithmetic to callers."""

    flat = np.asarray(actions)
    if flat.ndim != 1 or flat.dtype != np.dtype("i2"):
        raise ValueError("prospective actions must be one-dimensional int16")
    offset = int(row["action_offset"])
    count = int(row["action_count"])
    if offset < 0 or count < 0 or offset + count > len(flat):
        raise ValueError("prospective action offset/count is outside flat storage")
    result = tuple(int(value) for value in flat[offset:offset + count])
    if len(set(result)) != len(result):
        raise ValueError("prospective action contains duplicate option indices")
    if any(value < 0 or value >= MAX_OPTIONS for value in result):
        raise ValueError("prospective action contains an invalid option index")
    return result


def encode_prospective_context_features(
    cls_scalars: np.ndarray | Any,
    select_type: int,
    select_context: int,
    *,
    d_model: int,
) -> np.ndarray:
    """Project one real encoded observation into the v1 context feature space."""

    destination = np.zeros((d_model,), dtype=np.float32)
    scalars = np.asarray(cls_scalars, dtype=np.float32).reshape(-1)
    scalar_slice = CONTEXT_FEATURE_LAYOUT["cls_scalars"]
    if not isinstance(scalar_slice, slice):
        raise AssertionError("invalid context feature layout")
    if scalars.shape != (scalar_slice.stop - scalar_slice.start,):
        raise ValueError("cls_scalars width differs from adapter v1")
    destination[scalar_slice] = scalars
    destination[CONTEXT_FEATURE_LAYOUT["select_type"]] = (
        np.float32(select_type) / _ENUM_RADIX
    )
    destination[CONTEXT_FEATURE_LAYOUT["select_context"]] = (
        np.float32(select_context) / _ENUM_RADIX
    )
    return destination.astype(np.float16)


def encode_prospective_branch_features(
    node: Any,
    *,
    d_model: int,
) -> np.ndarray:
    """Project one real simulated node without copying any prediction target."""

    destination = np.zeros((d_model,), dtype=np.float32)
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
    action_slice = BRANCH_FEATURE_LAYOUT["action_attr_mean"]
    if not isinstance(action_slice, slice):
        raise AssertionError("invalid combined-action feature layout")
    action_attr = np.asarray(node["action_attr_mean"], dtype=np.float32)
    if action_attr.shape != (ACTION_ATTR_WIDTH,):
        raise ValueError(
            f"action_attr_mean must have shape [{ACTION_ATTR_WIDTH}]"
        )
    if not np.isfinite(action_attr).all():
        raise ValueError("action_attr_mean contains NaN or infinity")
    destination[action_slice] = action_attr
    action_set_slice = BRANCH_FEATURE_LAYOUT["action_set_features"]
    if not isinstance(action_set_slice, slice):
        raise AssertionError("invalid combined-action set feature layout")
    action_set = np.asarray(node["action_set_features"], dtype=np.float32)
    if action_set.shape != (ACTION_SET_FEATURE_WIDTH,):
        raise ValueError(
            f"action_set_features must have shape [{ACTION_SET_FEATURE_WIDTH}]"
        )
    if not np.isfinite(action_set).all():
        raise ValueError("action_set_features contains NaN or infinity")
    destination[action_set_slice] = action_set
    return destination.astype(np.float16)


def load_real_prospective_planner_index(
    dataset_dir: str | Path,
    *,
    sidecar_name: str = "prospective_v2",
) -> RealProspectivePlannerIndex:
    """Open compact real sidecar arrays and validate every target join."""

    root = Path(dataset_dir)
    sidecar = root / sidecar_name
    manifest = _read_manifest(sidecar)
    dataset_manifest_path = root / "dataset_manifest.json"
    if not dataset_manifest_path.is_file():
        raise FileNotFoundError(
            f"prospective target joins require {dataset_manifest_path}"
        )
    with dataset_manifest_path.open(encoding="utf-8") as stream:
        dataset_manifest = json.load(stream)
    if (
        manifest.get("source", {}).get("bc_dataset_build_fingerprint")
        != dataset_manifest.get("build_fingerprint")
    ):
        raise ValueError(
            "prospective sidecar was built for a different BC corpus"
        )
    outputs = manifest["outputs"]

    def open_output(key: str) -> np.ndarray:
        filename = outputs.get(key)
        if not isinstance(filename, str):
            raise ValueError(f"prospective manifest lacks output {key}")
        return np.load(sidecar / filename, mmap_mode="r", allow_pickle=False)

    nodes = open_output("nodes")
    actions = open_output("actions")
    groups = open_output("groups")
    group_offsets = open_output("group_offsets")
    episode_sides = open_output("episode_sides")
    if len(nodes) != int(outputs["node_rows"]):
        raise ValueError("prospective sidecar node count disagrees with manifest")
    if len(actions) != int(outputs["action_rows"]):
        raise ValueError("prospective sidecar action count disagrees with manifest")
    if len(groups) != int(outputs["group_rows"]):
        raise ValueError("prospective sidecar group count disagrees with manifest")
    if len(episode_sides) != int(outputs["episode_side_rows"]):
        raise ValueError("prospective episode-side count disagrees with manifest")
    if len(group_offsets) != len(groups) + 1:
        raise ValueError("prospective group offsets have the wrong length")
    if not len(groups):
        raise ValueError("prospective sidecar has no real rollout groups")

    episode_meta = _load_required_array(root, "episode_meta")
    cls_scalars = _load_required_array(root, "cls_scalars")
    select_type = _load_required_array(root, "select_type")
    select_context = _load_required_array(root, "select_context")
    if not (
        len(episode_meta) == len(cls_scalars)
        == len(select_type) == len(select_context)
    ):
        raise ValueError("prospective source BC arrays have different row counts")
    required_groups = {
        "episode_side_index", "episode_key", "side", "step_id", "group_id",
        "trial_index", "determination_id", "node_start", "node_count",
        "target_row",
    }
    if groups.dtype.names is None or not required_groups <= set(groups.dtype.names):
        raise ValueError("prospective groups use an incompatible compact dtype")
    required_nodes = {
        "parent_index", "depth", "branch_order", "sibling_index",
        "action_index", "has_action_index", "action_offset", "action_count",
        "action_attr_mean", "action_set_features", "subselection_count",
        "select_type", "select_context", "src_pos", "tgt_pos", "verb",
        "entity_zone_relation_id", "valid",
    }
    if nodes.dtype.names is None or not required_nodes <= set(nodes.dtype.names):
        raise ValueError("prospective nodes use an incompatible compact dtype")
    required_episode_sides = {"episode_id", "episode_key", "side"}
    if (
        episode_sides.dtype.names is None
        or not required_episode_sides <= set(episode_sides.dtype.names)
    ):
        raise ValueError("prospective episode-side table is incompatible")
    if actions.dtype != np.dtype("i2") or actions.ndim != 1:
        raise ValueError("prospective flat actions must be int16")

    target_rows = np.asarray(groups["target_row"], dtype=np.int64)
    node_starts = np.asarray(groups["node_start"], dtype=np.uint64)
    node_counts = np.asarray(groups["node_count"], dtype=np.uint32)
    expected_offsets = np.empty((len(groups) + 1,), dtype=np.uint64)
    expected_offsets[:-1] = node_starts
    expected_offsets[-1] = np.uint64(len(nodes))
    if not np.array_equal(group_offsets, expected_offsets):
        raise ValueError("prospective group offsets disagree with group rows")
    if np.any(node_starts + node_counts > len(nodes)):
        raise ValueError("prospective group node slice is out of bounds")
    if len(groups) > 1 and np.any(
        node_starts[1:] != node_starts[:-1] + node_counts[:-1]
    ):
        raise ValueError("prospective group node slices are not contiguous")
    if np.any(target_rows < 0) or np.any(target_rows >= len(episode_meta)):
        raise ValueError("prospective group target_row is outside episode_meta")

    group_target_keys: list[tuple[str, int, int]] = []
    seen_ids: dict[tuple[str, int], tuple[Any, ...]] = {}
    for group_index, group in enumerate(groups):
        side_index = int(group["episode_side_index"])
        if side_index < 0 or side_index >= len(episode_sides):
            raise ValueError("prospective episode_side_index is out of bounds")
        episode_side = episode_sides[side_index]
        if (
            int(group["episode_key"]) != int(episode_side["episode_key"])
            or int(group["side"]) != int(episode_side["side"])
        ):
            raise ValueError("prospective group/episode-side identity mismatch")
        key = (
            str(episode_side["episode_id"]),
            int(group["side"]),
            int(group["step_id"]),
        )
        source = episode_meta[int(target_rows[group_index])]
        source_key = (
            str(source["episode_id"]),
            int(source["side"]),
            int(source["step_id"]),
        )
        if key != source_key:
            raise ValueError(
                f"prospective target_row does not join to group key: {key}"
            )
        group_target_keys.append(key)
        for namespace, identifier, payload in (
            ("episode", int(group["episode_key"]), (key[0],)),
            (
                "group",
                int(group["group_id"]),
                (key[0], key[1], key[2]),
            ),
            (
                "determination",
                int(group["determination_id"]),
                (int(group["group_id"]), int(group["trial_index"])),
            ),
        ):
            collision_key = (namespace, identifier)
            previous = seen_ids.setdefault(collision_key, payload)
            if previous != payload:
                raise ValueError(
                    f"prospective {namespace} uint64 collision in sidecar"
                )

        start = int(node_starts[group_index])
        stop = start + int(node_counts[group_index])
        for local_index, node in enumerate(nodes[start:stop]):
            parent = int(node["parent_index"])
            if parent >= local_index or parent < -1:
                raise ValueError(
                    "prospective parent_index must reference an earlier local node"
                )
            decode_prospective_action(actions, node)

    return RealProspectivePlannerIndex(
        dataset_dir=root,
        sidecar_dir=sidecar,
        manifest=manifest,
        nodes=nodes,
        actions=actions,
        groups=groups,
        group_offsets=group_offsets,
        episode_sides=episode_sides,
        episode_meta=episode_meta,
        cls_scalars=cls_scalars,
        select_type=select_type,
        select_context=select_context,
        group_target_keys=tuple(group_target_keys),
        group_target_rows=target_rows,
        node_starts=node_starts,
        node_counts=node_counts,
    )


def materialize_real_prospective_planner_batch(
    index: RealProspectivePlannerIndex,
    group_indices: np.ndarray | list[int] | tuple[int, ...],
    *,
    config: ProspectivePlannerConfig | None = None,
) -> ProspectivePlannerNumpyBatch:
    """Densify only an explicitly requested group slice."""

    planner_config = config or ProspectivePlannerConfig()
    planner_config.validate()
    selected = np.asarray(group_indices, dtype=np.int64).reshape(-1)
    if not len(selected):
        raise ValueError("at least one prospective group index is required")
    if np.any(selected < 0) or np.any(selected >= len(index)):
        raise IndexError("prospective group index is out of bounds")
    if len(set(int(value) for value in selected)) != len(selected):
        raise ValueError("prospective group indices must be unique")

    batch_size = len(selected)
    branch_count = max(int(index.node_counts[value]) for value in selected)
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

    for batch_index, group_index in enumerate(selected):
        group_index = int(group_index)
        decision_key = index.group_target_keys[group_index]
        source_row = int(index.group_target_rows[group_index])
        context[batch_index, 0] = encode_prospective_context_features(
            index.cls_scalars[source_row],
            int(index.select_type[source_row, 0]),
            int(index.select_context[source_row, 0]),
            d_model=planner_config.d_model,
        )
        coordinates[batch_index, 0, MATCH_TIME_AXIS] = decision_key[2]

        start = int(index.node_starts[group_index])
        stop = start + int(index.node_counts[group_index])
        for local_index, source_index in enumerate(range(start, stop)):
            node = index.nodes[source_index]
            node_index[batch_index, local_index] = source_index
            branch_tokens[batch_index, local_index] = (
                encode_prospective_branch_features(
                    node,
                    d_model=planner_config.d_model,
                )
            )
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

            parent_index[batch_index, local_index] = int(node["parent_index"])

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
        group_keys=tuple(
            (
                str(int(index.groups[value]["group_id"])),
                int(index.groups[value]["trial_index"]),
                str(int(index.groups[value]["determination_id"])),
            )
            for value in selected
        ),
    )
    batch.validate(planner_config)
    return batch


def load_real_prospective_planner_batch(
    dataset_dir: str | Path,
    *,
    sidecar_name: str = "prospective_v2",
    config: ProspectivePlannerConfig | None = None,
) -> ProspectivePlannerNumpyBatch:
    """Small-test compatibility wrapper that materializes every sidecar group."""

    index = load_real_prospective_planner_index(
        dataset_dir,
        sidecar_name=sidecar_name,
    )
    return materialize_real_prospective_planner_batch(
        index,
        np.arange(len(index), dtype=np.int64),
        config=config,
    )


__all__ = [
    "ACTION_ATTR_AGGREGATE_VERSION",
    "ACTION_ATTR_WIDTH",
    "ACTION_SET_FEATURE_VERSION",
    "ACTION_SET_FEATURE_WIDTH",
    "ACTION_SET_FOURIER_FREQUENCIES",
    "ACTION_SET_MOMENT_ORDER",
    "BRANCH_FEATURE_LAYOUT",
    "BRANCH_FEATURE_LAYOUT_VERSION",
    "CONTEXT_FEATURE_LAYOUT",
    "PROSPECTIVE_INPUT_ADAPTER_VERSION",
    "ProspectivePlannerNumpyBatch",
    "RealProspectivePlannerIndex",
    "aggregate_action_opt_attr",
    "aggregate_action_set_features",
    "decode_prospective_action",
    "encode_prospective_branch_features",
    "encode_prospective_context_features",
    "load_real_prospective_planner_batch",
    "load_real_prospective_planner_index",
    "materialize_real_prospective_planner_batch",
]
