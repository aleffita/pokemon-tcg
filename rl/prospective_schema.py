"""Shared, versioned contracts for the lateral prospective planner.

This module is deliberately backend-neutral.  It defines the coordinate order
used by RoPE-ND, the reversible entity/zone/relation encoding, and the tree
causal mask consumed by both MLX training and PyTorch inference.

The prospective planner is a side module: none of these helpers alter the
current TokenTransformer or its TBPTT memory contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


PROSPECTIVE_COORD_SCHEMA_VERSION = "1.0.0"
PROSPECTIVE_PLANNER_VERSION = "1.0.0"

MATCH_TIME_AXIS = 0
ROLLOUT_DEPTH_AXIS = 1
BRANCH_ACTION_AXIS = 2
ENTITY_ZONE_RELATION_AXIS = 3
N_PROSPECTIVE_AXES = 4

PROSPECTIVE_AXIS_NAMES = (
    "match_time",
    "rollout_depth",
    "branch_action",
    "entity_zone_relation",
)

BRANCH_POLICY_SCORE = "branch_policy_score"
SCALAR_RETURN = "scalar_return"
SCALAR_VALUE = "scalar_value"
KO_LOGIT = "ko_logit"
KO_PROBABILITY = "ko_probability"
PRIZE_LOGIT = "prize_logit"
EXPECTED_PRIZES = "expected_prizes"
TERMINAL_LOGIT = "terminal_logit"
TERMINAL_PROBABILITY = "terminal_probability"
UNCERTAINTY_LOGIT = "uncertainty_logit"
UNCERTAINTY = "uncertainty"
BRANCH_VALID = "branch_valid"

# Version 1 packs the encoded source position, target position, and option verb
# into a compact integer that remains exactly representable as FP32 throughout
# the current encoder domain.  -1 is the encoder's explicit "no position".
ENTITY_ZONE_RELATION_RADIX = 512
MIN_ENCODED_POSITION = -1
MAX_ENCODED_POSITION = ENTITY_ZONE_RELATION_RADIX - 2
MIN_ENCODED_VERB = 0
# 64 relation values fill the remaining high-order six bits.  Keeping this
# bound at 63 guarantees the complete packed v1 domain is < 2**24 and thus
# exactly representable when RoPE converts coordinates to FP32.
MAX_ENCODED_VERB = 63
MAX_EXACT_FP32_INTEGER = 1 << 24


def prospective_coordinate_schema() -> dict[str, Any]:
    """Return the JSON-serializable public coordinate/mask contract."""

    return {
        "version": PROSPECTIVE_COORD_SCHEMA_VERSION,
        "dtype": "int32",
        "shape_suffix": [N_PROSPECTIVE_AXES],
        "axes": [
            {
                "index": MATCH_TIME_AXIS,
                "name": "match_time",
                "semantics": "real replay step_id; never rollout-relative",
            },
            {
                "index": ROLLOUT_DEPTH_AXIS,
                "name": "rollout_depth",
                "semantics": "context=0; first simulated action=1",
            },
            {
                "index": BRANCH_ACTION_AXIS,
                "name": "branch_action",
                "semantics": "stable sibling/action index within its parent",
            },
            {
                "index": ENTITY_ZONE_RELATION_AXIS,
                "name": "entity_zone_relation",
                "semantics": "version-1 reversible src_pos/tgt_pos/verb coordinate",
            },
        ],
        "entity_zone_relation": {
            "radix": ENTITY_ZONE_RELATION_RADIX,
            "formula": "((verb * 512) + (src_pos + 1)) * 512 + (tgt_pos + 1)",
            "src_pos_range": [MIN_ENCODED_POSITION, MAX_ENCODED_POSITION],
            "tgt_pos_range": [MIN_ENCODED_POSITION, MAX_ENCODED_POSITION],
            "verb_range": [MIN_ENCODED_VERB, MAX_ENCODED_VERB],
        },
        "attention_mask": {
            "dtype": "float32",
            "allowed": 0.0,
            "blocked": "-inf",
            "tree_contract": (
                "context queries attend context; branch queries attend context, "
                "self, and strictly causal ancestors"
            ),
        },
    }


@dataclass(frozen=True)
class ProspectivePlannerConfig:
    """Serializable cross-backend architecture contract."""

    version: str = PROSPECTIVE_PLANNER_VERSION
    coord_schema_version: str = PROSPECTIVE_COORD_SCHEMA_VERSION
    d_model: int = 128
    nhead: int = 4
    nlayers: int = 2
    ff_dim: int = 512
    rope_base: float = 10_000.0
    uncertainty_floor: float = 1e-6
    max_prizes: float = 6.0
    parameter_dtype: str = "float16"
    reduction_dtype: str = "float32"

    def validate(self) -> None:
        if self.version != PROSPECTIVE_PLANNER_VERSION:
            raise ValueError(
                f"unsupported prospective planner version: {self.version!r}"
            )
        if self.coord_schema_version != PROSPECTIVE_COORD_SCHEMA_VERSION:
            raise ValueError(
                "unsupported prospective coordinate schema: "
                f"{self.coord_schema_version!r}"
            )
        if self.d_model <= 0 or self.nhead <= 0:
            raise ValueError("d_model and nhead must be positive")
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        head_dim = self.d_model // self.nhead
        if head_dim % (2 * N_PROSPECTIVE_AXES) != 0:
            raise ValueError(
                "RoPE-ND requires head_dim divisible by "
                f"{2 * N_PROSPECTIVE_AXES}, got {head_dim}"
            )
        if self.nlayers <= 0 or self.ff_dim <= 0:
            raise ValueError("nlayers and ff_dim must be positive")
        if self.rope_base <= 1.0:
            raise ValueError("rope_base must be greater than one")
        if self.uncertainty_floor <= 0.0:
            raise ValueError("uncertainty_floor must be positive")
        if self.max_prizes <= 0.0:
            raise ValueError("max_prizes must be positive")
        if self.parameter_dtype != "float16":
            raise ValueError("prospective planner parameters must be float16")
        if self.reduction_dtype != "float32":
            raise ValueError("prospective planner reductions must be float32")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def _as_integral_array(value: np.ndarray | Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype, got {array.dtype}")
    return array.astype(np.int64, copy=False)


def encode_entity_zone_relation(
    src_pos: np.ndarray | Any,
    tgt_pos: np.ndarray | Any,
    verb: np.ndarray | Any,
) -> np.ndarray:
    """Pack encoder source position, target position, and verb without collisions."""

    source = _as_integral_array(src_pos, name="src_pos")
    target = _as_integral_array(tgt_pos, name="tgt_pos")
    relation = _as_integral_array(verb, name="verb")
    source, target, relation = np.broadcast_arrays(source, target, relation)
    if np.any((source < MIN_ENCODED_POSITION) | (source > MAX_ENCODED_POSITION)):
        raise ValueError(
            f"src_pos must be in [{MIN_ENCODED_POSITION}, {MAX_ENCODED_POSITION}]"
        )
    if np.any((target < MIN_ENCODED_POSITION) | (target > MAX_ENCODED_POSITION)):
        raise ValueError(
            f"tgt_pos must be in [{MIN_ENCODED_POSITION}, {MAX_ENCODED_POSITION}]"
        )
    if np.any((relation < MIN_ENCODED_VERB) | (relation > MAX_ENCODED_VERB)):
        raise ValueError(
            f"verb must be in [{MIN_ENCODED_VERB}, {MAX_ENCODED_VERB}]"
        )
    packed = (
        (relation * ENTITY_ZONE_RELATION_RADIX + source + 1)
        * ENTITY_ZONE_RELATION_RADIX
        + target
        + 1
    )
    if np.any(packed >= MAX_EXACT_FP32_INTEGER):
        raise ValueError(
            "entity/zone/relation coordinate is not exactly representable as FP32"
        )
    return packed.astype(np.int32)


# Short public alias retained for backend implementations and serialized schema
# consumers that use "pack" terminology.
pack_entity_zone_relation = encode_entity_zone_relation


def decode_entity_zone_relation(
    coordinate: np.ndarray | Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`encode_entity_zone_relation` for schema audits."""

    packed = _as_integral_array(coordinate, name="coordinate")
    if np.any((packed < 0) | (packed >= MAX_EXACT_FP32_INTEGER)):
        raise ValueError("coordinate is outside the version-1 encoding domain")
    target = packed % ENTITY_ZONE_RELATION_RADIX - 1
    prefix = packed // ENTITY_ZONE_RELATION_RADIX
    source = prefix % ENTITY_ZONE_RELATION_RADIX - 1
    relation = prefix // ENTITY_ZONE_RELATION_RADIX
    if np.any(relation > MAX_ENCODED_VERB):
        raise ValueError("coordinate contains an invalid verb")
    return (
        source.astype(np.int32),
        target.astype(np.int32),
        relation.astype(np.int32),
    )


def coordinates_from_episode_meta(episode_meta: np.ndarray) -> np.ndarray:
    """Extract known root coordinates from real replay metadata.

    Replay metadata only establishes real match time.  Rollout depth, branch
    action, and entity/zone/relation remain zero until a prospective sidecar
    supplies actual simulated branches; they are never fabricated here.
    """

    metadata = np.asarray(episode_meta)
    if metadata.dtype.names is None or "step_id" not in metadata.dtype.names:
        raise ValueError("episode_meta must be structured and contain step_id")
    match_time = _as_integral_array(metadata["step_id"], name="step_id")
    if np.any(match_time < 0):
        raise ValueError("episode_meta.step_id must be non-negative")
    coordinates = np.zeros((len(metadata), N_PROSPECTIVE_AXES), dtype=np.int32)
    coordinates[:, MATCH_TIME_AXIS] = match_time.astype(np.int32)
    return coordinates


def validate_prospective_coordinates(coordinates: np.ndarray | Any) -> np.ndarray:
    """Validate and normalize a ``[..., 4]`` RoPE-ND coordinate tensor."""

    values = _as_integral_array(coordinates, name="coordinates")
    if values.ndim < 2 or values.shape[-1] != N_PROSPECTIVE_AXES:
        raise ValueError(
            "prospective coordinates must have shape [..., "
            f"{N_PROSPECTIVE_AXES}], got {values.shape}"
        )
    if np.any(values < 0):
        raise ValueError("prospective coordinates must be non-negative")
    if np.any(values > np.iinfo(np.int32).max):
        raise ValueError("prospective coordinates exceed int32")
    return values.astype(np.int32)


def build_tree_causal_attention_mask(
    parent_index: np.ndarray | Any,
    branch_valid: np.ndarray | Any,
    *,
    context_length: int,
    context_valid: np.ndarray | Any | None = None,
) -> np.ndarray:
    """Build an additive tree/causal attention mask.

    Context queries attend valid context keys only.  Each valid branch query
    attends all valid context keys plus itself and its causal ancestor chain.
    Branch parents are local branch indices and must precede their children.
    Invalid query rows retain a self edge solely to keep softmax finite; their
    outputs must still be removed with ``branch_valid``.
    """

    parents = _as_integral_array(parent_index, name="parent_index")
    valid = np.asarray(branch_valid, dtype=np.bool_)
    if parents.ndim != 2 or valid.shape != parents.shape:
        raise ValueError("parent_index and branch_valid must have equal [B,T] shape")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    batch_size, branch_count = parents.shape
    if context_valid is None:
        valid_context = np.ones((batch_size, context_length), dtype=np.bool_)
    else:
        valid_context = np.asarray(context_valid, dtype=np.bool_)
        if valid_context.shape != (batch_size, context_length):
            raise ValueError(
                "context_valid must have shape "
                f"{(batch_size, context_length)}, got {valid_context.shape}"
            )
    total_length = context_length + branch_count
    allowed = np.zeros(
        (batch_size, 1, total_length, total_length), dtype=np.bool_
    )

    for batch in range(batch_size):
        context_keys = np.flatnonzero(valid_context[batch])
        for query in range(context_length):
            if valid_context[batch, query]:
                allowed[batch, 0, query, context_keys] = True
            else:
                allowed[batch, 0, query, query] = True
        for branch in range(branch_count):
            query = context_length + branch
            if not valid[batch, branch]:
                allowed[batch, 0, query, query] = True
                continue
            allowed[batch, 0, query, context_keys] = True
            ancestor = branch
            visited: set[int] = set()
            while ancestor >= 0:
                if ancestor in visited:
                    raise ValueError("branch parent graph contains a cycle")
                if ancestor >= branch_count:
                    raise ValueError("branch parent index is out of range")
                if not valid[batch, ancestor]:
                    raise ValueError("valid branch cannot descend from an invalid parent")
                if ancestor > branch:
                    raise ValueError("branch parent must not point into the future")
                visited.add(ancestor)
                allowed[batch, 0, query, context_length + ancestor] = True
                parent = int(parents[batch, ancestor])
                if parent >= ancestor:
                    raise ValueError("branch parent must strictly precede its child")
                ancestor = parent

    return np.where(allowed, 0.0, -np.inf).astype(np.float32)


build_tree_attention_mask = build_tree_causal_attention_mask


def prospective_checkpoint_metadata(
    config: ProspectivePlannerConfig,
) -> dict[str, Any]:
    """Return the required metadata for a future lateral planner checkpoint."""

    return {
        "planner_version": PROSPECTIVE_PLANNER_VERSION,
        "coord_schema_version": PROSPECTIVE_COORD_SCHEMA_VERSION,
        "config": config.to_dict(),
    }


__all__ = [
    "BRANCH_ACTION_AXIS",
    "BRANCH_POLICY_SCORE",
    "BRANCH_VALID",
    "ENTITY_ZONE_RELATION_AXIS",
    "EXPECTED_PRIZES",
    "KO_LOGIT",
    "KO_PROBABILITY",
    "MATCH_TIME_AXIS",
    "N_PROSPECTIVE_AXES",
    "PRIZE_LOGIT",
    "PROSPECTIVE_AXIS_NAMES",
    "PROSPECTIVE_COORD_SCHEMA_VERSION",
    "PROSPECTIVE_PLANNER_VERSION",
    "ProspectivePlannerConfig",
    "ROLLOUT_DEPTH_AXIS",
    "SCALAR_RETURN",
    "SCALAR_VALUE",
    "TERMINAL_LOGIT",
    "TERMINAL_PROBABILITY",
    "UNCERTAINTY",
    "UNCERTAINTY_LOGIT",
    "build_tree_attention_mask",
    "build_tree_causal_attention_mask",
    "coordinates_from_episode_meta",
    "decode_entity_zone_relation",
    "encode_entity_zone_relation",
    "pack_entity_zone_relation",
    "prospective_checkpoint_metadata",
    "prospective_coordinate_schema",
    "validate_prospective_coordinates",
]
