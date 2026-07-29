"""Real player-view prospective search and optional PyTorch reranking.

The runtime uses the same official ``cg.api`` search path and shared
observation-conditioned determinization as the offline sidecar builder.  It
never reads a replay's hidden deck and rejects any determinization that needed
synthetic cards.

This module is additive.  Callers may fall back to the existing
TokenTransformer/TBPTT policy whenever search cannot produce a complete real
tree.
"""
from __future__ import annotations

from dataclasses import dataclass
import dataclasses
import itertools
import random
from typing import Any

import numpy as np
import torch

from rl.encoder.encoding import MAX_OPTIONS, TokenEncoder
from rl.prospective_input_adapter import (
    ProspectivePlannerNumpyBatch,
    aggregate_action_opt_attr,
    aggregate_action_set_features,
    encode_prospective_branch_features,
)
from rl.prospective_planner_torch import ProspectivePlannerTorch
from rl.prospective_schema import (
    BRANCH_ACTION_AXIS,
    BRANCH_POLICY_SCORE,
    ENTITY_ZONE_RELATION_AXIS,
    MATCH_TIME_AXIS,
    N_PROSPECTIVE_AXES,
    ROLLOUT_DEPTH_AXIS,
    ProspectivePlannerConfig,
    build_tree_attention_mask,
    pack_entity_zone_relation,
    validate_prospective_coordinates,
)
from rl.search_agent import _determinize


@dataclass(frozen=True)
class ProspectiveRuntimeConfig:
    """Checkpoint-owned runtime tree budget."""

    trials: int
    horizon: int
    max_branches: int
    gamma: float
    seed: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProspectiveRuntimeConfig":
        config = cls(
            trials=int(value["trials"]),
            horizon=int(value["horizon"]),
            max_branches=int(value["max_branches"]),
            gamma=float(value["gamma"]),
            seed=int(value["seed"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.trials <= 0 or self.horizon <= 0:
            raise ValueError("prospective trials and horizon must be positive")
        if not 0 < self.max_branches <= MAX_OPTIONS:
            raise ValueError("prospective max_branches is outside encoder bounds")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("prospective gamma must be in [0,1]")


@dataclass(frozen=True)
class RuntimeProspectiveTree:
    """Planner batch plus real root-action correspondence and audit counts."""

    batch: ProspectivePlannerNumpyBatch
    root_actions: tuple[tuple[int, ...], ...]
    root_indices: tuple[tuple[tuple[int, int], ...], ...]
    nodes: tuple[tuple[dict[str, Any], ...], ...]
    audit: dict[str, int]


@dataclass(frozen=True)
class ProspectiveRerankResult:
    """Aggregated root policy score selected across real determinizations."""

    action: tuple[int, ...]
    root_scores: tuple[float, ...]
    trials_used: int
    nodes_scored: int


def enumerate_legal_action_combinations(
    select: dict[str, Any],
    *,
    max_branches: int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate a stable bounded prefix of legal simple/multi-select actions."""

    options = select.get("option") or []
    count = len(options)
    if count > MAX_OPTIONS:
        raise ValueError("decision has more options than the encoder supports")
    min_count = max(0, int(select.get("minCount", 1) or 0))
    max_count = min(count, max(min_count, int(select.get("maxCount", 1) or 0)))
    if min_count > count:
        raise ValueError("decision minCount exceeds its option count")
    actions: list[tuple[int, ...]] = []
    for size in range(min_count, max_count + 1):
        for action in itertools.combinations(range(count), size):
            actions.append(tuple(int(index) for index in action))
            if len(actions) >= max_branches:
                # Offline sidecars canonicalize the bounded candidate set
                # lexicographically after enumeration (and, during training,
                # possible behavior-action inclusion). Runtime has no behavior
                # label, but must preserve the same ordering for the common set.
                return tuple(sorted(set(actions)))
    return tuple(sorted(set(actions)))


def _sample_legal_action(
    select: dict[str, Any],
    rng: random.Random,
) -> tuple[int, ...]:
    count = len(select.get("option") or [])
    if count > MAX_OPTIONS:
        raise ValueError("rollout decision exceeds encoder capacity")
    min_count = max(0, int(select.get("minCount", 1) or 0))
    max_count = min(count, max(min_count, int(select.get("maxCount", 1) or 0)))
    if min_count > count:
        raise ValueError("rollout minCount exceeds option count")
    size = rng.randint(min_count, max_count) if max_count > min_count else min_count
    return tuple(sorted(rng.sample(range(count), size))) if size else ()


def _node_features(
    observation: dict[str, Any],
    action: tuple[int, ...],
    encoder: TokenEncoder,
    own_deck: list[int],
    *,
    depth: int,
    root_branch_order: int,
    parent_index: int,
    valid: bool,
) -> dict[str, Any]:
    encoded = encoder.encode(
        observation,
        picked=set(),
        self_deck=own_deck,
        tracker=None,
        ability_slots=None,
    )
    if len(action) == 1:
        action_index = int(action[0])
        src_pos = int(encoded["opt_src_pos"][action_index])
        tgt_pos = int(encoded["opt_tgt_pos"][action_index])
        verb = int(encoded["opt_verb"][action_index])
        has_action_index = True
    else:
        action_index = -1
        src_pos = -1
        tgt_pos = -1
        verb = 0
        has_action_index = False
    relation = int(
        pack_entity_zone_relation(src_pos, tgt_pos, verb).reshape(()).item()
    )
    branch_order = root_branch_order if depth == 1 else 0
    return {
        "depth": depth,
        "root_branch_order": root_branch_order,
        "branch_order": branch_order,
        "sibling_index": branch_order,
        "action_index": action_index,
        "has_action_index": has_action_index,
        "action_attr_mean": aggregate_action_opt_attr(
            encoded["opt_attr"],
            action,
        ),
        "action_set_features": aggregate_action_set_features(
            encoded["opt_src_pos"],
            encoded["opt_tgt_pos"],
            encoded["opt_verb"],
            action,
        ),
        "subselection_count": len(action),
        "select_type": int(np.asarray(encoded["select_type"]).reshape(-1)[0]),
        "select_context": int(
            np.asarray(encoded["select_context"]).reshape(-1)[0]
        ),
        "src_pos": src_pos,
        "tgt_pos": tgt_pos,
        "verb": verb,
        "entity_zone_relation_id": relation,
        "parent_index": parent_index,
        "valid": bool(valid),
        "action": action,
    }


def _runtime_seed(
    config: ProspectiveRuntimeConfig,
    *,
    match_time: int,
    trial_index: int,
) -> int:
    # Integer mixing is stable across Python versions and does not depend on
    # salted process hashes.
    return (
        int(config.seed)
        + 0x9E3779B1 * (int(match_time) + 1)
        + 0x85EBCA77 * (int(trial_index) + 1)
    ) & ((1 << 63) - 1)


def build_runtime_prospective_tree(
    observation: dict[str, Any],
    own_deck: list[int],
    encoder: TokenEncoder,
    trunk_context: np.ndarray,
    *,
    planner_config: ProspectivePlannerConfig,
    runtime_config: ProspectiveRuntimeConfig,
    match_time: int,
) -> RuntimeProspectiveTree | None:
    """Search real legal branches and construct the shared planner tensor batch."""

    runtime_config.validate()
    planner_config.validate()
    select = observation.get("select")
    current = observation.get("current")
    if not isinstance(select, dict) or not isinstance(current, dict):
        return None
    root_actions = enumerate_legal_action_combinations(
        select,
        max_branches=runtime_config.max_branches,
    )
    if len(root_actions) < 2:
        return None

    from cg import api

    trial_nodes: list[list[dict[str, Any]]] = []
    root_indices: list[list[tuple[int, int]]] = []
    audit = {
        "determination_failures": 0,
        "synthetic_fill_rejections": 0,
        "search_failures": 0,
        "trials_requested": runtime_config.trials,
        "trials_used": 0,
    }
    for trial_index in range(runtime_config.trials):
        trial_seed = _runtime_seed(
            runtime_config,
            match_time=match_time,
            trial_index=trial_index,
        )
        determination_audit: dict[str, int] = {}
        cleanup_failed = False
        try:
            determination = _determinize(
                observation,
                own_deck,
                random.Random(trial_seed),
                encoder,
                audit=determination_audit,
            )
        except Exception:
            audit["determination_failures"] += 1
            continue
        if (
            int(determination_audit.get("opponent_synthetic_cards", 0))
            or int(determination_audit.get("self_synthetic_cards", 0))
        ):
            audit["synthetic_fill_rejections"] += 1
            continue
        try:
            root_state = api.search_begin(
                api.to_observation_class(observation),
                **determination,
                manual_coin=True,
            )
        except Exception:
            audit["search_failures"] += 1
            continue

        nodes: list[dict[str, Any]] = []
        roots_for_trial: list[tuple[int, int]] = []
        try:
            for root_order, root_action in enumerate(root_actions):
                state = root_state
                current_observation = observation
                action = root_action
                parent_index = -1
                created_search_ids: list[int] = []
                branch_rng = random.Random(trial_seed)
                try:
                    for depth in range(1, runtime_config.horizon + 1):
                        local_index = len(nodes)
                        try:
                            next_state = api.search_step(
                                state.searchId, list(action)
                            )
                            created_search_ids.append(int(next_state.searchId))
                            next_observation = dataclasses.asdict(
                                next_state.observation
                            )
                        except Exception:
                            nodes.append(
                                _node_features(
                                    current_observation,
                                    action,
                                    encoder,
                                    own_deck,
                                    depth=depth,
                                    root_branch_order=root_order,
                                    parent_index=parent_index,
                                    valid=False,
                                )
                            )
                            audit["search_failures"] += 1
                            if depth == 1:
                                roots_for_trial.append(
                                    (local_index, root_order)
                                )
                            break
                        nodes.append(
                            _node_features(
                                current_observation,
                                action,
                                encoder,
                                own_deck,
                                depth=depth,
                                root_branch_order=root_order,
                                parent_index=parent_index,
                                valid=True,
                            )
                        )
                        if depth == 1:
                            roots_for_trial.append((local_index, root_order))
                        parent_index = local_index
                        state = next_state
                        current_observation = next_observation
                        result = int(
                            (current_observation.get("current") or {}).get(
                                "result", -1
                            )
                        )
                        if result >= 0 or current_observation.get("select") is None:
                            break
                        action = _sample_legal_action(
                            current_observation["select"], branch_rng
                        )
                finally:
                    for search_id in reversed(created_search_ids):
                        try:
                            api.search_release(search_id)
                        except Exception:
                            pass
        finally:
            try:
                api.search_release(root_state.searchId)
            except Exception:
                pass
            try:
                api.search_end()
            except Exception:
                audit["search_failures"] += 1
                cleanup_failed = True
        if cleanup_failed:
            continue
        trial_nodes.append(nodes)
        root_indices.append(roots_for_trial)
        audit["trials_used"] += 1

    if not trial_nodes:
        return None

    batch_size = len(trial_nodes)
    branch_count = max(len(nodes) for nodes in trial_nodes)
    context_array = np.asarray(trunk_context, dtype=np.float16)
    if (
        context_array.ndim != 2
        or context_array.shape[0] <= 0
        or context_array.shape[1] != planner_config.d_model
        or not np.isfinite(context_array).all()
    ):
        raise ValueError("trunk_context must be finite [C,d_model] FP16")
    context_length = context_array.shape[0]
    context = np.repeat(
        context_array.reshape(1, context_length, -1), batch_size, axis=0
    ).astype(np.float16)
    branch_tokens = np.zeros(
        (batch_size, branch_count, planner_config.d_model), dtype=np.float16
    )
    coordinates = np.zeros(
        (
            batch_size,
            context_length + branch_count,
            N_PROSPECTIVE_AXES,
        ),
        dtype=np.int32,
    )
    coordinates[:, :, MATCH_TIME_AXIS] = int(match_time)
    branch_valid = np.zeros((batch_size, branch_count), dtype=np.bool_)
    parent_index = np.full((batch_size, branch_count), -1, dtype=np.int32)
    node_index = np.full((batch_size, branch_count), -1, dtype=np.int64)

    for batch_index, nodes in enumerate(trial_nodes):
        for local_index, node in enumerate(nodes):
            branch_tokens[batch_index, local_index] = (
                encode_prospective_branch_features(
                    node,
                    d_model=planner_config.d_model,
                )
            )
            branch_valid[batch_index, local_index] = bool(node["valid"])
            parent_index[batch_index, local_index] = int(node["parent_index"])
            node_index[batch_index, local_index] = local_index
            branch_position = context_length + local_index
            coordinates[
                batch_index, branch_position, ROLLOUT_DEPTH_AXIS
            ] = int(node["depth"])
            coordinates[
                batch_index, branch_position, BRANCH_ACTION_AXIS
            ] = int(node["branch_order"])
            coordinates[
                batch_index, branch_position, ENTITY_ZONE_RELATION_AXIS
            ] = int(node["entity_zone_relation_id"])

    coordinates = validate_prospective_coordinates(coordinates)
    attention_mask = build_tree_attention_mask(
        parent_index,
        branch_valid,
        context_length=context_length,
    )
    planner_batch = ProspectivePlannerNumpyBatch(
        context=context,
        branch_tokens=branch_tokens,
        coordinates=coordinates,
        attention_mask=attention_mask,
        branch_valid=branch_valid,
        parent_index=parent_index,
        node_index=node_index,
        group_keys=tuple(
            ("runtime", trial_index, f"runtime-{trial_index}")
            for trial_index in range(batch_size)
        ),
    )
    planner_batch.validate(planner_config)
    return RuntimeProspectiveTree(
        batch=planner_batch,
        root_actions=root_actions,
        root_indices=tuple(tuple(items) for items in root_indices),
        nodes=tuple(tuple(nodes) for nodes in trial_nodes),
        audit=audit,
    )


def rerank_with_prospective_planner(
    planner: ProspectivePlannerTorch,
    tree: RuntimeProspectiveTree,
) -> ProspectiveRerankResult | None:
    """Aggregate trained branch-policy scores across real trials."""

    batch = tree.batch
    with torch.inference_mode():
        outputs = planner(
            torch.from_numpy(batch.context),
            torch.from_numpy(batch.branch_tokens),
            torch.from_numpy(batch.coordinates),
            torch.from_numpy(batch.attention_mask),
            torch.from_numpy(batch.branch_valid),
        )
    scores = (
        outputs[BRANCH_POLICY_SCORE].detach().to(torch.float32).cpu().numpy()
    )
    grouped_scores: list[list[float]] = [
        [] for _ in range(len(tree.root_actions))
    ]
    nodes_scored = 0
    for trial_index, roots in enumerate(tree.root_indices):
        for local_index, root_order in roots:
            if not bool(batch.branch_valid[trial_index, local_index]):
                continue
            value = float(scores[trial_index, local_index])
            if np.isfinite(value):
                grouped_scores[root_order].append(value)
                nodes_scored += 1
    aggregate = tuple(
        float(np.mean(values)) if values else float("-inf")
        for values in grouped_scores
    )
    finite = [index for index, value in enumerate(aggregate) if np.isfinite(value)]
    if not finite:
        return None
    best = max(finite, key=lambda index: (aggregate[index], -index))
    return ProspectiveRerankResult(
        action=tree.root_actions[best],
        root_scores=aggregate,
        trials_used=int(tree.audit["trials_used"]),
        nodes_scored=nodes_scored,
    )


__all__ = [
    "ProspectiveRerankResult",
    "ProspectiveRuntimeConfig",
    "RuntimeProspectiveTree",
    "build_runtime_prospective_tree",
    "enumerate_legal_action_combinations",
    "rerank_with_prospective_planner",
]
