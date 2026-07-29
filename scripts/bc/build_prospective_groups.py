"""Build additive prospective counterfactual groups from real replay decisions.

This pipeline is deliberately lateral to BC training. It reads real player-view
observations from Kaggle replay ZIPs, branches legal actions through the official
``cg.api`` search engine, and writes versioned sidecars only. It never changes
the policy input schema and never consumes the replay's hidden opponent deck for
simulation.

Example smoke build:

  uv run tcg-build-prospective \
    --config configs/smoke.json \
    --zip data/bc_replay_zip/2026-07-28.zip \
    --out data/bc_data/bc_smoke_2026_07_28/prospective_v1 \
    --max-groups 4 --trials 1 --horizon 2 --max-branches 4
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Iterator
import zipfile

import numpy as np

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import MAX_OPTIONS, TokenEncoder
from rl.prospective_input_adapter import (
    ACTION_ATTR_AGGREGATE_VERSION,
    ACTION_ATTR_WIDTH,
    ACTION_SET_FEATURE_VERSION,
    ACTION_SET_FEATURE_WIDTH,
    ACTION_SET_FOURIER_FREQUENCIES,
    ACTION_SET_MOMENT_ORDER,
    BRANCH_FEATURE_LAYOUT,
    BRANCH_FEATURE_LAYOUT_VERSION,
    PROSPECTIVE_INPUT_ADAPTER_VERSION,
    aggregate_action_opt_attr,
    aggregate_action_set_features,
)
from rl.prospective_schema import (
    MAX_ENCODED_POSITION,
    MAX_ENCODED_VERB,
    MIN_ENCODED_POSITION,
    MIN_ENCODED_VERB,
    PROSPECTIVE_COORD_SCHEMA_VERSION,
    pack_entity_zone_relation,
    prospective_coordinate_schema,
)
from rl.search_agent import _determinize
from rl.train_config import load_config


SCHEMA_VERSION = 2
PLANNER_VERSION = 1
ROOT_PARENT_ID = ""

NODE_DTYPE = np.dtype([
    ("parent_index", "i4"),
    ("depth", "u2"),
    ("root_branch_order", "u2"),
    ("branch_order", "u2"),
    ("sibling_index", "u2"),
    ("action_index", "i2"),
    ("has_action_index", "bool"),
    ("action_offset", "u8"),
    ("action_count", "u2"),
    ("action_attr_mean", "f4", (ACTION_ATTR_WIDTH,)),
    ("action_set_features", "f4", (ACTION_SET_FEATURE_WIDTH,)),
    ("subselection_count", "u2"),
    ("select_type", "i2"),
    ("select_context", "i2"),
    ("src_pos", "i2"),
    ("tgt_pos", "i2"),
    ("verb", "i2"),
    ("entity_zone_relation_id", "u4"),
    ("behavior_selected", "bool"),
    ("has_behavior_logprob", "bool"),
    ("behavior_logprob", "f4"),
    ("has_reference_logprob", "bool"),
    ("reference_logprob", "f4"),
    ("valid", "bool"),
    ("failure_stage_code", "u1"),
    ("failure_type_code", "u2"),
    ("terminal", "bool"),
    ("terminal_result", "i1"),
    ("win_signal", "i1"),
    ("loss_signal", "i1"),
    ("draw_signal", "i1"),
    ("ko_signal", "i1"),
    ("ko_against_signal", "i1"),
    ("prizes_taken", "u1"),
    ("prizes_lost", "u1"),
    ("reward_terminal", "f4"),
    ("reward_ko", "f4"),
    ("reward_prize", "f4"),
    ("scalar_reward", "f4"),
    ("scalar_return", "f4"),
])

BRANCH_DTYPE = np.dtype([
    ("episode_side_index", "u4"),
    ("step_id", "u4"),
    ("target_row", "i8"),
    ("group_id", "u8"),
    ("branch_order", "u2"),
    ("sibling_index", "u2"),
    ("action_index", "i2"),
    ("has_action_index", "bool"),
    ("action_offset", "u8"),
    ("action_count", "u2"),
    ("action_attr_mean", "f4", (ACTION_ATTR_WIDTH,)),
    ("action_set_features", "f4", (ACTION_SET_FEATURE_WIDTH,)),
    ("src_pos", "i2"),
    ("tgt_pos", "i2"),
    ("verb", "i2"),
    ("entity_zone_relation_id", "u4"),
    ("behavior_selected", "bool"),
    ("has_behavior_logprob", "bool"),
    ("behavior_logprob", "f4"),
    ("has_reference_logprob", "bool"),
    ("reference_logprob", "f4"),
    ("requested_trials", "u2"),
    ("valid_trials", "u2"),
    ("failed_trials", "u2"),
    ("valid", "bool"),
    ("mean_scalar_return", "f4"),
    ("terminal_rate", "f4"),
    ("win_rate", "f4"),
    ("ko_rate", "f4"),
    ("expected_prizes_taken", "f4"),
])

GROUP_DTYPE = np.dtype([
    ("episode_side_index", "u4"),
    ("episode_key", "u8"),
    ("side", "u1"),
    ("step_id", "u4"),
    ("group_id", "u8"),
    ("trial_index", "u2"),
    ("trial_seed", "u8"),
    ("determination_id", "u8"),
    ("node_start", "u8"),
    ("node_count", "u4"),
    ("target_row", "i8"),
    ("determination_mode", "u1"),
    ("determination_candidates", "u4"),
])

EPISODE_SIDE_DTYPE = np.dtype([
    ("episode_id", "U64"),
    ("episode_key", "u8"),
    ("side", "u1"),
    ("player_name", "U128"),
    ("opponent_name", "U128"),
    ("is_self", "bool"),
    ("outcome", "i1"),
])

ACTION_DTYPE = np.dtype("i2")

DETERMINATION_MODE_CODES = {
    "failed": 0,
    "consistent": 1,
    "best_overlap": 2,
    "rejected": 3,
}
FAILURE_STAGE_CODES = {"": 0, "determinize": 1, "search_begin": 2, "search_step": 3}


@dataclasses.dataclass(frozen=True)
class RealDecision:
    episode_id: str
    side: int
    step_id: int
    observation: dict[str, Any]
    behavior_action: tuple[int, ...]
    own_deck: tuple[int, ...]
    player_name: str
    opponent_name: str
    outcome: int
    is_self: bool


def _player_names(episode: dict[str, Any]) -> tuple[str, str]:
    info = episode.get("info") or {}
    team_names = info.get("TeamNames")
    if isinstance(team_names, list) and len(team_names) >= 2:
        return str(team_names[0] or ""), str(team_names[1] or "")
    agents = info.get("Agents") or []
    names: list[str] = []
    for side in range(2):
        agent = (
            agents[side]
            if side < len(agents) and isinstance(agents[side], dict)
            else {}
        )
        names.append(str(agent.get("Name") or agent.get("name") or ""))
    return names[0], names[1]


def _stable_hex(*parts: Any) -> str:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _stable_u64(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _stable_seed(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


@dataclasses.dataclass
class StableIdRegistry:
    """Fail loudly if distinct identity payloads ever collide at 64 bits."""

    payload_by_id: dict[tuple[str, int], tuple[Any, ...]] = dataclasses.field(
        default_factory=dict
    )

    def get(self, namespace: str, *parts: Any) -> int:
        identifier = _stable_u64(namespace, *parts)
        key = (namespace, identifier)
        payload = tuple(parts)
        previous = self.payload_by_id.setdefault(key, payload)
        if previous != payload:
            raise RuntimeError(
                f"BLAKE2b uint64 collision in {namespace}: "
                f"{previous!r} versus {payload!r}"
            )
        return identifier


def _entity_relation(
    observation: dict[str, Any],
    action: tuple[int, ...],
    encoder: TokenEncoder,
    own_deck: tuple[int, ...],
    *,
    opt_attr_override: np.ndarray | None = None,
) -> tuple[int, int, int, int, np.ndarray, np.ndarray]:
    """Map a single legal option onto the trainer's public E/Z/R coordinate.

    Multi-select actions do not have one unambiguous source/target relation, so
    their scalar coordinate is the documented neutral sentinel. The full
    subselection is stored losslessly in the compact flat action array.
    """
    select = observation.get("select") or {}
    options = select.get("option") or []
    src_pos, tgt_pos, verb = -1, -1, 0
    encoded = encoder.encode(
        observation,
        picked=set(),
        self_deck=list(own_deck),
        tracker=None,
        ability_slots=None,
    )
    action_attr_source = (
        encoded["opt_attr"]
        if opt_attr_override is None
        else np.asarray(opt_attr_override, dtype=np.float32)
    )
    action_attr_mean = aggregate_action_opt_attr(action_attr_source, action)
    action_set_features = aggregate_action_set_features(
        encoded["opt_src_pos"],
        encoded["opt_tgt_pos"],
        encoded["opt_verb"],
        action,
    )
    if len(action) == 1 and 0 <= action[0] < min(len(options), MAX_OPTIONS):
        option_index = action[0]
        src_pos = int(encoded["opt_src_pos"][option_index])
        tgt_pos = int(encoded["opt_tgt_pos"][option_index])
        verb = int(encoded["opt_verb"][option_index])
    if (
        not MIN_ENCODED_POSITION <= src_pos <= MAX_ENCODED_POSITION
        or not MIN_ENCODED_POSITION <= tgt_pos <= MAX_ENCODED_POSITION
    ):
        raise ValueError(
            f"prospective E/Z/R position outside v1 range: src={src_pos}, tgt={tgt_pos}"
        )
    if not MIN_ENCODED_VERB <= verb <= MAX_ENCODED_VERB:
        raise ValueError(f"prospective E/Z/R verb outside v1 range: {verb}")
    relation_id = int(
        pack_entity_zone_relation(src_pos, tgt_pos, verb).item()
    )
    return (
        src_pos,
        tgt_pos,
        verb,
        relation_id,
        action_attr_mean,
        action_set_features,
    )


def _legal_actions(select: dict[str, Any], max_branches: int) -> list[tuple[int, ...]]:
    """Enumerate a stable bounded prefix of legal selections."""
    options = select.get("option") or []
    count = len(options)
    min_count = max(0, int(select.get("minCount", 1) or 0))
    max_count = min(count, max(min_count, int(select.get("maxCount", 1) or 0)))
    actions: list[tuple[int, ...]] = []
    for size in range(min_count, max_count + 1):
        for action in itertools.combinations(range(count), size):
            actions.append(tuple(int(index) for index in action))
            if len(actions) >= max_branches:
                return actions
    return actions


def _sample_legal_action(select: dict[str, Any], rng: random.Random) -> tuple[int, ...]:
    """Sample one legal continuation without constructing an exponential action set."""
    count = len(select.get("option") or [])
    min_count = max(0, int(select.get("minCount", 1) or 0))
    max_count = min(count, max(min_count, int(select.get("maxCount", 1) or 0)))
    size = rng.randint(min_count, max_count) if max_count > min_count else min_count
    return tuple(sorted(rng.sample(range(count), size))) if size else ()


def _iter_real_decisions(
    episode: dict[str, Any],
    episode_id: str,
    *,
    both_sides: bool,
    self_aliases: frozenset[str],
) -> Iterator[RealDecision]:
    """Yield only actual replay decision observations and their recorded responses."""
    rewards = episode.get("rewards") or []
    if len(rewards) != 2 or rewards[0] is None or rewards[1] is None:
        return
    names = _player_names(episode)
    winner = 0 if rewards[0] > rewards[1] else 1
    sides = (0, 1) if both_sides else (winner,)
    for side in sides:
        entries = [
            step[side]
            for step in (episode.get("steps") or [])
            if len(step) > side
        ]
        own_deck: tuple[int, ...] | None = None
        step_id = 0
        outcome = 1 if rewards[side] > rewards[1 - side] else (
            -1 if rewards[side] < rewards[1 - side] else 0
        )
        for index, agent_step in enumerate(entries):
            observation = agent_step.get("observation") or {}
            action = agent_step.get("action")
            if isinstance(action, list) and len(action) == 60:
                own_deck = tuple(int(card) for card in action)
                continue
            select = observation.get("select")
            if select is None or own_deck is None:
                continue
            behavior = entries[index + 1].get("action") if index + 1 < len(entries) else None
            option_count = len(select.get("option") or [])
            min_count = max(0, int(select.get("minCount", 1) or 0))
            max_count = min(
                option_count,
                max(min_count, int(select.get("maxCount", 1) or 0)),
            )
            if not (
                isinstance(behavior, list)
                and min_count <= len(behavior) <= max_count
                and all(
                    isinstance(value, int) and 0 <= value < option_count
                    for value in behavior
                )
                and len(set(behavior)) == len(behavior)
            ):
                continue
            yield RealDecision(
                episode_id=episode_id,
                side=side,
                step_id=step_id,
                observation=observation,
                behavior_action=tuple(sorted(int(value) for value in behavior)),
                own_deck=own_deck,
                player_name=names[side],
                opponent_name=names[1 - side],
                outcome=outcome,
                is_self=names[side] in self_aliases,
            )
            step_id += 1


def _prize_count(observation: dict[str, Any], player: int) -> int:
    players = (observation.get("current") or {}).get("players") or []
    if not 0 <= player < len(players):
        return 0
    return len(players[player].get("prize") or [])


def _reward_fields(
    before: dict[str, Any],
    after: dict[str, Any],
    perspective: int,
) -> dict[str, Any]:
    before_mine = _prize_count(before, perspective)
    after_mine = _prize_count(after, perspective)
    before_theirs = _prize_count(before, 1 - perspective)
    after_theirs = _prize_count(after, 1 - perspective)
    prizes_taken = max(0, before_mine - after_mine)
    prizes_lost = max(0, before_theirs - after_theirs)
    current = after.get("current") or {}
    result = int(current.get("result", -1))
    terminal = result >= 0
    win = int(terminal and result == perspective)
    loss = int(terminal and result == 1 - perspective)
    draw = int(terminal and result == 2)
    ko = int(prizes_taken > 0)
    ko_against = int(prizes_lost > 0)
    reward_terminal = float(win - loss)
    reward_ko = float(ko - ko_against)
    reward_prize = float(prizes_taken - prizes_lost) / 6.0
    scalar_reward = reward_terminal + reward_prize
    return {
        "terminal": terminal,
        "terminal_result": result,
        "win_signal": win,
        "loss_signal": loss,
        "draw_signal": draw,
        "ko_signal": ko,
        "ko_against_signal": ko_against,
        "prizes_taken": prizes_taken,
        "prizes_lost": prizes_lost,
        "reward_terminal": reward_terminal,
        "reward_ko": reward_ko,
        "reward_prize": reward_prize,
        "scalar_reward": scalar_reward,
    }


def _determination_mode(audit: dict[str, int]) -> tuple[str, int]:
    if int(audit.get("best_overlap_determinizations", 0)):
        return "best_overlap", 1
    candidates = int(audit.get("consistent_candidates_total", 0))
    return "consistent", candidates


def _failed_node(
    decision: RealDecision,
    group_id: str,
    branch_id: str,
    encoder: TokenEncoder,
    *,
    trial_index: int,
    trial_seed: int,
    root_branch_order: int,
    action: tuple[int, ...],
    observation: dict[str, Any],
    failure_stage: str,
    failure_type: str,
    determination_mode: str,
    determination_candidates: int,
    opt_attr_override: np.ndarray | None = None,
) -> dict[str, Any]:
    select = observation.get("select") or {}
    (
        src_pos,
        tgt_pos,
        verb,
        relation_id,
        action_attr_mean,
        action_set_features,
    ) = _entity_relation(
        observation,
        action,
        encoder,
        decision.own_deck,
        opt_attr_override=opt_attr_override,
    )
    return {
        "episode_id": decision.episode_id,
        "side": decision.side,
        "step_id": decision.step_id,
        "group_id": group_id,
        "branch_id": branch_id,
        "parent_branch_id": ROOT_PARENT_ID,
        "has_parent": False,
        "depth": 1,
        "trial_index": trial_index,
        "trial_seed": trial_seed,
        "determination_id": _stable_hex(group_id, trial_index, "determination"),
        "root_branch_order": root_branch_order,
        "branch_order": root_branch_order,
        "sibling_index": root_branch_order,
        "action_index": action[0] if len(action) == 1 else -1,
        "has_action_index": len(action) == 1,
        "action": action,
        "action_attr_mean": action_attr_mean,
        "action_set_features": action_set_features,
        "subselection_count": len(action),
        "select_type": int(select.get("type", -1) or 0),
        "select_context": int(select.get("context", -1) or 0),
        "src_pos": src_pos,
        "tgt_pos": tgt_pos,
        "verb": verb,
        "entity_zone_relation_id": relation_id,
        "behavior_selected": action == decision.behavior_action,
        "has_behavior_logprob": False,
        "behavior_logprob": 0.0,
        "has_reference_logprob": False,
        "reference_logprob": 0.0,
        "valid": False,
        "failure_stage": failure_stage,
        "failure_type": failure_type,
        "determination_mode": determination_mode,
        "determination_candidates": determination_candidates,
        "terminal": False,
        "terminal_result": -1,
        "win_signal": 0,
        "loss_signal": 0,
        "draw_signal": 0,
        "ko_signal": 0,
        "ko_against_signal": 0,
        "prizes_taken": 0,
        "prizes_lost": 0,
        "reward_terminal": 0.0,
        "reward_ko": 0.0,
        "reward_prize": 0.0,
        "scalar_reward": 0.0,
        "scalar_return": 0.0,
        "player_name": decision.player_name,
        "opponent_name": decision.opponent_name,
        "is_self": decision.is_self,
        "outcome": decision.outcome,
    }


def _valid_node(
    decision: RealDecision,
    group_id: str,
    branch_id: str,
    parent_branch_id: str,
    encoder: TokenEncoder,
    *,
    has_parent: bool,
    depth: int,
    trial_index: int,
    trial_seed: int,
    root_branch_order: int,
    branch_order: int,
    action: tuple[int, ...],
    observation: dict[str, Any],
    rewards: dict[str, Any],
    determination_mode: str,
    determination_candidates: int,
    opt_attr_override: np.ndarray | None = None,
) -> dict[str, Any]:
    select = observation.get("select") or {}
    (
        src_pos,
        tgt_pos,
        verb,
        relation_id,
        action_attr_mean,
        action_set_features,
    ) = _entity_relation(
        observation,
        action,
        encoder,
        decision.own_deck,
        opt_attr_override=opt_attr_override,
    )
    return {
        "episode_id": decision.episode_id,
        "side": decision.side,
        "step_id": decision.step_id,
        "group_id": group_id,
        "branch_id": branch_id,
        "parent_branch_id": parent_branch_id,
        "has_parent": has_parent,
        "depth": depth,
        "trial_index": trial_index,
        "trial_seed": trial_seed,
        "determination_id": _stable_hex(group_id, trial_index, "determination"),
        "root_branch_order": root_branch_order,
        "branch_order": branch_order,
        "sibling_index": branch_order,
        "action_index": action[0] if len(action) == 1 else -1,
        "has_action_index": len(action) == 1,
        "action": action,
        "action_attr_mean": action_attr_mean,
        "action_set_features": action_set_features,
        "subselection_count": len(action),
        "select_type": int(select.get("type", -1) or 0),
        "select_context": int(select.get("context", -1) or 0),
        "src_pos": src_pos,
        "tgt_pos": tgt_pos,
        "verb": verb,
        "entity_zone_relation_id": relation_id,
        "behavior_selected": depth == 1 and action == decision.behavior_action,
        "has_behavior_logprob": False,
        "behavior_logprob": 0.0,
        "has_reference_logprob": False,
        "reference_logprob": 0.0,
        "valid": True,
        "failure_stage": "",
        "failure_type": "",
        "determination_mode": determination_mode,
        "determination_candidates": determination_candidates,
        **rewards,
        "scalar_return": 0.0,
        "player_name": decision.player_name,
        "opponent_name": decision.opponent_name,
        "is_self": decision.is_self,
        "outcome": decision.outcome,
    }


def _rollout_branch(
    decision: RealDecision,
    group_id: str,
    root_action: tuple[int, ...],
    root_branch_order: int,
    encoder: TokenEncoder,
    *,
    trial_index: int,
    trial_seed: int,
    root_state: Any,
    determination_mode: str,
    determination_candidates: int,
    root_opt_attr: np.ndarray,
    horizon: int,
    gamma: float,
) -> list[dict[str, Any]]:
    from cg import api

    rng = random.Random(trial_seed)
    nodes: list[dict[str, Any]] = []
    current_observation = decision.observation
    action = root_action
    parent_id = ROOT_PARENT_ID
    state = root_state
    created_search_ids: list[int] = []
    try:
        for depth in range(1, horizon + 1):
            branch_id = _stable_hex(group_id, trial_index, root_branch_order, depth)
            try:
                next_state = api.search_step(state.searchId, list(action))
                created_search_ids.append(int(next_state.searchId))
                next_observation = dataclasses.asdict(next_state.observation)
            except Exception as exc:
                failed = _failed_node(
                    decision,
                    group_id,
                    branch_id,
                    encoder,
                    trial_index=trial_index,
                    trial_seed=trial_seed,
                    root_branch_order=root_branch_order,
                    action=action,
                    observation=current_observation,
                    failure_stage="search_step",
                    failure_type=type(exc).__name__,
                    determination_mode=determination_mode,
                    determination_candidates=determination_candidates,
                    opt_attr_override=root_opt_attr if depth == 1 else None,
                )
                failed["parent_branch_id"] = parent_id
                failed["has_parent"] = depth > 1
                failed["depth"] = depth
                failed["branch_order"] = root_branch_order if depth == 1 else 0
                failed["sibling_index"] = failed["branch_order"]
                nodes.append(failed)
                break
            rewards = _reward_fields(
                current_observation, next_observation, decision.side
            )
            nodes.append(_valid_node(
                decision,
                group_id,
                branch_id,
                parent_id,
                encoder,
                has_parent=(depth > 1),
                depth=depth,
                trial_index=trial_index,
                trial_seed=trial_seed,
                root_branch_order=root_branch_order,
                branch_order=root_branch_order if depth == 1 else 0,
                action=action,
                observation=current_observation,
                rewards=rewards,
                determination_mode=determination_mode,
                determination_candidates=determination_candidates,
                opt_attr_override=root_opt_attr if depth == 1 else None,
            ))
            parent_id = branch_id
            state = next_state
            current_observation = next_observation
            if rewards["terminal"] or current_observation.get("select") is None:
                break
            action = _sample_legal_action(
                current_observation["select"], rng
            )
    finally:
        for search_id in reversed(created_search_ids):
            try:
                api.search_release(search_id)
            except Exception:
                pass

    running_return = 0.0
    for node in reversed(nodes):
        running_return = float(node["scalar_reward"]) + gamma * running_return
        node["scalar_return"] = running_return
    return nodes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class MaterializedRootOptAttr:
    """Memory-mapped BC option features with a compact key-to-row index."""

    values: np.ndarray
    first_row: dict[tuple[str, int, int], int]
    metadata: np.ndarray
    dataset_build_fingerprint: str
    dataset_contract_sha256: str

    def row_for(self, key: tuple[str, int, int]) -> int:
        try:
            return self.first_row[key]
        except KeyError as error:
            raise KeyError(f"no materialized BC root row for {key}") from error

    def get(self, key: tuple[str, int, int]) -> np.ndarray:
        # This is a view into the mmap. It deliberately does not copy the
        # [MAX_OPTIONS, ACTION_ATTR_WIDTH] row.
        return self.values[self.row_for(key)]


def _load_materialized_root_opt_attr(
    dataset_dir: Path,
    *,
    dataset_contract: dict[str, Any] | None = None,
) -> MaterializedRootOptAttr:
    """Load target-free BC input features for exact root train/runtime parity."""

    metadata_path = dataset_dir / "episode_meta.npy"
    attributes_path = dataset_dir / "opt_attr.npy"
    if not metadata_path.is_file() or not attributes_path.is_file():
        raise FileNotFoundError(
            "prospective root features require the materialized BC corpus "
            f"at {dataset_dir} (episode_meta.npy + opt_attr.npy)"
        )
    metadata = np.load(metadata_path, mmap_mode="r", allow_pickle=False)
    attributes = np.load(attributes_path, mmap_mode="r", allow_pickle=False)
    if len(metadata) != len(attributes):
        raise ValueError("BC episode_meta and opt_attr row counts differ")
    if attributes.ndim != 3 or attributes.shape[2] != ACTION_ATTR_WIDTH:
        raise ValueError(
            "BC opt_attr has incompatible shape for prospective action features"
        )
    required = {"episode_id", "side", "step_id"}
    if metadata.dtype.names is None or not required <= set(metadata.dtype.names):
        raise ValueError("BC episode_meta lacks prospective root join fields")
    manifest_path = dataset_dir / "dataset_manifest.json"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as stream:
            dataset_manifest = json.load(stream)
        contract_bytes = manifest_path.read_bytes()
    elif dataset_contract is not None:
        dataset_manifest = dataset_contract
        contract_bytes = json.dumps(
            dataset_contract, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    else:
        raise FileNotFoundError(
            "prospective root parity requires dataset_manifest.json or the "
            "in-progress BC build contract"
        )
    dataset_build_fingerprint = dataset_manifest.get("build_fingerprint")
    if not isinstance(dataset_build_fingerprint, str) or not dataset_build_fingerprint:
        raise ValueError("BC dataset manifest lacks build_fingerprint")
    first_row: dict[tuple[str, int, int], int] = {}
    for row_index, row in enumerate(metadata):
        key = (
            str(row["episode_id"]),
            int(row["side"]),
            int(row["step_id"]),
        )
        first_row.setdefault(key, row_index)
    return MaterializedRootOptAttr(
        values=attributes,
        first_row=first_row,
        metadata=metadata,
        dataset_build_fingerprint=dataset_build_fingerprint,
        dataset_contract_sha256=hashlib.sha256(contract_bytes).hexdigest(),
    )


def _failure_type_code(name: str) -> int:
    if not name:
        return 0
    code = int.from_bytes(
        hashlib.blake2b(name.encode("utf-8"), digest_size=2).digest(), "little"
    )
    return code or 1


def _compact_tree(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Pack one bounded tree; offsets are local until the shard merge."""

    nodes = np.zeros((len(records),), dtype=NODE_DTYPE)
    actions: list[int] = []
    local_by_id: dict[str, int] = {}
    for index, record in enumerate(records):
        branch_id = str(record["branch_id"])
        if branch_id in local_by_id:
            raise RuntimeError("duplicate branch id inside one prospective tree")
        parent_index = -1
        if bool(record["has_parent"]):
            parent_id = str(record["parent_branch_id"])
            if parent_id not in local_by_id:
                raise RuntimeError("prospective nodes are not parent-before-child")
            parent_index = local_by_id[parent_id]
        local_by_id[branch_id] = index
        action = tuple(int(value) for value in record["action"])
        if any(value < 0 or value >= MAX_OPTIONS for value in action):
            raise RuntimeError("prospective action is outside int16 option domain")
        offset = len(actions)
        actions.extend(action)
        values = {
            "parent_index": parent_index,
            "depth": record["depth"],
            "root_branch_order": record["root_branch_order"],
            "branch_order": record["branch_order"],
            "sibling_index": record["sibling_index"],
            "action_index": record["action_index"],
            "has_action_index": record["has_action_index"],
            "action_offset": offset,
            "action_count": len(action),
            "action_attr_mean": record["action_attr_mean"],
            "action_set_features": record["action_set_features"],
            "subselection_count": len(action),
            "select_type": record["select_type"],
            "select_context": record["select_context"],
            "src_pos": record["src_pos"],
            "tgt_pos": record["tgt_pos"],
            "verb": record["verb"],
            "entity_zone_relation_id": record["entity_zone_relation_id"],
            "behavior_selected": record["behavior_selected"],
            "has_behavior_logprob": record["has_behavior_logprob"],
            "behavior_logprob": record["behavior_logprob"],
            "has_reference_logprob": record["has_reference_logprob"],
            "reference_logprob": record["reference_logprob"],
            "valid": record["valid"],
            "failure_stage_code": FAILURE_STAGE_CODES[record["failure_stage"]],
            "failure_type_code": _failure_type_code(record["failure_type"]),
            "terminal": record["terminal"],
            "terminal_result": record["terminal_result"],
            "win_signal": record["win_signal"],
            "loss_signal": record["loss_signal"],
            "draw_signal": record["draw_signal"],
            "ko_signal": record["ko_signal"],
            "ko_against_signal": record["ko_against_signal"],
            "prizes_taken": record["prizes_taken"],
            "prizes_lost": record["prizes_lost"],
            "reward_terminal": record["reward_terminal"],
            "reward_ko": record["reward_ko"],
            "reward_prize": record["reward_prize"],
            "scalar_reward": record["scalar_reward"],
            "scalar_return": record["scalar_return"],
        }
        for field, value in values.items():
            nodes[index][field] = value
    return nodes, np.asarray(actions, dtype=ACTION_DTYPE)


def _write_shard(
    work_dir: Path,
    shard_index: int,
    trees: list[dict[str, Any]],
) -> Path:
    node_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    groups = np.zeros((len(trees),), dtype=GROUP_DTYPE)
    node_cursor = 0
    action_cursor = 0
    shard_audit = {
        "determination_failures": 0,
        "synthetic_fill_rejections": 0,
        "search_failures": 0,
    }
    for index, tree in enumerate(trees):
        compact_nodes, compact_actions = _compact_tree(tree["records"])
        if len(compact_nodes):
            compact_nodes["action_offset"] += action_cursor
        node_chunks.append(compact_nodes)
        action_chunks.append(compact_actions)
        meta = tree["group"]
        for field in GROUP_DTYPE.names or ():
            groups[index][field] = (
                node_cursor if field == "node_start" else
                len(compact_nodes) if field == "node_count" else
                meta[field]
            )
        node_cursor += len(compact_nodes)
        action_cursor += len(compact_actions)
        for key in shard_audit:
            shard_audit[key] += int(tree["audit"].get(key, 0))

    nodes = (
        np.concatenate(node_chunks)
        if node_chunks else np.empty((0,), dtype=NODE_DTYPE)
    )
    actions = (
        np.concatenate(action_chunks)
        if action_chunks else np.empty((0,), dtype=ACTION_DTYPE)
    )
    temporary = Path(tempfile.mkdtemp(prefix=".shard-", dir=work_dir))
    final = work_dir / f"shard_{shard_index:08d}"
    try:
        np.save(temporary / "nodes.npy", nodes, allow_pickle=False)
        np.save(temporary / "actions.npy", actions, allow_pickle=False)
        np.save(temporary / "groups.npy", groups, allow_pickle=False)
        marker = {
            "shard_index": shard_index,
            "group_keys": [
                [int(row["group_id"]), int(row["trial_index"])]
                for row in groups
            ],
            "node_rows": len(nodes),
            "action_rows": len(actions),
            "group_rows": len(groups),
            "audit": shard_audit,
        }
        (temporary / ".done").write_text(
            json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, final)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final


def _completed_shards(work_dir: Path) -> list[Path]:
    result = []
    for path in sorted(work_dir.glob("shard_*")):
        if not path.is_dir() or not (path / ".done").is_file():
            raise RuntimeError(f"partial prospective shard blocks resume: {path}")
        result.append(path)
    return result


def _merge_shards(
    shards: list[Path],
    staging: Path,
    episode_sides: np.ndarray,
) -> tuple[np.memmap, np.memmap, np.memmap, np.ndarray, np.ndarray]:
    totals = {"nodes": 0, "actions": 0, "groups": 0}
    for shard in shards:
        for name in totals:
            array = np.load(shard / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            totals[name] += len(array)
    nodes = np.lib.format.open_memmap(
        staging / "prospective_nodes.npy",
        mode="w+",
        dtype=NODE_DTYPE,
        shape=(totals["nodes"],),
    )
    actions = np.lib.format.open_memmap(
        staging / "prospective_actions.npy",
        mode="w+",
        dtype=ACTION_DTYPE,
        shape=(totals["actions"],),
    )
    groups = np.lib.format.open_memmap(
        staging / "prospective_groups.npy",
        mode="w+",
        dtype=GROUP_DTYPE,
        shape=(totals["groups"],),
    )
    node_cursor = action_cursor = group_cursor = 0
    for shard in shards:
        shard_nodes = np.load(shard / "nodes.npy", mmap_mode="r", allow_pickle=False)
        shard_actions = np.load(shard / "actions.npy", mmap_mode="r", allow_pickle=False)
        shard_groups = np.load(shard / "groups.npy", mmap_mode="r", allow_pickle=False)
        node_stop = node_cursor + len(shard_nodes)
        action_stop = action_cursor + len(shard_actions)
        group_stop = group_cursor + len(shard_groups)
        nodes[node_cursor:node_stop] = shard_nodes
        if len(shard_nodes):
            nodes["action_offset"][node_cursor:node_stop] += action_cursor
        actions[action_cursor:action_stop] = shard_actions
        groups[group_cursor:group_stop] = shard_groups
        if len(shard_groups):
            groups["node_start"][group_cursor:group_stop] += node_cursor
        node_cursor, action_cursor, group_cursor = node_stop, action_stop, group_stop
    nodes.flush()
    actions.flush()
    groups.flush()
    offsets = np.empty((len(groups) + 1,), dtype=np.uint64)
    if len(groups):
        offsets[:-1] = groups["node_start"]
        offsets[-1] = np.uint64(len(nodes))
    else:
        offsets[0] = 0
    np.save(staging / "prospective_group_offsets.npy", offsets, allow_pickle=False)
    np.save(
        staging / "prospective_episode_sides.npy",
        episode_sides,
        allow_pickle=False,
    )
    return nodes, actions, groups, offsets, episode_sides


def _summarize_branches(
    nodes: np.ndarray,
    groups: np.ndarray,
) -> np.memmap:
    """Stream branch aggregation one contiguous decision group at a time."""

    branch_rows = 0
    previous_group_id: int | None = None
    for group in groups:
        group_id = int(group["group_id"])
        if group_id == previous_group_id:
            continue
        start = int(group["node_start"])
        stop = start + int(group["node_count"])
        branch_rows += int(np.count_nonzero(nodes["depth"][start:stop] == 1))
        previous_group_id = group_id
    destination = Path(nodes.filename).parent / "prospective_branches.npy"
    branches = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=BRANCH_DTYPE,
        shape=(branch_rows,),
    )

    output_index = 0
    current_group_id: int | None = None
    current: dict[int, dict[str, Any]] = {}

    def write_current() -> None:
        nonlocal output_index
        for order in sorted(current):
            summary = current[order]
            group = summary["group"]
            node = summary["node"]
            valid = int(summary["valid_trials"])
            denominator = max(valid, 1)
            values = {
                "episode_side_index": group["episode_side_index"],
                "step_id": group["step_id"],
                "target_row": group["target_row"],
                "group_id": group["group_id"],
                "branch_order": node["branch_order"],
                "sibling_index": node["sibling_index"],
                "action_index": node["action_index"],
                "has_action_index": node["has_action_index"],
                "action_offset": node["action_offset"],
                "action_count": node["action_count"],
                "action_attr_mean": node["action_attr_mean"],
                "action_set_features": node["action_set_features"],
                "src_pos": node["src_pos"],
                "tgt_pos": node["tgt_pos"],
                "verb": node["verb"],
                "entity_zone_relation_id": node["entity_zone_relation_id"],
                "behavior_selected": node["behavior_selected"],
                "has_behavior_logprob": node["has_behavior_logprob"],
                "behavior_logprob": node["behavior_logprob"],
                "has_reference_logprob": node["has_reference_logprob"],
                "reference_logprob": node["reference_logprob"],
                "requested_trials": summary["requested_trials"],
                "valid_trials": valid,
                "failed_trials": summary["requested_trials"] - valid,
                "valid": bool(valid),
                "mean_scalar_return": summary["return_sum"] / denominator,
                "terminal_rate": summary["terminal_sum"] / denominator,
                "win_rate": summary["win_sum"] / denominator,
                "ko_rate": summary["ko_sum"] / denominator,
                "expected_prizes_taken": summary["prizes_sum"] / denominator,
            }
            for field, value in values.items():
                branches[output_index][field] = value
            output_index += 1

    for group in groups:
        group_id = int(group["group_id"])
        if current_group_id is None:
            current_group_id = group_id
        elif group_id != current_group_id:
            write_current()
            current = {}
            current_group_id = group_id
        start = int(group["node_start"])
        stop = start + int(group["node_count"])
        for node in nodes[start:stop]:
            if int(node["depth"]) != 1:
                continue
            order = int(node["root_branch_order"])
            summary = current.setdefault(order, {
                "group": group,
                "node": node.copy(),
                "requested_trials": 0,
                "valid_trials": 0,
                "return_sum": 0.0,
                "terminal_sum": 0,
                "win_sum": 0,
                "ko_sum": 0,
                "prizes_sum": 0,
            })
            summary["requested_trials"] += 1
            if bool(node["valid"]):
                summary["valid_trials"] += 1
                summary["return_sum"] += float(node["scalar_return"])
                summary["terminal_sum"] += int(node["terminal"])
                summary["win_sum"] += int(node["win_signal"])
                summary["ko_sum"] += int(node["ko_signal"])
                summary["prizes_sum"] += int(node["prizes_taken"])
    if current_group_id is not None:
        write_current()
    if output_index != branch_rows:
        raise RuntimeError("prospective branch pre-count/merge mismatch")
    branches.flush()
    return branches


def build(
    zip_path: Path | list[Path] | tuple[Path, ...],
    out_dir: Path,
    *,
    config_path: Path,
    max_episodes: int,
    max_groups: int,
    max_branches: int,
    trials: int,
    horizon: int,
    gamma: float,
    flush_groups: int = 32,
    bc_dataset_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path=str(config_path))
    if (
        trials <= 0 or horizon <= 0 or max_branches < 2
        or max_groups < 0 or flush_groups <= 0
    ):
        raise ValueError(
            "trials/horizon/flush_groups must be positive, max_branches >= 2, "
            "and max_groups >= 0 (0 means all branchable decisions)"
        )
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing prospective dataset: {out_dir}"
        )
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir.with_name(f"{out_dir.name}.work")
    work_dir.mkdir(parents=True, exist_ok=True)

    encoder = TokenEncoder(get_card_table())
    aliases = frozenset(cfg.bc_self_aliases)
    root_opt_attr = _load_materialized_root_opt_attr(
        out_dir.parent,
        dataset_contract=bc_dataset_contract,
    )
    id_registry = StableIdRegistry()
    zip_paths = (
        (zip_path,)
        if isinstance(zip_path, Path)
        else tuple(Path(path) for path in zip_path)
    )
    if not zip_paths:
        raise ValueError("at least one real replay ZIP is required")
    episode_refs: list[tuple[Path, str]] = []
    seen_episodes: dict[str, tuple[int, int]] = {}
    source_archives: list[dict[str, Any]] = []
    for source_path in zip_paths:
        json_members = 0
        with zipfile.ZipFile(source_path) as archive:
            for info in archive.infolist():
                if not info.filename.endswith(".json"):
                    continue
                json_members += 1
                episode_id = Path(info.filename).stem
                identity = (int(info.CRC), int(info.file_size))
                previous = seen_episodes.get(episode_id)
                if previous is None:
                    seen_episodes[episode_id] = identity
                    episode_refs.append((source_path, info.filename))
                elif previous != identity:
                    raise ValueError(
                        "episode id collision with different replay content: "
                        f"{episode_id}"
                    )
        source_archives.append({
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "json_members": json_members,
        })
    episode_refs = episode_refs[:max_episodes] if max_episodes else episode_refs
    build_identity = {
        "schema_version": SCHEMA_VERSION,
        "input_adapter_version": PROSPECTIVE_INPUT_ADAPTER_VERSION,
        "action_aggregate_version": ACTION_ATTR_AGGREGATE_VERSION,
        "action_set_feature_version": ACTION_SET_FEATURE_VERSION,
        "branch_feature_layout_version": BRANCH_FEATURE_LAYOUT_VERSION,
        "sources": source_archives,
        "bc_dataset_build_fingerprint": root_opt_attr.dataset_build_fingerprint,
        "bc_dataset_contract_sha256": root_opt_attr.dataset_contract_sha256,
        "config": {
            "path": str(config_path),
            "seed": int(cfg.seed),
            "max_episodes": max_episodes,
            "max_groups": max_groups,
            "max_branches": max_branches,
            "trials": trials,
            "horizon": horizon,
            "gamma": gamma,
            "workers": 1,
            "self_aliases": sorted(aliases),
        },
    }
    identity_path = work_dir / "build_contract.json"
    if identity_path.is_file():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        if previous != build_identity:
            raise RuntimeError(
                f"prospective resume contract changed; preserve or remove {work_dir}"
            )
    else:
        identity_path.write_text(
            json.dumps(build_identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    shards = _completed_shards(work_dir)
    def completed_group_keys() -> Iterator[tuple[int, int]]:
        for shard in shards:
            marker = json.loads((shard / ".done").read_text(encoding="utf-8"))
            for group_id, trial_index in marker["group_keys"]:
                yield int(group_id), int(trial_index)

    completed_iter = completed_group_keys()
    next_completed = next(completed_iter, None)
    next_shard = len(shards)
    pending: list[dict[str, Any]] = []
    selected_episode_ids: list[str] = []
    episode_side_rows: list[tuple[Any, ...]] = []
    episode_side_index: dict[tuple[str, int], int] = {}
    groups_emitted = 0
    candidate_roots = 0
    skipped_unmaterialized_roots = 0
    skipped_unmaterialized_sides: set[tuple[str, int]] = set()

    def flush_pending() -> None:
        nonlocal pending, next_shard
        if not pending:
            return
        _write_shard(work_dir, next_shard, pending)
        next_shard += 1
        pending = []

    for member_index, (source_path, member) in enumerate(episode_refs):
        episode_group_quota: int | None = None
        if max_groups:
            base_quota, extra = divmod(max_groups, max(len(episode_refs), 1))
            episode_group_quota = base_quota + int(member_index < extra)
            if episode_group_quota == 0:
                continue
        with zipfile.ZipFile(source_path) as archive:
            episode = json.loads(archive.read(member))
        episode_id = Path(member).stem
        selected_episode_ids.append(episode_id)
        episode_groups = 0
        for decision in _iter_real_decisions(
            episode,
            episode_id,
            both_sides=bool(cfg.bc_both_sides),
            self_aliases=aliases,
        ):
            candidate_roots += 1
            decision_key = (decision.episode_id, decision.side, decision.step_id)
            target_row = root_opt_attr.first_row.get(decision_key)
            if target_row is None:
                # The BC encoder deliberately truncates a side after a tracker,
                # legality, or encoding failure. Replay iteration can still see
                # later decisions, but they cannot be prospective roots because
                # no real BC input row exists to join as their training target.
                # Never synthesize that row and never abort unrelated episodes.
                skipped_unmaterialized_roots += 1
                skipped_unmaterialized_sides.add(
                    (decision.episode_id, decision.side)
                )
                continue
            target_meta = root_opt_attr.metadata[target_row]
            target_key = (
                str(target_meta["episode_id"]),
                int(target_meta["side"]),
                int(target_meta["step_id"]),
            )
            if target_key != decision_key:
                raise RuntimeError(
                    f"prospective target_row mismatch: {decision_key} != {target_key}"
                )
            materialized_root_opt_attr = root_opt_attr.get(decision_key)
            select = decision.observation.get("select") or {}
            actions = _legal_actions(select, max_branches)
            if decision.behavior_action not in actions:
                if len(actions) < max_branches:
                    actions.append(decision.behavior_action)
                elif actions:
                    actions[-1] = decision.behavior_action
            actions = sorted(set(actions))
            if len(actions) < 2:
                continue
            side_key = (decision.episode_id, decision.side)
            if side_key not in episode_side_index:
                episode_key = id_registry.get("episode", decision.episode_id)
                episode_side_index[side_key] = len(episode_side_rows)
                episode_side_rows.append((
                    decision.episode_id,
                    episode_key,
                    decision.side,
                    decision.player_name,
                    decision.opponent_name,
                    decision.is_self,
                    decision.outcome,
                ))
            side_row = episode_side_index[side_key]
            episode_key = int(episode_side_rows[side_row][1])
            group_id = id_registry.get(
                "group",
                SCHEMA_VERSION,
                decision.episode_id,
                decision.side,
                decision.step_id,
            )
            groups_emitted += 1
            for trial_index in range(trials):
                determination_id = id_registry.get(
                    "determination", group_id, trial_index
                )
                group_key = (group_id, trial_index)
                if next_completed is not None:
                    if group_key != next_completed:
                        raise RuntimeError(
                            "completed prospective shards are not a deterministic "
                            f"prefix: expected {group_key}, found {next_completed}"
                        )
                    next_completed = next(completed_iter, None)
                    continue
                trial_seed = _stable_seed(cfg.seed, group_id, trial_index)
                tree_audit = {
                    "determination_failures": 0,
                    "synthetic_fill_rejections": 0,
                    "search_failures": 0,
                }
                determination_audit: dict[str, int] = {}
                records: list[dict[str, Any]] = []
                mode = "failed"
                candidate_count = 0
                try:
                    determination = _determinize(
                        decision.observation,
                        list(decision.own_deck),
                        random.Random(trial_seed),
                        encoder,
                        audit=determination_audit,
                    )
                except Exception as exc:
                    tree_audit["determination_failures"] += 1
                    for order, action in enumerate(actions):
                        records.append(_failed_node(
                            decision, str(group_id),
                            _stable_hex(group_id, trial_index, order, 1), encoder,
                            trial_index=trial_index, trial_seed=trial_seed,
                            root_branch_order=order, action=action,
                            observation=decision.observation,
                            failure_stage="determinize",
                            failure_type=type(exc).__name__,
                            determination_mode=mode,
                            determination_candidates=0,
                            opt_attr_override=materialized_root_opt_attr,
                        ))
                else:
                    if (
                        int(determination_audit.get("opponent_synthetic_cards", 0))
                        or int(determination_audit.get("self_synthetic_cards", 0))
                    ):
                        mode = "rejected"
                        tree_audit["synthetic_fill_rejections"] += 1
                        for order, action in enumerate(actions):
                            records.append(_failed_node(
                                decision, str(group_id),
                                _stable_hex(group_id, trial_index, order, 1), encoder,
                                trial_index=trial_index, trial_seed=trial_seed,
                                root_branch_order=order, action=action,
                                observation=decision.observation,
                                failure_stage="determinize",
                                failure_type="SyntheticFillRejected",
                                determination_mode=mode,
                                determination_candidates=0,
                                opt_attr_override=materialized_root_opt_attr,
                            ))
                    else:
                        mode, candidate_count = _determination_mode(determination_audit)
                        from cg import api
                        try:
                            root_state = api.search_begin(
                                api.to_observation_class(decision.observation),
                                **determination,
                                manual_coin=True,
                            )
                        except Exception as exc:
                            tree_audit["search_failures"] += 1
                            for order, action in enumerate(actions):
                                records.append(_failed_node(
                                    decision, str(group_id),
                                    _stable_hex(group_id, trial_index, order, 1),
                                    encoder,
                                    trial_index=trial_index,
                                    trial_seed=trial_seed,
                                    root_branch_order=order,
                                    action=action,
                                    observation=decision.observation,
                                    failure_stage="search_begin",
                                    failure_type=type(exc).__name__,
                                    determination_mode=mode,
                                    determination_candidates=candidate_count,
                                    opt_attr_override=materialized_root_opt_attr,
                                ))
                        else:
                            try:
                                for order, action in enumerate(actions):
                                    branch_records = _rollout_branch(
                                        decision, str(group_id), action, order, encoder,
                                        trial_index=trial_index,
                                        trial_seed=trial_seed,
                                        root_state=root_state,
                                        determination_mode=mode,
                                        determination_candidates=candidate_count,
                                        root_opt_attr=materialized_root_opt_attr,
                                        horizon=horizon,
                                        gamma=gamma,
                                    )
                                    tree_audit["search_failures"] += sum(
                                        not record["valid"]
                                        for record in branch_records
                                    )
                                    records.extend(branch_records)
                            finally:
                                try:
                                    api.search_release(root_state.searchId)
                                except Exception:
                                    tree_audit["search_failures"] += 1
                                try:
                                    api.search_end()
                                except Exception:
                                    tree_audit["search_failures"] += 1
                pending.append({
                    "group": {
                        "episode_side_index": side_row,
                        "episode_key": episode_key,
                        "side": decision.side,
                        "step_id": decision.step_id,
                        "group_id": group_id,
                        "trial_index": trial_index,
                        "trial_seed": trial_seed,
                        "determination_id": determination_id,
                        "target_row": target_row,
                        "determination_mode": DETERMINATION_MODE_CODES[mode],
                        "determination_candidates": candidate_count,
                    },
                    "records": records,
                    "audit": tree_audit,
                })
                if len(pending) >= flush_groups:
                    flush_pending()
            episode_groups += 1
            if (
                episode_group_quota is not None
                and episode_groups >= episode_group_quota
            ):
                break
    flush_pending()
    if next_completed is not None:
        raise RuntimeError(
            "completed prospective shards extend beyond selected real groups"
        )
    shards = _completed_shards(work_dir)
    if not shards:
        raise RuntimeError("prospective corpus produced no branchable real groups")
    episode_sides = np.asarray(episode_side_rows, dtype=EPISODE_SIDE_DTYPE)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        nodes, actions, groups, offsets, episode_sides = _merge_shards(
            shards, staging, episode_sides
        )
        branches = _summarize_branches(nodes, groups)
        if len(branches) == 0 or int(np.count_nonzero(branches["valid"])) == 0:
            raise RuntimeError(
                "prospective corpus produced no valid counterfactual branch"
            )
        for array, name in ((nodes, "nodes"), (branches, "branches")):
            for field in array.dtype.names or ():
                if np.issubdtype(array.dtype[field], np.floating):
                    if not np.isfinite(array[field]).all():
                        raise RuntimeError(f"{name}.{field} contains NaN or infinity")
        audit_counts = {
            "episodes_read": len(selected_episode_ids),
            "candidate_roots": candidate_roots,
            "groups_emitted": groups_emitted,
            "tree_groups_emitted": len(groups),
            "shards_emitted": len(shards),
            "skipped_unmaterialized_roots": skipped_unmaterialized_roots,
            "sides_with_unmaterialized_roots": len(skipped_unmaterialized_sides),
            "determination_failures": 0,
            "synthetic_fill_rejections": 0,
            "search_failures": 0,
        }
        for shard in shards:
            marker = json.loads((shard / ".done").read_text(encoding="utf-8"))
            for key, value in marker["audit"].items():
                audit_counts[key] += int(value)
        contract = {
        "schema_version": SCHEMA_VERSION,
        "planner_version": PLANNER_VERSION,
        "input_adapter_version": PROSPECTIVE_INPUT_ADAPTER_VERSION,
        "prospective_coord_schema_version": PROSPECTIVE_COORD_SCHEMA_VERSION,
        "prospective_coordinate_schema": prospective_coordinate_schema(),
        "action_feature_schema": {
            "aggregate_version": ACTION_ATTR_AGGREGATE_VERSION,
            "branch_feature_layout_version": BRANCH_FEATURE_LAYOUT_VERSION,
            "field": "action_attr_mean",
            "dtype": "float32",
            "shape": [ACTION_ATTR_WIDTH],
            "reduction": "arithmetic mean over selected encoder opt_attr rows",
            "empty_action": "all-zero vector",
            "source": "real TokenEncoder opt_attr; no prediction target",
            "adapter_slots": [
                int(BRANCH_FEATURE_LAYOUT["action_attr_mean"].start),
                int(BRANCH_FEATURE_LAYOUT["action_attr_mean"].stop),
            ],
            "action_set_feature_version": ACTION_SET_FEATURE_VERSION,
            "action_set_field": "action_set_features",
            "action_set_dtype": "float32",
            "action_set_shape": [ACTION_SET_FEATURE_WIDTH],
            "action_set_adapter_slots": [
                int(BRANCH_FEATURE_LAYOUT["action_set_features"].start),
                int(BRANCH_FEATURE_LAYOUT["action_set_features"].stop),
            ],
            "action_set_moment_order": list(ACTION_SET_MOMENT_ORDER),
            "action_set_fourier_frequencies": list(
                ACTION_SET_FOURIER_FREQUENCIES
            ),
            "action_set_semantics": (
                "order-invariant normalized min/mean/max moments of "
                "option_index/src_pos/tgt_pos/verb followed by mean sin/cos "
                "Fourier features of option_index"
            ),
            "root_opt_attr_source": (
                "first BC subrow joined by episode_id/side/step_id; includes "
                "materialized would_ko features exactly as trained"
            ),
            "descendant_opt_attr_source": (
                "real cg.api descendant observation encoded without would_ko "
                "annotation in both offline builder and runtime"
            ),
        },
        "source": {
            "archives": source_archives,
            "episodes_selected": max_episodes,
            "episode_ids": selected_episode_ids,
            "bc_dataset_build_fingerprint": root_opt_attr.dataset_build_fingerprint,
            "bc_dataset_contract_sha256": root_opt_attr.dataset_contract_sha256,
        },
        "config": {
            "path": str(config_path),
            "seed": int(cfg.seed),
            "max_episodes": max_episodes,
            "max_groups": max_groups,
            "max_branches": max_branches,
            "trials": trials,
            "horizon": horizon,
            "gamma": gamma,
            "workers": 1,
            "self_aliases": sorted(aliases),
        },
        "semantics": {
            "real_state_source": "player-view observations from source replay",
            "hidden_opponent_deck_used": False,
            "synthetic_fill_allowed": False,
            "common_random_numbers": (
                "one determinization, one cg.api root search state, and one trial "
                "seed reused across all root branches in a group/trial"
            ),
            "engine_internal_rng": (
                "the public cg.api exposes no seed for its internal RNG; root "
                "comparisons share one search-state snapshot, but depth>1 output "
                "is not guaranteed byte-identical across separate processes"
            ),
            "rollout_policy": (
                "root enumerates legal bounded actions; descendants use seeded "
                "legal selection sampling with manual coin choices"
            ),
            "scalar_reward": "terminal(+1/-1) + (prizes_taken-prizes_lost)/6",
            "scalar_return": "backward discounted sum of scalar_reward",
            "ko_signal": "prizes taken after the action; engine-derived proxy",
            "missing_logprob": "has_*_logprob=false and value=0; NaN forbidden",
            "unmaterialized_root": (
                "skip the replay decision and record it in audit; a prospective "
                "root must join an existing real BC row and is never synthesized"
            ),
        },
        "rope_nd_axes": [
            {
                "name": "match_time",
                "field": "step_id",
                "unit": "real replay decision",
            },
            {
                "name": "rollout_depth",
                "field": "depth",
                "unit": (
                    "implicit replay context is depth 0; first simulated legal "
                    "action is depth 1"
                ),
            },
            {
                "name": "branch_action",
                "fields": ["branch_order", "sibling_index", "action_index"],
                "unit": "stable legal action order within parent",
            },
            {
                "name": "entity_zone_relation",
                "fields": ["src_pos", "tgt_pos", "verb", "entity_zone_relation_id"],
                "unit": (
                    "v1 exact packing: ((verb*512)+(src_pos+1))*512+(tgt_pos+1)"
                ),
            },
        ],
        "rope_nd_axis_order": [
            "match_time",
            "rollout_depth",
            "branch_action",
            "entity_zone_relation",
        ],
        "outputs": {
            "nodes": "prospective_nodes.npy",
            "branches": "prospective_branches.npy",
            "actions": "prospective_actions.npy",
            "groups": "prospective_groups.npy",
            "group_offsets": "prospective_group_offsets.npy",
            "episode_sides": "prospective_episode_sides.npy",
            "node_rows": len(nodes),
            "branch_rows": len(branches),
            "action_rows": len(actions),
            "group_rows": len(groups),
            "episode_side_rows": len(episode_sides),
            "node_dtype": nodes.dtype.descr,
            "branch_dtype": branches.dtype.descr,
            "group_dtype": groups.dtype.descr,
            "episode_side_dtype": episode_sides.dtype.descr,
            "action_dtype": actions.dtype.str,
            "node_itemsize": nodes.dtype.itemsize,
            "branch_itemsize": branches.dtype.itemsize,
        },
        "storage": {
            "version": "compact-sharded-v1",
            "group_order": (
                "source archive, ZIP member, real decision, trial_index"
            ),
            "node_order": "parent-before-child within each group tree",
            "parent_index": "int32 local node index; -1 denotes a root",
            "actions": (
                "flat int16 prospective_actions.npy addressed by "
                "action_offset/action_count"
            ),
            "group_offsets": (
                "uint64 node offsets, equal to groups.node_start plus terminal row"
            ),
            "failure_stage_codes": FAILURE_STAGE_CODES,
            "failure_type_code": "BLAKE2b uint16 of exception class name; zero is none",
            "determination_mode_codes": DETERMINATION_MODE_CODES,
            "resume_work_dir": str(work_dir),
            "flush_groups": flush_groups,
        },
        "audit": audit_counts,
        }
        if len(source_archives) == 1:
            contract["source"]["path"] = source_archives[0]["path"]
            contract["source"]["sha256"] = source_archives[0]["sha256"]
        fingerprint_source = json.dumps(
            contract, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        contract["fingerprint"] = hashlib.sha256(fingerprint_source).hexdigest()
        (staging / "prospective_manifest.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, out_dir)
        # Shards are necessary for crash-safe resume only until the atomic
        # publication succeeds. Keeping them afterwards would duplicate the
        # prospective corpus on disk.
        shutil.rmtree(work_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_config.json")
    parser.add_argument(
        "--zip",
        required=True,
        action="append",
        dest="zip_paths",
        help="Real replay ZIP; repeat for a multi-archive BC corpus",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-groups", type=int, default=32)
    parser.add_argument("--max-branches", type=int, default=8)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--flush-groups",
        type=int,
        default=32,
        help="Persist this many (decision, trial) trees per resumable shard",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(config_path=args.config)
    max_episodes = (
        int(args.max_episodes)
        if args.max_episodes is not None
        else int(cfg.max_episodes)
    )
    manifest = build(
        [Path(path) for path in args.zip_paths],
        Path(args.out),
        config_path=Path(args.config),
        max_episodes=max_episodes,
        max_groups=int(args.max_groups),
        max_branches=int(args.max_branches),
        trials=int(args.trials),
        horizon=int(args.horizon),
        gamma=float(args.gamma),
        flush_groups=int(args.flush_groups),
    )
    print(
        "[prospective] "
        f"groups={manifest['audit']['groups_emitted']} "
        f"branches={manifest['outputs']['branch_rows']} "
        f"nodes={manifest['outputs']['node_rows']} "
        f"fingerprint={manifest['fingerprint'][:12]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
