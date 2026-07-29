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


SCHEMA_VERSION = 1
PLANNER_VERSION = 1
ROOT_PARENT_ID = ""

NODE_DTYPE = np.dtype([
    ("episode_id", "U64"),
    ("side", "i4"),
    ("step_id", "i4"),
    ("group_id", "U64"),
    ("branch_id", "U64"),
    ("parent_branch_id", "U64"),
    ("has_parent", "bool"),
    ("depth", "i4"),
    ("trial_index", "i4"),
    ("trial_seed", "u8"),
    ("determination_id", "U64"),
    ("root_branch_order", "i4"),
    ("branch_order", "i4"),
    ("sibling_index", "i4"),
    ("action_index", "i4"),
    ("has_action_index", "bool"),
    ("action_json", "U2048"),
    ("subselection_count", "i4"),
    ("select_type", "i4"),
    ("select_context", "i4"),
    ("src_pos", "i4"),
    ("tgt_pos", "i4"),
    ("verb", "i4"),
    ("entity_zone_relation_id", "i4"),
    ("entity_zone_relation_json", "U2048"),
    ("behavior_selected", "bool"),
    ("has_behavior_logprob", "bool"),
    ("behavior_logprob", "f4"),
    ("has_reference_logprob", "bool"),
    ("reference_logprob", "f4"),
    ("valid", "bool"),
    ("failure_stage", "U48"),
    ("failure_type", "U64"),
    ("determination_mode", "U32"),
    ("determination_candidates", "i4"),
    ("terminal", "bool"),
    ("terminal_result", "i4"),
    ("win_signal", "i1"),
    ("loss_signal", "i1"),
    ("draw_signal", "i1"),
    ("ko_signal", "i1"),
    ("ko_against_signal", "i1"),
    ("prizes_taken", "i4"),
    ("prizes_lost", "i4"),
    ("reward_terminal", "f4"),
    ("reward_ko", "f4"),
    ("reward_prize", "f4"),
    ("scalar_reward", "f4"),
    ("scalar_return", "f4"),
    ("player_name", "U128"),
    ("opponent_name", "U128"),
    ("is_self", "bool"),
    ("outcome", "i1"),
])

BRANCH_DTYPE = np.dtype([
    ("episode_id", "U64"),
    ("side", "i4"),
    ("step_id", "i4"),
    ("group_id", "U64"),
    ("branch_id", "U64"),
    ("parent_branch_id", "U64"),
    ("has_parent", "bool"),
    ("depth", "i4"),
    ("branch_order", "i4"),
    ("sibling_index", "i4"),
    ("action_index", "i4"),
    ("has_action_index", "bool"),
    ("action_json", "U2048"),
    ("src_pos", "i4"),
    ("tgt_pos", "i4"),
    ("verb", "i4"),
    ("entity_zone_relation_id", "i4"),
    ("entity_zone_relation_json", "U2048"),
    ("behavior_selected", "bool"),
    ("has_behavior_logprob", "bool"),
    ("behavior_logprob", "f4"),
    ("has_reference_logprob", "bool"),
    ("reference_logprob", "f4"),
    ("requested_trials", "i4"),
    ("valid_trials", "i4"),
    ("failed_trials", "i4"),
    ("valid", "bool"),
    ("mean_scalar_return", "f4"),
    ("terminal_rate", "f4"),
    ("win_rate", "f4"),
    ("ko_rate", "f4"),
    ("expected_prizes_taken", "f4"),
    ("player_name", "U128"),
    ("opponent_name", "U128"),
    ("is_self", "bool"),
    ("outcome", "i1"),
])


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


def _stable_seed(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _entity_relation(
    observation: dict[str, Any],
    action: tuple[int, ...],
    encoder: TokenEncoder,
    own_deck: tuple[int, ...],
) -> tuple[int, int, int, int, str]:
    """Map a single legal option onto the trainer's public E/Z/R coordinate.

    Multi-select actions do not have one unambiguous source/target relation, so
    their scalar coordinate is the documented neutral sentinel while the full
    subselection remains losslessly available in ``action_json`` and the
    versioned component payload.
    """
    select = observation.get("select") or {}
    options = select.get("option") or []
    selected = [
        {"option_index": index, "option": options[index]}
        for index in action
        if 0 <= index < len(options)
    ]
    src_pos, tgt_pos, verb = -1, -1, 0
    coordinate_status = "multi_select_neutral"
    if len(action) == 1 and 0 <= action[0] < min(len(options), MAX_OPTIONS):
        encoded = encoder.encode(
            observation,
            picked=set(),
            self_deck=list(own_deck),
            tracker=None,
            ability_slots=None,
        )
        option_index = action[0]
        src_pos = int(encoded["opt_src_pos"][option_index])
        tgt_pos = int(encoded["opt_tgt_pos"][option_index])
        verb = int(encoded["opt_verb"][option_index])
        coordinate_status = "encoded_single_option"
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
    components = {
        "prospective_coord_schema_version": PROSPECTIVE_COORD_SCHEMA_VERSION,
        "select_type": int(select.get("type", -1) or 0),
        "select_context": int(select.get("context", -1) or 0),
        "src_pos": src_pos,
        "tgt_pos": tgt_pos,
        "verb": verb,
        "coordinate_status": coordinate_status,
        "selected": selected,
    }
    rendered = _canonical_json(components)
    return src_pos, tgt_pos, verb, relation_id, rendered


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
) -> dict[str, Any]:
    select = observation.get("select") or {}
    src_pos, tgt_pos, verb, relation_id, relation_json = _entity_relation(
        observation, action, encoder, decision.own_deck
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
        "action_json": _canonical_json(action),
        "subselection_count": len(action),
        "select_type": int(select.get("type", -1) or 0),
        "select_context": int(select.get("context", -1) or 0),
        "src_pos": src_pos,
        "tgt_pos": tgt_pos,
        "verb": verb,
        "entity_zone_relation_id": relation_id,
        "entity_zone_relation_json": relation_json,
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
) -> dict[str, Any]:
    select = observation.get("select") or {}
    src_pos, tgt_pos, verb, relation_id, relation_json = _entity_relation(
        observation, action, encoder, decision.own_deck
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
        "action_json": _canonical_json(action),
        "subselection_count": len(action),
        "select_type": int(select.get("type", -1) or 0),
        "select_context": int(select.get("context", -1) or 0),
        "src_pos": src_pos,
        "tgt_pos": tgt_pos,
        "verb": verb,
        "entity_zone_relation_id": relation_id,
        "entity_zone_relation_json": relation_json,
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


def _dicts_to_array(records: list[dict[str, Any]], dtype: np.dtype) -> np.ndarray:
    return np.array(
        [tuple(record[name] for name in dtype.names or ()) for record in records],
        dtype=dtype,
    )


def _branch_summaries(
    decisions_by_group: dict[str, RealDecision],
    root_actions: dict[tuple[str, int], tuple[int, ...]],
    nodes: list[dict[str, Any]],
    trials: int,
    encoder: TokenEncoder,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for node in nodes:
        if int(node["depth"]) == 1:
            grouped.setdefault(
                (str(node["group_id"]), int(node["root_branch_order"])), []
            ).append(node)
    summaries = []
    for (group_id, order), action in sorted(root_actions.items()):
        decision = decisions_by_group[group_id]
        records = grouped.get((group_id, order), [])
        valid = [record for record in records if record["valid"]]
        src_pos, tgt_pos, verb, relation_id, relation_json = _entity_relation(
            decision.observation, action, encoder, decision.own_deck
        )
        valid_count = len(valid)
        denominator = max(valid_count, 1)
        summaries.append({
            "episode_id": decision.episode_id,
            "side": decision.side,
            "step_id": decision.step_id,
            "group_id": group_id,
            "branch_id": _stable_hex(group_id, "root", order),
            "parent_branch_id": ROOT_PARENT_ID,
            "has_parent": False,
            "depth": 1,
            "branch_order": order,
            "sibling_index": order,
            "action_index": action[0] if len(action) == 1 else -1,
            "has_action_index": len(action) == 1,
            "action_json": _canonical_json(action),
            "src_pos": src_pos,
            "tgt_pos": tgt_pos,
            "verb": verb,
            "entity_zone_relation_id": relation_id,
            "entity_zone_relation_json": relation_json,
            "behavior_selected": action == decision.behavior_action,
            "has_behavior_logprob": False,
            "behavior_logprob": 0.0,
            "has_reference_logprob": False,
            "reference_logprob": 0.0,
            "requested_trials": trials,
            "valid_trials": valid_count,
            "failed_trials": trials - valid_count,
            "valid": bool(valid_count),
            "mean_scalar_return": sum(
                float(record["scalar_return"]) for record in valid
            ) / denominator,
            "terminal_rate": sum(bool(record["terminal"]) for record in valid) / denominator,
            "win_rate": sum(int(record["win_signal"]) for record in valid) / denominator,
            "ko_rate": sum(int(record["ko_signal"]) for record in valid) / denominator,
            "expected_prizes_taken": sum(
                int(record["prizes_taken"]) for record in valid
            ) / denominator,
            "player_name": decision.player_name,
            "opponent_name": decision.opponent_name,
            "is_self": decision.is_self,
            "outcome": decision.outcome,
        })
    return summaries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
) -> dict[str, Any]:
    cfg = load_config(config_path=str(config_path))
    if trials <= 0 or horizon <= 0 or max_branches < 2 or max_groups < 0:
        raise ValueError(
            "trials/horizon must be positive, max_branches >= 2, and "
            "max_groups >= 0 (0 means all branchable decisions)"
        )
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    encoder = TokenEncoder(get_card_table())
    aliases = frozenset(cfg.bc_self_aliases)
    node_records: list[dict[str, Any]] = []
    decisions_by_group: dict[str, RealDecision] = {}
    root_actions: dict[tuple[str, int], tuple[int, ...]] = {}
    audit_counts: dict[str, int] = {
        "episodes_read": 0,
        "groups_emitted": 0,
        "determination_failures": 0,
        "synthetic_fill_rejections": 0,
        "search_failures": 0,
    }
    selected_episode_ids: list[str] = []

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
    episode_refs = (
        episode_refs[:max_episodes] if max_episodes else episode_refs
    )
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
        audit_counts["episodes_read"] += 1
        episode_groups = 0
        for decision in _iter_real_decisions(
            episode,
            episode_id,
            both_sides=bool(cfg.bc_both_sides),
            self_aliases=aliases,
        ):
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
            group_id = _stable_hex(
                SCHEMA_VERSION,
                decision.episode_id,
                decision.side,
                decision.step_id,
            )
            decisions_by_group[group_id] = decision
            for order, action in enumerate(actions):
                root_actions[(group_id, order)] = action

            for trial_index in range(trials):
                trial_seed = _stable_seed(
                    cfg.seed, group_id, trial_index
                )
                determination_audit: dict[str, int] = {}
                try:
                    determination = _determinize(
                        decision.observation,
                        list(decision.own_deck),
                        random.Random(trial_seed),
                        encoder,
                        audit=determination_audit,
                    )
                except Exception as exc:
                    audit_counts["determination_failures"] += 1
                    for order, action in enumerate(actions):
                        node_records.append(_failed_node(
                            decision,
                            group_id,
                            _stable_hex(group_id, trial_index, order, 1),
                            encoder,
                            trial_index=trial_index,
                            trial_seed=trial_seed,
                            root_branch_order=order,
                            action=action,
                            observation=decision.observation,
                            failure_stage="determinize",
                            failure_type=type(exc).__name__,
                            determination_mode="failed",
                            determination_candidates=0,
                        ))
                    continue
                if (
                    int(determination_audit.get("opponent_synthetic_cards", 0))
                    or int(determination_audit.get("self_synthetic_cards", 0))
                ):
                    audit_counts["synthetic_fill_rejections"] += 1
                    for order, action in enumerate(actions):
                        node_records.append(_failed_node(
                            decision,
                            group_id,
                            _stable_hex(group_id, trial_index, order, 1),
                            encoder,
                            trial_index=trial_index,
                            trial_seed=trial_seed,
                            root_branch_order=order,
                            action=action,
                            observation=decision.observation,
                            failure_stage="determinize",
                            failure_type="SyntheticFillRejected",
                            determination_mode="rejected",
                            determination_candidates=0,
                        ))
                    continue
                mode, candidate_count = _determination_mode(
                    determination_audit
                )
                from cg import api
                try:
                    root_state = api.search_begin(
                        api.to_observation_class(decision.observation),
                        **determination,
                        manual_coin=True,
                    )
                except Exception as exc:
                    audit_counts["search_failures"] += 1
                    for order, action in enumerate(actions):
                        node_records.append(_failed_node(
                            decision,
                            group_id,
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
                        ))
                    continue
                try:
                    for order, action in enumerate(actions):
                        records = _rollout_branch(
                            decision,
                            group_id,
                            action,
                            order,
                            encoder,
                            trial_index=trial_index,
                            trial_seed=trial_seed,
                            root_state=root_state,
                            determination_mode=mode,
                            determination_candidates=candidate_count,
                            horizon=horizon,
                            gamma=gamma,
                        )
                        audit_counts["search_failures"] += sum(
                            not record["valid"] for record in records
                        )
                        node_records.extend(records)
                finally:
                    try:
                        api.search_release(root_state.searchId)
                    except Exception:
                        audit_counts["search_failures"] += 1
                    try:
                        api.search_end()
                    except Exception:
                        audit_counts["search_failures"] += 1
            audit_counts["groups_emitted"] += 1
            episode_groups += 1
            if (
                episode_group_quota is not None
                and episode_groups >= episode_group_quota
            ):
                break

    branch_records = _branch_summaries(
        decisions_by_group, root_actions, node_records, trials, encoder
    )
    nodes = _dicts_to_array(node_records, NODE_DTYPE)
    branches = _dicts_to_array(branch_records, BRANCH_DTYPE)
    if len(branches) == 0 or int(np.count_nonzero(branches["valid"])) == 0:
        raise RuntimeError("prospective corpus produced no valid counterfactual branch")
    for array, name in ((nodes, "nodes"), (branches, "branches")):
        for field in array.dtype.names or ():
            if np.issubdtype(array.dtype[field], np.floating):
                if not np.isfinite(array[field]).all():
                    raise RuntimeError(f"{name}.{field} contains NaN or infinity")

    contract = {
        "schema_version": SCHEMA_VERSION,
        "planner_version": PLANNER_VERSION,
        "prospective_coord_schema_version": PROSPECTIVE_COORD_SCHEMA_VERSION,
        "prospective_coordinate_schema": prospective_coordinate_schema(),
        "source": {
            "archives": source_archives,
            "episodes_selected": max_episodes,
            "episode_ids": selected_episode_ids,
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
            "node_rows": len(nodes),
            "branch_rows": len(branches),
            "node_dtype": nodes.dtype.descr,
            "branch_dtype": branches.dtype.descr,
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

    out_dir = out_dir.resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{out_dir.name}.", dir=out_dir.parent
    ))
    try:
        np.save(staging / "prospective_nodes.npy", nodes, allow_pickle=False)
        np.save(staging / "prospective_branches.npy", branches, allow_pickle=False)
        (staging / "prospective_manifest.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if out_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite existing prospective dataset: {out_dir}"
            )
        os.replace(staging, out_dir)
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
