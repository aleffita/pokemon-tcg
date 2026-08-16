"""Bounded sibling-fiber GRPO for AR-020.

Each fiber starts from the same frozen-root recurrent base and common random
seed, but takes a distinct legal action at the first logical decision. The
first probe applies relative credit only to that branching conditional action;
continuation decisions remain provenance evidence and receive no gradient.
"""

from __future__ import annotations

import math
import importlib.util
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from rl.encoder.encoding import SUBMIT_ACTION
from rl.env.env import CabtEnv
from rl.policy_infer_torch import TORCH_INFERENCE_FORMAT
from scripts.rl.ppo_micro_update import sha256_file
from scripts.rl.trajectory_group_grpo import (
    _learner_logprobs,
    _masked_distribution,
    _parameter_delta,
    flatten_provenance_bundle,
    normalize_group_returns,
    recompute_logprobs_by_decision,
    save_grpo_candidate_checkpoint,
    trajectory_from_bundle,
)
from scripts.rl.trajectory_probe import (
    APPROVED_STAGE4_ROOT_SHA256,
    DateBoundEncoder,
    _StatefulMirror,
    as_model_input,
    collect_episode,
    digest_tensor,
    initial_memory,
    sha256_bytes,
)


SIBLING_FORMAT = "ptcg-stage4-sibling-fiber-grpo-v1"
DEFAULT_OUTPUT = Path("experiments/autoresearch/AR-020")


def _finite(value: torch.Tensor | float) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    return math.isfinite(float(value))


def _prospective_auxiliary_examples(
    trajectory_groups: list[list[dict[str, Any]]],
) -> tuple[list[tuple[dict[str, Any], dict[str, float]]], list[dict[str, Any]]]:
    """Derive dense future targets and one first-state deck target per fiber group."""
    prospective: list[tuple[dict[str, Any], dict[str, float]]] = []
    deck_examples: list[dict[str, Any]] = []
    for group in trajectory_groups:
        if group and group[0]["decisions"]:
            deck_examples.append(group[0]["decisions"][0][0])
        for trajectory in group:
            decisions = trajectory["decisions"]
            samples = [decision[0] for decision in decisions]
            if not samples:
                continue
            scalars = [sample["model_input"]["cls_scalars"][0].to(torch.float32) for sample in samples]
            turns = [int(round(float(value[0].item()) * 50.0)) for value in scalars]
            own_prizes = [float(value[9].item()) * 6.0 for value in scalars]
            opp_prizes = [float(value[10].item()) * 6.0 for value in scalars]
            terminal_return = float(trajectory["terminal_return"])
            for decision_index, sample in enumerate(samples):
                end = decision_index
                while end + 1 < len(samples) and turns[end + 1] == turns[decision_index]:
                    end += 1
                own_taken_turn = own_prizes[decision_index] - own_prizes[end]
                opp_taken_turn = opp_prizes[decision_index] - opp_prizes[end]
                future_prize_margin = (
                    own_prizes[decision_index]
                    - own_prizes[-1]
                    - opp_prizes[decision_index]
                    + opp_prizes[-1]
                ) / 6.0
                prospective.append(
                    (
                        sample,
                        {
                            "ko": float(abs(own_taken_turn) + abs(opp_taken_turn) > 0.5),
                            "prize": float(own_taken_turn - opp_taken_turn),
                            "terminal": float(decision_index == len(samples) - 1),
                            "return": float(terminal_return + future_prize_margin),
                        },
                    )
                )
    return prospective, deck_examples


def _batch_aux_samples(samples: list[dict[str, Any]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    keys = tuple(samples[0]["model_input"])
    model_input = {
        key: torch.cat([sample["model_input"][key].detach().to("cpu") for sample in samples], dim=0)
        for key in keys
    }
    memories = torch.cat(
        [sample["memory_input"].detach().to(device="cpu", dtype=torch.float32) for sample in samples],
        dim=0,
    )
    return model_input, memories


def _backward_prospective_auxiliary(
    model: torch.nn.Module,
    examples: list[tuple[dict[str, Any], dict[str, float]]],
    *,
    weight: float,
    batch_size: int,
) -> dict[str, float]:
    if weight <= 0.0 or not examples:
        return {"loss": 0.0, "batches": 0, "examples": len(examples)}
    batch_count = math.ceil(len(examples) / batch_size)
    component_sums = {"ko": 0.0, "prize": 0.0, "terminal": 0.0, "return": 0.0}
    total_loss = 0.0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        model_input, memories = _batch_aux_samples([item[0] for item in batch])
        predictions = model.aux_predictions(model_input, memory_in=memories)
        targets = {
            name: torch.as_tensor([item[1][name] for item in batch], dtype=torch.float32)
            for name in component_sums
        }
        losses = {
            "ko": torch.nn.functional.binary_cross_entropy_with_logits(
                predictions["ko_logit"], targets["ko"]
            ),
            "prize": torch.nn.functional.smooth_l1_loss(
                predictions["prize_pred"], targets["prize"]
            ),
            "terminal": torch.nn.functional.binary_cross_entropy_with_logits(
                predictions["terminal_logit"], targets["terminal"]
            ),
            "return": torch.nn.functional.smooth_l1_loss(
                predictions["return_pred"], targets["return"]
            ),
        }
        combined = 0.5 * losses["ko"] + 0.5 * losses["prize"] + 0.25 * losses["terminal"] + losses["return"]
        ((weight / batch_count) * combined).backward()
        total_loss += float(combined.detach().item())
        for name, value in losses.items():
            component_sums[name] += float(value.detach().item())
    return {
        "loss": total_loss / batch_count,
        "batches": batch_count,
        "examples": len(examples),
        **{f"{name}_loss": value / batch_count for name, value in component_sums.items()},
    }


def _backward_deck_reconstruction(
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    *,
    weight: float,
    batch_size: int,
    epoch: int,
) -> dict[str, float]:
    if weight <= 0.0 or not samples:
        return {"loss": 0.0, "batches": 0, "examples": len(samples)}
    batch_count = math.ceil(len(samples) / batch_size)
    total_loss = 0.0
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        model_input, memories = _batch_aux_samples(batch)
        deck_ids = model_input["self_deck_id"].clone()
        valid = deck_ids > 0
        columns = torch.arange(deck_ids.shape[1]).unsqueeze(0)
        masked = valid & ((columns + epoch) % 5 == 0)
        model_input["self_deck_id"] = deck_ids.masked_fill(masked, 0)
        if "self_deck_meta_bucket" in model_input:
            model_input["self_deck_meta_bucket"] = model_input[
                "self_deck_meta_bucket"
            ].masked_fill(masked, 0)
        logits = model.deck_card_logits(model_input, memory_in=memories)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_loss = -log_probs.gather(1, deck_ids.clamp_min(0))
        per_example = (token_loss * valid).sum(1) / valid.sum(1).clamp_min(1)
        loss = per_example.mean()
        ((weight / batch_count) * loss).backward()
        total_loss += float(loss.detach().item())
    return {
        "loss": total_loss / batch_count,
        "batches": batch_count,
        "examples": len(samples),
        "masked_fraction": 0.2,
    }


def _input_digests(model_input: dict[str, torch.Tensor]) -> list[dict[str, str]]:
    return [
        {"name": name, "sha256": digest_tensor(value)}
        for name, value in model_input.items()
    ]


def _base_signature(encoded: dict[str, np.ndarray], model_input: dict[str, torch.Tensor], memory: torch.Tensor) -> dict[str, Any]:
    mask = np.asarray(encoded["action_mask"], dtype=np.float32).reshape(-1)
    return {
        "action_mask_sha256": sha256_bytes(mask.tobytes(order="C")),
        "memory_input_sha256": digest_tensor(memory),
        "model_input_digests": _input_digests(model_input),
    }


def _branch_candidates(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    requested_k: int,
    generator: torch.Generator,
    uniform_mix: float = 0.0,
) -> list[int]:
    """Choose a dynamic sibling set from one real decision mask.

    ``requested_k`` is only a cap.  The effective K is the number of distinct
    non-submit legal actions available at this exact base, or all legal
    actions when the fiber is smaller.  Submit is excluded when there are at
    least two ordinary choices because it is a control action, not a play.
    """
    if requested_k < 2:
        raise ValueError("requested sibling K must be at least two")
    if not 0.0 <= uniform_mix <= 1.0 or not math.isfinite(uniform_mix):
        raise ValueError("uniform sibling mix must be finite and in [0, 1]")
    flat_mask = action_mask.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    flat_logits = logits.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    legal = [int(index) for index, value in enumerate(flat_mask.tolist()) if value >= 0.5]
    non_submit = [index for index in legal if index != SUBMIT_ACTION]
    pool = non_submit if len(non_submit) >= 2 else legal
    effective_k = min(requested_k, len(pool))
    if effective_k < 2:
        return []
    distribution = _masked_distribution(
        flat_logits.reshape(1, -1),
        flat_mask.reshape(1, -1),
    )
    pool_mask = torch.zeros_like(flat_mask, dtype=torch.bool)
    pool_mask[pool] = True
    pool_probs = distribution.probs.reshape(-1).masked_fill(~pool_mask, 0.0)
    pool_probs = pool_probs / pool_probs.sum()
    if uniform_mix:
        uniform_probs = pool_mask.to(dtype=torch.float32)
        uniform_probs = uniform_probs / uniform_probs.sum()
        pool_probs = (1.0 - uniform_mix) * pool_probs + uniform_mix * uniform_probs
    actions = torch.multinomial(
        pool_probs,
        num_samples=effective_k,
        replacement=False,
        generator=generator,
    )
    result = [int(value) for value in actions.tolist()]
    if len(set(result)) != effective_k or any(item not in pool for item in result):
        raise AssertionError("dynamic sibling selection returned illegal or duplicate actions")
    return result


class _ExternalAgentOpponent:
    """Adapter for a repository-local tournament agent callable.

    Public tournament agents consume the same raw CABT observation dictionary
    as the engine. They are deterministic/stateful at the module level rather
    than recurrent PyTorch policies, so the adapter deliberately does not
    invent a second learned memory lane.
    """

    def __init__(self, agent: Callable[[dict[str, Any]], list[int]]) -> None:
        self.agent = agent

    def reset_episode(self, _episode_id: str) -> None:
        # The module is freshly loaded for each probe seed. The raw observation
        # contains the complete public state needed by these agents.
        return None

    def set_side(self, _side: int) -> None:
        return None

    def on_terminal(self, _agent_return: float) -> None:
        return None

    def __call__(self, raw_obs: dict[str, Any], _rng: Any, **_) -> list[int]:
        picks = self.agent(raw_obs)
        if not isinstance(picks, (list, tuple)):
            raise TypeError(f"external opponent returned {type(picks).__name__}, expected a list")
        return [int(item) for item in picks]


def load_external_opponent(path: str | Path) -> _ExternalAgentOpponent:
    """Load one local tournament agent without purging the learner modules."""
    agent_path = Path(path).resolve()
    if not agent_path.is_file():
        raise FileNotFoundError(f"external opponent main.py not found: {agent_path}")
    module_name = f"ptcg_external_{abs(hash(str(agent_path)))}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, agent_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load external opponent: {agent_path}")
    module = importlib.util.module_from_spec(spec)
    old_cwd = os.getcwd()
    agent_dir = str(agent_path.parent)
    sys.path.insert(0, agent_dir)
    try:
        os.chdir(agent_dir)
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
        sys.path.pop(0)
    opponent = getattr(module, "agent", None)
    if not callable(opponent):
        raise AttributeError(f"external opponent has no callable agent: {agent_path}")
    return _ExternalAgentOpponent(opponent)


def _reset_opponent(opponent: Any, episode_id: str) -> None:
    callback = getattr(opponent, "reset_episode", None)
    if callback is not None:
        callback(episode_id)


def _set_opponent_side(opponent: Any, side: int) -> None:
    callback = getattr(opponent, "set_side", None)
    if callback is not None:
        callback(side)


def _probe_branch_fibers(
    *,
    model: torch.nn.Module,
    encoder: DateBoundEncoder,
    deck: list[int],
    opponent_deck: list[int] | None,
    seed: int,
    games: int,
    opponent_factory: Callable[[], Any] | None = None,
    branch_uniform_mix: float = 0.0,
) -> tuple[list[int], dict[str, Any], CabtEnv, dict[str, np.ndarray], dict[str, Any]]:
    """Find a real in-game branch and keep its live env as an exact snapshot."""
    for seed_offset in range(32):
        probe_seed = seed + seed_offset
        opponent = (
            opponent_factory()
            if opponent_factory is not None
            else _StatefulMirror(model, encoder, np.random.default_rng(probe_seed + 1000))
        )
        env = CabtEnv(
            agent_deck=deck,
            opponent_deck=opponent_deck or deck,
            opponent_fn=opponent,
            encoder=encoder,
            seed=probe_seed,
            max_steps=4000,
            reset_hook=lambda _attempt, opponent=opponent: _reset_opponent(opponent, "branch-probe"),
        )
        keep_env = False
        try:
            encoded, reset_info = env.reset(seed=probe_seed)
            _set_opponent_side(opponent, 1 - int(reset_info["agent_index"]))
            memory = initial_memory(model)
            decision_memory = memory
            decision_index = 0
            substep = 0
            picked: set[int] = set()
            prefix_actions: list[int] = []
            prefix_generator = torch.Generator(device="cpu").manual_seed(probe_seed + 7000)
            sibling_generator = torch.Generator(device="cpu").manual_seed(probe_seed + 9000)
            for _scout_step in range(512):
                model_input = as_model_input(encoded, encoder.int_keys)
                with torch.inference_mode():
                    logits, _value, memory_out = model.logits_value(
                        model_input, memory_in=decision_memory
                    )
                mask = torch.as_tensor(encoded["action_mask"], dtype=torch.float32).reshape(1, -1)
                candidates = _branch_candidates(
                    logits,
                    mask,
                    games,
                    sibling_generator,
                    uniform_mix=branch_uniform_mix,
                )
                if substep == 0 and len(candidates) >= 2:
                    legal = [
                        int(index)
                        for index, is_legal in enumerate(mask.reshape(-1).tolist())
                        if float(is_legal) >= 0.5
                    ]
                    base = _base_signature(encoded, model_input, decision_memory)
                    base.update(
                        {
                            "requested_seed": seed,
                            "probe_seed": probe_seed,
                            "probe_seed_offset": seed_offset,
                            "agent_side": int(reset_info["agent_index"]),
                            "branch_decision_index": decision_index,
                            "prefix_action_count": len(prefix_actions),
                            "prefix_actions_sha256": sha256_bytes(
                                ",".join(str(item) for item in prefix_actions).encode("ascii")
                            ),
                            "legal_action_count": len(legal),
                            "legal_actions": legal,
                            "requested_group_size": games,
                            "effective_group_size": len(candidates),
                            "branch_candidates": candidates,
                            "branch_uniform_mix": branch_uniform_mix,
                        }
                    )
                    keep_env = True
                    return candidates, base, env, encoded, {
                        **reset_info,
                        "branch_memory": decision_memory.detach().to(dtype=torch.float32).clone(),
                    }
                action = int(torch.multinomial(
                    _masked_distribution(logits, mask).probs.reshape(-1),
                    num_samples=1,
                    generator=prefix_generator,
                ).item())
                if float(mask.reshape(-1)[action].item()) < 0.5:
                    raise AssertionError("scout selected an illegal action")
                prefix_actions.append(action)
                if action != SUBMIT_ACTION:
                    picked.add(action)
                select = getattr(env, "_obs", {}).get("select") or {}
                max_count = int(select.get("maxCount", 1))
                next_encoded, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    break
                encoded = next_encoded
                decision_complete = action == SUBMIT_ACTION or len(picked) >= max_count
                if decision_complete:
                    decision_memory = memory_out.detach().to(dtype=torch.float32).clone()
                    memory = decision_memory
                    picked = set()
                    decision_index += 1
                    substep = 0
                else:
                    substep += 1
            # This real episode had no sufficiently branching state.
            continue
        finally:
            if not keep_env:
                env.close()
    raise ValueError(
        "bounded seed search found no base with at least two distinct legal fibers"
    )


def _start_forked_fiber(
    *,
    env: CabtEnv,
    model: torch.nn.Module,
    encoder: DateBoundEncoder,
    episode_id: str,
    action: int,
    reset_seed: int,
    deck_content_hash: str,
    deck_source_file_hash: str,
    model_hash: str,
    base_observation: dict[str, np.ndarray],
    base_reset_info: dict[str, Any],
    opponent_mode: str,
) -> tuple[int, int]:
    """Start one exact-snapshot continuation and return ``(pid, read_fd)``."""
    if not hasattr(os, "fork"):
        raise RuntimeError("AR-020 exact sibling snapshots require os.fork on this platform")
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        payload: dict[str, Any]
        try:
            bundle: list[dict[str, Any]] = []
            action_generator = torch.Generator(device="cpu").manual_seed(reset_seed + 7000)
            opponent = env.opponent_fn
            rows = collect_episode(
                env,
                model,
                encoder,
                episode_id,
                opponent_mode,
                reset_seed,
                deck_content_hash,
                deck_source_file_hash,
                model_hash,
                action_generator,
                bundle,
                on_episode_reset=lambda agent_side: _set_opponent_side(opponent, 1 - agent_side),
                action_overrides={(0, 0): action},
                initial_observation=base_observation,
                initial_reset_info=base_reset_info,
                initial_memory_state=base_reset_info["branch_memory"],
            )
            payload = {"ok": True, "rows": rows, "bundle": bundle}
        except BaseException as exc:  # serialize the child failure for the coordinator
            payload = {
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        try:
            with os.fdopen(write_fd, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            os._exit(0)
    os.close(write_fd)
    return pid, read_fd


def _finish_forked_fiber(pid: int, read_fd: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Block on one already-running continuation and reap it exactly once."""
    try:
        with os.fdopen(read_fd, "rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        os.waitpid(pid, 0)
        raise RuntimeError(f"forked sibling worker returned no payload: {exc}") from exc
    _, status = os.waitpid(pid, 0)
    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
        raise RuntimeError(f"forked sibling worker exited abnormally: status={status}")
    if not payload.get("ok"):
        raise RuntimeError(
            f"forked sibling worker failed: {payload.get('error')}\n{payload.get('traceback', '')}"
        )
    return payload["rows"], payload["bundle"]


def _collect_forked_fiber(
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Backward-compatible synchronous wrapper around the split fork API."""
    pid, read_fd = _start_forked_fiber(**kwargs)
    return _finish_forked_fiber(pid, read_fd)


def _collect_forked_fibers(
    *,
    actions: list[int],
    episode_ids: list[str],
    **shared: Any,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Run all sibling futures concurrently from the same copy-on-write base."""
    if len(actions) != len(episode_ids):
        raise ValueError("parallel sibling actions and episode ids must align")
    jobs: list[tuple[int, int]] = []
    try:
        for action, episode_id in zip(actions, episode_ids, strict=True):
            jobs.append(
                _start_forked_fiber(
                    action=action,
                    episode_id=episode_id,
                    **shared,
                )
            )
    except BaseException:
        for pid, read_fd in jobs:
            try:
                os.close(read_fd)
            except OSError:
                pass
            os.waitpid(pid, 0)
        raise

    results: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    errors: list[BaseException] = []
    for pid, read_fd in jobs:
        try:
            results.append(_finish_forked_fiber(pid, read_fd))
        except BaseException as exc:
            errors.append(exc)
    if errors:
        raise RuntimeError(f"parallel sibling collection failed: {errors[0]}") from errors[0]
    return results


def collect_sibling_fiber_group(
    *,
    model: torch.nn.Module,
    encoder: DateBoundEncoder,
    deck: list[int],
    deck_content_hash: str,
    deck_source_file_hash: str,
    model_hash: str,
    opponent_deck: list[int] | None = None,
    opponent_deck_content_hash: str | None = None,
    opponent_deck_source_file_hash: str | None = None,
    games: int = 4,
    seed: int = 20020,
    episode_prefix: str | None = None,
    opponent_factory: Callable[[], Any] | None = None,
    opponent_agent_path: str | None = None,
    opponent_mode: str | None = None,
    branch_uniform_mix: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect K same-base forced-fiber continuations with common randomness."""
    if games < 2:
        raise ValueError("AR-020 requested K cap must be at least two")
    if not 0.0 <= branch_uniform_mix <= 1.0 or not math.isfinite(branch_uniform_mix):
        raise ValueError("branch uniform mix must be finite and in [0, 1]")
    actions, base, base_env, base_observation, base_reset_info = _probe_branch_fibers(
        model=model,
        encoder=encoder,
        deck=deck,
        opponent_deck=opponent_deck,
        seed=seed,
        games=games,
        opponent_factory=opponent_factory,
        branch_uniform_mix=branch_uniform_mix,
    )
    collection_seed = int(base["probe_seed"])
    resolved_opponent_mode = opponent_mode or (
        "current_vs_external_policy_true_recurrent"
        if opponent_factory is not None
        else "current_vs_current_true_recurrent"
    )
    trajectories: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        episode_ids = [
            (
                f"{episode_prefix}-fiber-{fiber_index:03d}"
                if episode_prefix
                else f"sibling-fiber-{fiber_index:03d}"
            )
            for fiber_index in range(len(actions))
        ]
        fiber_payloads = _collect_forked_fibers(
            actions=actions,
            episode_ids=episode_ids,
            env=base_env,
            model=model,
            encoder=encoder,
            reset_seed=collection_seed,
            deck_content_hash=deck_content_hash,
            deck_source_file_hash=deck_source_file_hash,
            model_hash=model_hash,
            base_observation=base_observation,
            base_reset_info=base_reset_info,
            opponent_mode=resolved_opponent_mode,
        )
        for fiber_index, (action, payload) in enumerate(zip(actions, fiber_payloads, strict=True)):
            rows, bundle = payload
            trajectory = trajectory_from_bundle(rows, bundle)
            observed_base = {
                "action_mask_sha256": rows[0]["legal_action_mask_digest"],
                "memory_input_sha256": rows[0]["memory_input_digest"],
                "model_input_digests": rows[0]["model_input_digests"],
            }
            if observed_base != {
                key: base[key]
                for key in ("action_mask_sha256", "memory_input_sha256", "model_input_digests")
            }:
                raise ValueError(f"fiber {fiber_index} did not start from the shared base state")
            if int(rows[0]["action"]) != int(action):
                raise ValueError(f"fiber {fiber_index} override was not applied")
            trajectory["branch"] = {
                "decision_index": 0,
                "substep": 0,
                "action": int(action),
                "action_mask_sha256": rows[0]["legal_action_mask_digest"],
                "behavior_logprob": float(bundle[0]["behavior_logprob"]),
            }
            trajectory["branch_base"] = dict(base)
            trajectory["agent_side"] = int(rows[0]["side"])
            trajectories.append(trajectory)
    finally:
        base_env.close()
    effective_games = len(trajectories)
    if len({int(item["branch"]["action"]) for item in trajectories}) != effective_games:
        raise AssertionError("sibling fibers must use distinct branching actions")
    elapsed = time.perf_counter() - started
    decisions = sum(int(item["logical_decisions"]) for item in trajectories)
    substeps = sum(int(item["substeps"]) for item in trajectories)
    returns = [float(item["terminal_return"]) for item in trajectories]
    return trajectories, {
        "requested_games": games,
        "games": effective_games,
        "requested_seed": seed,
        "collection_seed": collection_seed,
        "probe_seed_offset": int(base["probe_seed_offset"]),
        "collection_seconds": elapsed,
        "parallel_fibers": True,
        "parallel_workers": effective_games,
        "games_per_second": effective_games / elapsed if elapsed else None,
        "logical_decisions": decisions,
        "substeps": substeps,
        "decisions_per_second": decisions / elapsed if elapsed else None,
        "substeps_per_second": substeps / elapsed if elapsed else None,
        "returns": returns,
        "branch_uniform_mix": branch_uniform_mix,
        "branch_actions": actions,
        "effective_group_size": effective_games,
        "branch_base": base,
        "agent_deck_content_sha256": deck_content_hash,
        "agent_deck_source_file_sha256": deck_source_file_hash,
        "opponent_deck_content_sha256": opponent_deck_content_hash,
        "opponent_deck_source_file_sha256": opponent_deck_source_file_hash,
        "opponent_agent_path": opponent_agent_path,
        "opponent_mode": resolved_opponent_mode,
        "trajectory_summaries": [
            {
                "episode_id": item["episode_id"],
                "terminal_return": item["terminal_return"],
                "branch_action": item["branch"]["action"],
                "logical_decisions": item["logical_decisions"],
                "substeps": item["substeps"],
            }
            for item in trajectories
        ],
    }


def _validate_group(trajectories: list[dict[str, Any]]) -> None:
    if len(trajectories) < 2:
        raise ValueError("sibling-fiber GRPO requires at least two fibers")
    base_keys = ("action_mask_sha256", "memory_input_sha256", "model_input_digests")
    reference = trajectories[0].get("branch_base")
    if not isinstance(reference, dict):
        raise ValueError("sibling fiber is missing branch base provenance")
    for trajectory in trajectories:
        branch = trajectory.get("branch")
        base = trajectory.get("branch_base")
        if not isinstance(branch, dict) or not isinstance(base, dict):
            raise ValueError("sibling fiber is missing branch metadata")
        if (branch.get("decision_index"), branch.get("substep")) != (0, 0):
            raise ValueError("sibling credit must target the first logical-action substep")
        if not bool(trajectory.get("decisions")) or not trajectory["decisions"][0]:
            raise ValueError("sibling fiber has no branching decision")
        sample = trajectory["decisions"][0][0]
        if int(sample["action"]) != int(branch["action"]):
            raise ValueError("branch metadata does not match the retained sample")
        if float(sample["action_mask"][int(sample["action"])].item()) < 0.5:
            raise ValueError("sibling branch action is illegal")
        if any(base.get(key) != reference.get(key) for key in base_keys):
            raise ValueError("sibling fibers do not share one base state")
    if len({int(item["branch"]["action"]) for item in trajectories}) != len(trajectories):
        raise ValueError("sibling fibers must use distinct branch actions")


def sibling_fiber_grpo_update(
    model: torch.nn.Module,
    root_reference: torch.nn.Module,
    trajectories: list[dict[str, Any]],
    *,
    clip_epsilon: float = 0.2,
    learning_rate: float = 1e-5,
    advantage_epsilon: float = 1e-8,
    credit_scope: str = "branch_and_continuation",
    continuation_discount: float = 0.97,
) -> dict[str, Any]:
    """Apply relative terminal credit at the branch and along its future.

    ``branch_and_continuation`` keeps the sibling comparison at the branch but
    lets the terminal outcome train the subsequent logical decisions too. The
    discount is a credit horizon, not a simulator horizon: every rollout still
    runs to terminal completion.
    """
    _validate_group(trajectories)
    if not 0.0 < clip_epsilon < 1.0 or not math.isfinite(clip_epsilon):
        raise ValueError("clip_epsilon must be finite and between zero and one")
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    if credit_scope not in {"branch_only", "branch_and_continuation"}:
        raise ValueError("credit_scope must be branch_only or branch_and_continuation")
    if not 0.0 < continuation_discount <= 1.0 or not math.isfinite(continuation_discount):
        raise ValueError("continuation_discount must be finite and in (0, 1]")
    returns = [float(item["terminal_return"]) for item in trajectories]
    advantages, group_stats = normalize_group_returns(returns, epsilon=advantage_epsilon)
    learner, behavior, decision_mapping, _substep_mapping = recompute_logprobs_by_decision(
        model, trajectories
    )
    branch_mask_values: list[bool] = []
    discount_values: list[float] = []
    for trajectory in trajectories:
        for decision_index, _decision in enumerate(trajectory["decisions"]):
            is_branch = decision_index == 0
            branch_mask_values.append(is_branch)
            discount_values.append(
                1.0
                if is_branch or credit_scope == "branch_only"
                else continuation_discount**decision_index
            )
    branch_mask = torch.as_tensor(branch_mask_values, dtype=torch.bool)
    if not torch.equal(decision_mapping, torch.cat([
        torch.full((len(item["decisions"]),), index, dtype=torch.long)
        for index, item in enumerate(trajectories)
    ])):
        raise AssertionError("sibling logical decision mapping changed during recomputation")
    credit = advantages[decision_mapping] * torch.as_tensor(discount_values, dtype=torch.float32)
    if credit_scope == "branch_only":
        credit = credit.masked_fill(~branch_mask, 0.0)
    ratio = torch.exp(learner - behavior.detach())
    if not _finite(ratio):
        raise ValueError("sibling-fiber importance ratios are non-finite")
    active = credit != 0.0
    if not bool(active.any().item()):
        raise ValueError("sibling update has no credited logical decisions")
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratio * credit.detach(), clipped * credit.detach())
    policy_loss = -surrogate[active].mean()
    zero_variance = bool(group_stats["zero_variance"])
    gradient_norm = torch.tensor(0.0)
    started = time.perf_counter()
    if zero_variance:
        loss = policy_loss.detach() * 0.0
        optimizer_steps = 0
    else:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        optimizer.zero_grad(set_to_none=True)
        loss = policy_loss
        if not _finite(loss):
            raise ValueError("sibling-fiber policy loss is non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not _finite(gradient_norm):
            raise ValueError("sibling-fiber gradient norm is non-finite")
        optimizer.step()
        model.eval()
        optimizer_steps = 1
    update_seconds = time.perf_counter() - started
    metrics: dict[str, Any] = {
        "algorithm": "sibling_fiber_grpo",
        "precision": "FP32",
        "policy_only": True,
        "branch_only_credit": credit_scope == "branch_only",
        "continuation_credit": credit_scope == "branch_and_continuation",
        "value_loss": 0.0,
        "optimizer_steps": optimizer_steps,
        "update_seconds": update_seconds,
        "group_size": len(trajectories),
        "zero_variance_group": zero_variance,
        "return_mean": group_stats["return_mean"],
        "return_std": group_stats["return_std"],
        "credit_scope": credit_scope,
        "continuation_discount": continuation_discount,
        "logical_decisions": int(learner.numel()),
        "branch_logical_decisions": int(branch_mask.sum().item()),
        "continuation_logical_decisions": int((~branch_mask).sum().item()),
        "credited_logical_actions": int(active.sum().item()),
        "continuation_credit_sum": float(credit[~branch_mask].sum().item()),
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "gradient_norm": float(gradient_norm.detach().item()),
        "ratio_mean": float(ratio.detach()[active].mean().item()),
        "ratio_min": float(ratio.detach()[active].min().item()),
        "ratio_max": float(ratio.detach()[active].max().item()),
        "clip_fraction": float(
            ((ratio.detach()[active] < 1.0 - clip_epsilon) | (ratio.detach()[active] > 1.0 + clip_epsilon))
            .to(torch.float32)
            .mean()
            .item()
        ),
        "approx_kl_behavior": float((behavior.detach()[active] - learner.detach()[active]).mean().item()),
        **_parameter_delta(model, root_reference),
    }
    if not all(_finite(value) for value in metrics.values() if isinstance(value, (float, int))):
        raise ValueError("sibling-fiber metrics are non-finite")
    return metrics


def sibling_fiber_grpo_update_groups(
    model: torch.nn.Module,
    root_reference: torch.nn.Module,
    trajectory_groups: list[list[dict[str, Any]]],
    *,
    clip_epsilon: float = 0.2,
    learning_rate: float = 1e-5,
    advantage_epsilon: float = 1e-8,
    credit_scope: str = "branch_and_continuation",
    continuation_discount: float = 0.97,
    update_epochs: int = 1,
    deck_group_advantages: list[float] | None = None,
    deck_relative_weight: float = 0.0,
    prospective_aux_weight: float = 0.0,
    deck_aux_weight: float = 0.0,
    aux_batch_size: int = 256,
) -> dict[str, Any]:
    """Apply repeated clipped updates over sibling and inter-deck credit.

    Each group is normalized against its own terminal-return distribution and
    validated against its own exact recurrent base. The groups are only
    combined after those per-base checks. ``deck_group_advantages`` adds a
    second group-relative signal computed between learner decks under the same
    opponent and paired seed; it never mixes observations or legal masks from
    different bases. Behavior logprobs remain frozen across every epoch.
    """
    if not trajectory_groups:
        raise ValueError("sibling-fiber GRPO requires at least one trajectory group")
    if not 0.0 < clip_epsilon < 1.0 or not math.isfinite(clip_epsilon):
        raise ValueError("clip_epsilon must be finite and between zero and one")
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    if credit_scope not in {"branch_only", "branch_and_continuation"}:
        raise ValueError("credit_scope must be branch_only or branch_and_continuation")
    if not 0.0 < continuation_discount <= 1.0 or not math.isfinite(continuation_discount):
        raise ValueError("continuation_discount must be finite and in (0, 1]")
    if update_epochs < 1:
        raise ValueError("update_epochs must be at least one")
    if not math.isfinite(deck_relative_weight) or deck_relative_weight < 0.0:
        raise ValueError("deck_relative_weight must be finite and non-negative")
    if not math.isfinite(prospective_aux_weight) or prospective_aux_weight < 0.0:
        raise ValueError("prospective_aux_weight must be finite and non-negative")
    if not math.isfinite(deck_aux_weight) or deck_aux_weight < 0.0:
        raise ValueError("deck_aux_weight must be finite and non-negative")
    if aux_batch_size < 1:
        raise ValueError("aux_batch_size must be at least one")
    if prospective_aux_weight > 0.0 and not hasattr(model, "aux_predictions"):
        raise ValueError("prospective auxiliary training requires model.aux_predictions")
    if deck_aux_weight > 0.0 and not hasattr(model, "deck_card_logits"):
        raise ValueError("deck reconstruction requires model.deck_card_logits")
    if deck_group_advantages is None:
        deck_group_advantages = [0.0] * len(trajectory_groups)
    if len(deck_group_advantages) != len(trajectory_groups):
        raise ValueError("deck_group_advantages must align one-to-one with trajectory groups")
    if not all(math.isfinite(float(value)) for value in deck_group_advantages):
        raise ValueError("deck_group_advantages must be finite")

    credit_parts: list[torch.Tensor] = []
    branch_mask_parts: list[torch.Tensor] = []
    group_stats_list: list[dict[str, Any]] = []
    prepared_groups: list[
        tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]
    ] = []
    total_active = 0
    for group_index, trajectories in enumerate(trajectory_groups):
        _validate_group(trajectories)
        returns = [float(item["terminal_return"]) for item in trajectories]
        advantages, group_stats = normalize_group_returns(returns, epsilon=advantage_epsilon)
        deck_advantage = float(deck_group_advantages[group_index])
        combined_advantages = advantages + deck_relative_weight * deck_advantage
        expected_mapping = torch.cat(
            [
                torch.full((len(item["decisions"]),), index, dtype=torch.long)
                for index, item in enumerate(trajectories)
            ]
        )
        branch_values: list[bool] = []
        discount_values: list[float] = []
        for trajectory in trajectories:
            for decision_index, _decision in enumerate(trajectory["decisions"]):
                is_branch = decision_index == 0
                branch_values.append(is_branch)
                discount_values.append(
                    1.0
                    if is_branch or credit_scope == "branch_only"
                    else continuation_discount**decision_index
                )
        branch_mask = torch.as_tensor(branch_values, dtype=torch.bool)
        credit = combined_advantages[expected_mapping] * torch.as_tensor(
            discount_values, dtype=torch.float32
        )
        if credit_scope == "branch_only":
            credit = credit.masked_fill(~branch_mask, 0.0)
        active = credit != 0.0
        total_active += int(active.sum().item())
        credit_parts.append(credit)
        branch_mask_parts.append(branch_mask)
        group_stats_list.append(
            {
                **group_stats,
                "deck_group_advantage": deck_advantage,
                "deck_relative_weight": deck_relative_weight,
                "combined_advantages": combined_advantages.tolist(),
            }
        )
        prepared_groups.append((trajectories, expected_mapping, branch_mask, credit, group_stats))

    if prospective_aux_weight > 0.0 or deck_aux_weight > 0.0:
        prospective_examples, deck_examples = _prospective_auxiliary_examples(trajectory_groups)
    else:
        prospective_examples, deck_examples = [], []
    auxiliary_signal = (
        prospective_aux_weight > 0.0 and bool(prospective_examples)
    ) or (deck_aux_weight > 0.0 and bool(deck_examples))
    if total_active == 0 and not auxiliary_signal:
        # A round-robin stratum can be completely outcome-homogeneous even
        # when every fiber and provenance invariant is valid. Treat that as
        # an observed no-signal update, matching the single-group fail-closed
        # behavior, so a multi-deck run still emits a truthful root-equivalent
        # candidate and records the zero-variance coverage instead of losing
        # the whole matrix at the update boundary.
        zero_variance_groups = sum(bool(item["zero_variance"]) for item in group_stats_list)
        logical_decisions = sum(
            len(trajectory["decisions"])
            for trajectories in trajectory_groups
            for trajectory in trajectories
        )
        branch_logical_decisions = sum(
            1
            for trajectories in trajectory_groups
            for trajectory in trajectories
            if trajectory["decisions"]
        )
        continuation_logical_decisions = logical_decisions - branch_logical_decisions
        return {
            "algorithm": "sibling_fiber_grpo_grouped",
            "precision": "FP32",
            "policy_only": True,
            "branch_only_credit": credit_scope == "branch_only",
            "continuation_credit": credit_scope == "branch_and_continuation",
            "value_loss": 0.0,
            "optimizer_steps": 0,
            "requested_update_epochs": update_epochs,
            "epoch_metrics": [],
            "update_seconds": 0.0,
            "group_count": len(trajectory_groups),
            "group_sizes": [len(group) for group in trajectory_groups],
            "zero_variance_groups": zero_variance_groups,
            "zero_variance_group": zero_variance_groups == len(group_stats_list),
            "return_means": [item["return_mean"] for item in group_stats_list],
            "return_stds": [item["return_std"] for item in group_stats_list],
            "return_mean": float(np.mean([item["return_mean"] for item in group_stats_list])),
            "return_std": float(np.mean([item["return_std"] for item in group_stats_list])),
            "credit_scope": credit_scope,
            "continuation_discount": continuation_discount,
            "logical_decisions": logical_decisions,
            "branch_logical_decisions": branch_logical_decisions,
            "continuation_logical_decisions": continuation_logical_decisions,
            "credited_logical_actions": 0,
            "continuation_credit_sum": 0.0,
            "deck_relative_credit": deck_relative_weight > 0.0,
            "deck_relative_weight": deck_relative_weight,
            "deck_group_advantages": [float(value) for value in deck_group_advantages],
            "prospective_aux_weight": prospective_aux_weight,
            "deck_aux_weight": deck_aux_weight,
            "prospective_aux_examples": len(prospective_examples),
            "deck_aux_examples": len(deck_examples),
            "loss": 0.0,
            "policy_loss": 0.0,
            "gradient_norm": 0.0,
            "ratio_mean": None,
            "ratio_min": None,
            "ratio_max": None,
            "clip_fraction": 0.0,
            "approx_kl_behavior": 0.0,
            "no_update_reason": "all_groups_zero_variance",
            **_parameter_delta(model, root_reference),
        }

    started = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    credit = torch.cat(credit_parts)
    branch_mask = torch.cat(branch_mask_parts)
    active = credit != 0.0
    zero_variance_groups = sum(bool(item["zero_variance"]) for item in group_stats_list)
    epoch_metrics: list[dict[str, Any]] = []
    last_learner = last_behavior = last_ratio = None
    last_policy_loss = 0.0
    last_gradient_norm = torch.tensor(0.0)
    last_aux_metrics: dict[str, Any] = {}
    last_deck_metrics: dict[str, Any] = {}
    for epoch in range(update_epochs):
        optimizer.zero_grad(set_to_none=True)
        learner_parts: list[torch.Tensor] = []
        behavior_parts: list[torch.Tensor] = []
        surrogate_sums: list[torch.Tensor] = []
        policy_groups = prepared_groups if total_active > 0 else []
        for trajectories, expected_mapping, _branch_mask, group_credit, _group_stats in policy_groups:
            # Recompute one group at a time so multi-epoch training never
            # retains the full rollout graph. The stored behavior logprobs are
            # immutable; only learner logprobs change across epochs.
            learner, behavior, decision_mapping, _substep_mapping = recompute_logprobs_by_decision(
                model, trajectories
            )
            if not torch.equal(decision_mapping, expected_mapping):
                raise AssertionError("sibling logical decision mapping changed during recomputation")
            ratio = torch.exp(learner - behavior.detach())
            if not _finite(ratio):
                raise ValueError("grouped sibling importance ratios are non-finite")
            clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
            surrogate = torch.minimum(
                ratio * group_credit.detach(), clipped * group_credit.detach()
            )
            group_active = group_credit != 0.0
            if bool(group_active.any().item()):
                surrogate_sum = surrogate[group_active].sum()
                if not _finite(surrogate_sum):
                    raise ValueError("grouped sibling policy loss is non-finite")
                (-(surrogate_sum / float(total_active))).backward()
                surrogate_sums.append(surrogate_sum.detach())
            else:
                surrogate_sums.append(torch.tensor(0.0))
            learner_parts.append(learner.detach())
            behavior_parts.append(behavior.detach())
            del learner, behavior, decision_mapping, _substep_mapping, ratio, clipped, surrogate

        learner_epoch = torch.cat(learner_parts) if learner_parts else torch.empty(0)
        behavior_epoch = torch.cat(behavior_parts) if behavior_parts else torch.empty(0)
        ratio_epoch = torch.exp(learner_epoch - behavior_epoch)
        if epoch == 0 and total_active > 0:
            identity_error = float((ratio_epoch[active] - 1.0).abs().max().item())
            if identity_error > 5e-5:
                raise ValueError(
                    f"initial learner/behavior ratio is not identity: max_error={identity_error}"
                )
        surrogate_total = torch.stack(surrogate_sums).sum() if surrogate_sums else torch.tensor(0.0)
        policy_loss_value = (
            -float(surrogate_total.item()) / float(total_active) if total_active > 0 else 0.0
        )
        last_aux_metrics = _backward_prospective_auxiliary(
            model,
            prospective_examples,
            weight=prospective_aux_weight,
            batch_size=aux_batch_size,
        )
        last_deck_metrics = _backward_deck_reconstruction(
            model,
            deck_examples,
            weight=deck_aux_weight,
            batch_size=aux_batch_size,
            epoch=epoch,
        )
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not _finite(gradient_norm):
            raise ValueError("sibling-fiber gradient norm is non-finite")
        optimizer.step()
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "policy_loss": policy_loss_value,
                "prospective_aux_loss": last_aux_metrics["loss"],
                "deck_aux_loss": last_deck_metrics["loss"],
                "gradient_norm": float(gradient_norm.detach().item()),
                "ratio_mean_pre_step": (
                    float(ratio_epoch[active].mean().item()) if total_active > 0 else None
                ),
                "ratio_min_pre_step": (
                    float(ratio_epoch[active].min().item()) if total_active > 0 else None
                ),
                "ratio_max_pre_step": (
                    float(ratio_epoch[active].max().item()) if total_active > 0 else None
                ),
                "clip_fraction_pre_step": float(
                    (
                        (ratio_epoch[active] < 1.0 - clip_epsilon)
                        | (ratio_epoch[active] > 1.0 + clip_epsilon)
                    ).to(torch.float32).mean().item()
                ) if total_active > 0 else 0.0,
                "approx_kl_behavior_pre_step": (
                    float((behavior_epoch[active] - learner_epoch[active]).mean().item())
                    if total_active > 0
                    else 0.0
                ),
                **_parameter_delta(model, root_reference),
            }
        )
        last_learner, last_behavior, last_ratio = learner_epoch, behavior_epoch, ratio_epoch
        last_policy_loss = policy_loss_value
        last_gradient_norm = gradient_norm.detach()

    model.eval()
    optimizer_steps = update_epochs
    update_seconds = time.perf_counter() - started
    assert last_learner is not None and last_behavior is not None and last_ratio is not None
    loss_value = (
        last_policy_loss
        + prospective_aux_weight * float(last_aux_metrics.get("loss", 0.0))
        + deck_aux_weight * float(last_deck_metrics.get("loss", 0.0))
    )
    logical_decisions = sum(
        len(trajectory["decisions"])
        for trajectories in trajectory_groups
        for trajectory in trajectories
    )
    metrics: dict[str, Any] = {
        "algorithm": "sibling_fiber_grpo_grouped",
        "precision": "FP32",
        "policy_only": prospective_aux_weight == 0.0 and deck_aux_weight == 0.0,
        "auxiliary_training": prospective_aux_weight > 0.0 or deck_aux_weight > 0.0,
        "branch_only_credit": credit_scope == "branch_only",
        "continuation_credit": credit_scope == "branch_and_continuation",
        "value_loss": 0.0,
        "optimizer_steps": optimizer_steps,
        "requested_update_epochs": update_epochs,
        "epoch_metrics": epoch_metrics,
        "update_seconds": update_seconds,
        "group_count": len(trajectory_groups),
        "group_sizes": [len(group) for group in trajectory_groups],
        "zero_variance_groups": zero_variance_groups,
        "zero_variance_group": zero_variance_groups == len(group_stats_list),
        "return_means": [item["return_mean"] for item in group_stats_list],
        "return_stds": [item["return_std"] for item in group_stats_list],
        "return_mean": float(np.mean([item["return_mean"] for item in group_stats_list])),
        "return_std": float(np.mean([item["return_std"] for item in group_stats_list])),
        "credit_scope": credit_scope,
        "continuation_discount": continuation_discount,
        "logical_decisions": logical_decisions,
        "branch_logical_decisions": int(branch_mask.sum().item()),
        "continuation_logical_decisions": int((~branch_mask).sum().item()),
        "credited_logical_actions": int(active.sum().item()),
        "continuation_credit_sum": float(credit[~branch_mask].sum().item()),
        "deck_relative_credit": deck_relative_weight > 0.0,
        "deck_relative_weight": deck_relative_weight,
        "deck_group_advantages": [float(value) for value in deck_group_advantages],
        "prospective_aux_weight": prospective_aux_weight,
        "deck_aux_weight": deck_aux_weight,
        "prospective_auxiliary": last_aux_metrics,
        "deck_reconstruction": last_deck_metrics,
        "loss": loss_value,
        "policy_loss": last_policy_loss,
        "gradient_norm": float(last_gradient_norm.item()),
        "ratio_mean": float(last_ratio[active].mean().item()) if total_active > 0 else None,
        "ratio_min": float(last_ratio[active].min().item()) if total_active > 0 else None,
        "ratio_max": float(last_ratio[active].max().item()) if total_active > 0 else None,
        "clip_fraction": float(
            ((last_ratio[active] < 1.0 - clip_epsilon) | (last_ratio[active] > 1.0 + clip_epsilon))
            .to(torch.float32)
            .mean()
            .item()
        ) if total_active > 0 else 0.0,
        "approx_kl_behavior": (
            float((last_behavior[active] - last_learner[active]).mean().item())
            if total_active > 0
            else 0.0
        ),
        "initial_ratio_max_abs_error": (
            float(
                max(
                    abs(epoch_metrics[0]["ratio_min_pre_step"] - 1.0),
                    abs(epoch_metrics[0]["ratio_max_pre_step"] - 1.0),
                )
            )
            if total_active > 0
            else None
        ),
        **_parameter_delta(model, root_reference),
    }
    if not all(_finite(value) for value in metrics.values() if isinstance(value, (float, int))):
        raise ValueError("grouped sibling-fiber metrics are non-finite")
    return metrics


__all__ = [
    "APPROVED_STAGE4_ROOT_SHA256",
    "DEFAULT_OUTPUT",
    "SIBLING_FORMAT",
    "collect_sibling_fiber_group",
    "sibling_fiber_grpo_update",
    "sibling_fiber_grpo_update_groups",
    "flatten_provenance_bundle",
    "save_grpo_candidate_checkpoint",
]
