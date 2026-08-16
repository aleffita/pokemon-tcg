"""Bounded sibling-fiber GRPO for AR-020.

Each fiber starts from the same frozen-root recurrent base and common random
seed, but takes a distinct legal action at the first logical decision. The
first probe applies relative credit only to that branching conditional action;
continuation decisions remain provenance evidence and receive no gradient.
"""

from __future__ import annotations

import math
import os
import pickle
import time
import traceback
from pathlib import Path
from typing import Any

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
) -> list[int]:
    """Choose a dynamic sibling set from one real decision mask.

    ``requested_k`` is only a cap.  The effective K is the number of distinct
    non-submit legal actions available at this exact base, or all legal
    actions when the fiber is smaller.  Submit is excluded when there are at
    least two ordinary choices because it is a control action, not a play.
    """
    if requested_k < 2:
        raise ValueError("requested sibling K must be at least two")
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


def _probe_branch_fibers(
    *,
    model: torch.nn.Module,
    encoder: DateBoundEncoder,
    deck: list[int],
    opponent_deck: list[int] | None,
    seed: int,
    games: int,
) -> tuple[list[int], dict[str, Any], CabtEnv, dict[str, np.ndarray], dict[str, Any]]:
    """Find a real in-game branch and keep its live env as an exact snapshot."""
    for seed_offset in range(32):
        probe_seed = seed + seed_offset
        mirror = _StatefulMirror(model, encoder, np.random.default_rng(probe_seed + 1000))
        env = CabtEnv(
            agent_deck=deck,
            opponent_deck=opponent_deck or deck,
            opponent_fn=mirror,
            encoder=encoder,
            seed=probe_seed,
            max_steps=4000,
            reset_hook=lambda _attempt, mirror=mirror: mirror.reset_episode("branch-probe"),
        )
        keep_env = False
        try:
            encoded, reset_info = env.reset(seed=probe_seed)
            mirror.set_side(1 - int(reset_info["agent_index"]))
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
                candidates = _branch_candidates(logits, mask, games, sibling_generator)
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


def _collect_forked_fiber(
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Continue one exact engine snapshot in a fork and return its trajectory."""
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
            mirror = env.opponent_fn
            rows = collect_episode(
                env,
                model,
                encoder,
                episode_id,
                "sibling_fiber_current_vs_current_true_recurrent",
                reset_seed,
                deck_content_hash,
                deck_source_file_hash,
                model_hash,
                action_generator,
                bundle,
                on_episode_reset=lambda agent_side: mirror.set_side(1 - agent_side),
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect K same-base forced-fiber continuations with common randomness."""
    if games < 2:
        raise ValueError("AR-020 requested K cap must be at least two")
    actions, base, base_env, base_observation, base_reset_info = _probe_branch_fibers(
        model=model,
        encoder=encoder,
        deck=deck,
        opponent_deck=opponent_deck,
        seed=seed,
        games=games,
    )
    collection_seed = int(base["probe_seed"])
    trajectories: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for fiber_index, action in enumerate(actions):
            episode_id = f"sibling-fiber-{fiber_index:03d}"
            rows, bundle = _collect_forked_fiber(
                env=base_env,
                model=model,
                encoder=encoder,
                episode_id=episode_id,
                action=action,
                reset_seed=collection_seed,
                deck_content_hash=deck_content_hash,
                deck_source_file_hash=deck_source_file_hash,
                model_hash=model_hash,
                base_observation=base_observation,
                base_reset_info=base_reset_info,
            )
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
        "games_per_second": effective_games / elapsed if elapsed else None,
        "logical_decisions": decisions,
        "substeps": substeps,
        "decisions_per_second": decisions / elapsed if elapsed else None,
        "substeps_per_second": substeps / elapsed if elapsed else None,
        "returns": returns,
        "branch_actions": actions,
        "effective_group_size": effective_games,
        "branch_base": base,
        "agent_deck_content_sha256": deck_content_hash,
        "agent_deck_source_file_sha256": deck_source_file_hash,
        "opponent_deck_content_sha256": opponent_deck_content_hash,
        "opponent_deck_source_file_sha256": opponent_deck_source_file_hash,
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


__all__ = [
    "APPROVED_STAGE4_ROOT_SHA256",
    "DEFAULT_OUTPUT",
    "SIBLING_FORMAT",
    "collect_sibling_fiber_group",
    "sibling_fiber_grpo_update",
    "flatten_provenance_bundle",
    "save_grpo_candidate_checkpoint",
]
