"""In-memory trajectory-group GRPO for the first AR-019 micro-update.

The collector intentionally reuses the corrected AR-018 current-vs-current
recurrent environment path.  Only compact tensors needed by one update stay
in memory; no rollout bundle is serialized by this module.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from rl.env.env import CabtEnv
from rl.policy_infer_torch import TORCH_INFERENCE_FORMAT
from scripts.rl.ppo_micro_update import sha256_file
from scripts.rl.trajectory_probe import (
    APPROVED_STAGE4_ROOT_SHA256,
    DateBoundEncoder,
    _StatefulMirror,
    collect_episode,
    composite_behavior_logprob,
    deck_content_sha256,
    digest_tensor,
    load_deck,
    load_stage4,
    validate_bundle,
    validate_rows,
)


GRPO_FORMAT = "ptcg-stage4-trajectory-group-grpo-v1"
DEFAULT_OUTPUT = Path("experiments/autoresearch/AR-019")


def _finite(value: torch.Tensor | float) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    return math.isfinite(float(value))


def _masked_distribution(logits: torch.Tensor, masks: torch.Tensor) -> torch.distributions.Categorical:
    if logits.ndim != 2 or masks.ndim != 2 or logits.shape != masks.shape:
        raise ValueError(f"GRPO logits/masks shape mismatch: {logits.shape} != {masks.shape}")
    legal = masks >= 0.5
    if not bool(legal.any(dim=1).all().item()):
        raise ValueError("GRPO batch contains an all-illegal action mask")
    return torch.distributions.Categorical(logits=logits.masked_fill(~legal, float("-inf")))


def normalize_group_returns(
    returns: Iterable[float],
    *,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    """Normalize terminal returns and fail closed on a zero-variance group."""
    values = torch.as_tensor([float(value) for value in returns], dtype=torch.float32)
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("trajectory group must contain at least two returns")
    if not _finite(values) or not all(float(value) in (-1.0, 0.0, 1.0) for value in values):
        raise ValueError("trajectory returns must be finite and in {-1, 0, +1}")
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    mean = values.mean()
    std = values.std(unbiased=False)
    zero_variance = bool(float(std.item()) <= epsilon)
    if zero_variance:
        # A homogeneous terminal group contains no relative policy signal.
        # Returning zero advantages makes the optimizer a safe no-op.
        advantages = torch.zeros_like(values)
    else:
        advantages = (values - mean) / (std + epsilon)
    if not _finite(advantages):
        raise ValueError("normalized group advantages are non-finite")
    return advantages, {
        "return_mean": float(mean.item()),
        "return_std": float(std.item()),
        "zero_variance": zero_variance,
    }


def trajectory_from_bundle(
    rows: list[dict[str, Any]],
    bundle: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert one collected agent lane into compact logical decisions."""
    validate_rows(rows)
    validate_bundle(bundle, rows)
    terminal_rows = [row for row in rows if bool(row.get("terminal"))]
    if len(terminal_rows) != 1:
        raise ValueError("trajectory must contain exactly one terminal row")
    terminal_return = float(terminal_rows[0]["reward"])
    if terminal_return not in (-1.0, 0.0, 1.0):
        raise ValueError(f"terminal return is not in {{-1, 0, +1}}: {terminal_return}")

    grouped: dict[int, list[dict[str, Any]]] = {}
    for sample in bundle:
        grouped.setdefault(int(sample["decision_index"]), []).append(sample)
    decisions: list[tuple[dict[str, Any], ...]] = []
    for decision_index in sorted(grouped):
        samples = grouped[decision_index]
        if [int(sample["substep"]) for sample in samples] != list(range(len(samples))):
            raise ValueError(f"decision {decision_index} has non-contiguous substeps")
        logical_values = {float(sample.get("logical_action_logprob")) for sample in samples}
        if len(logical_values) != 1:
            raise ValueError(f"decision {decision_index} has inconsistent logical logprobs")
        expected = composite_behavior_logprob(
            [float(sample["behavior_logprob"]) for sample in samples]
        )
        observed = next(iter(logical_values))
        if not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"decision {decision_index} logical logprob is not the substep sum")
        decisions.append(tuple(samples))
    if not decisions:
        raise ValueError("trajectory contains no logical decisions")
    return {
        "episode_id": str(rows[0]["episode_id"]),
        "terminal_return": terminal_return,
        "decisions": tuple(decisions),
        "logical_decisions": len(decisions),
        "substeps": len(bundle),
    }


def _flatten_trajectory_decisions(
    trajectories: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    samples: list[dict[str, Any]] = []
    decision_trajectory: list[int] = []
    decision_substeps: list[int] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        for decision in trajectory["decisions"]:
            if not decision:
                raise ValueError("logical decision cannot be empty")
            samples.extend(decision)
            decision_trajectory.append(trajectory_index)
            decision_substeps.append(len(decision))
    if not samples or not decision_trajectory:
        raise ValueError("trajectory group contains no update samples")
    return samples, decision_trajectory, decision_substeps


def _batch_samples(samples: list[dict[str, Any]]) -> tuple[
    dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    keys = tuple(samples[0]["model_input"])
    if any(tuple(sample["model_input"]) != keys for sample in samples):
        raise ValueError("trajectory model-input key order differs across substeps")
    model_input = {
        key: torch.cat(
            [sample["model_input"][key].detach().to(device="cpu") for sample in samples],
            dim=0,
        )
        for key in keys
    }
    masks = torch.stack(
        [sample["action_mask"].detach().to(device="cpu", dtype=torch.float32) for sample in samples]
    )
    memories = torch.cat(
        [sample["memory_input"].detach().to(device="cpu", dtype=torch.float32) for sample in samples],
        dim=0,
    )
    actions = torch.as_tensor([int(sample["action"]) for sample in samples], dtype=torch.long)
    behavior_logprobs = torch.as_tensor(
        [float(sample["behavior_logprob"]) for sample in samples], dtype=torch.float32
    )
    if not _finite(behavior_logprobs):
        raise ValueError("behavior substep logprobs are non-finite")
    return model_input, masks, memories, actions, behavior_logprobs


def _learner_logprobs(
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model_input, masks, memories, actions, behavior_substep = _batch_samples(samples)
    logits, _values, _memory_out = model.logits_value(model_input, memory_in=memories)
    distribution = _masked_distribution(logits, masks)
    learner_substep = distribution.log_prob(actions)
    if not _finite(learner_substep):
        raise ValueError("learner substep logprobs are non-finite")
    return learner_substep, behavior_substep, masks


def expand_group_advantages(
    trajectories: list[dict[str, Any]],
    advantages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand one trajectory advantage to decisions, then their substeps."""
    _samples, decision_trajectory, decision_substeps = _flatten_trajectory_decisions(trajectories)
    mapping = torch.as_tensor(decision_trajectory, dtype=torch.long)
    decision_advantages = advantages[mapping]
    substep_advantages = torch.repeat_interleave(
        decision_advantages, torch.as_tensor(decision_substeps, dtype=torch.long)
    )
    return decision_advantages, substep_advantages, mapping


def recompute_logprobs_by_decision(
    model: torch.nn.Module,
    trajectories: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return learner/behavior logical logprobs and decision-to-trajectory map."""
    samples, decision_trajectory, decision_substeps = _flatten_trajectory_decisions(trajectories)
    learner_substep, behavior_substep, _masks = _learner_logprobs(model, samples)
    learner_decisions: list[torch.Tensor] = []
    behavior_decisions: list[torch.Tensor] = []
    offset = 0
    for count in decision_substeps:
        end = offset + count
        learner_decisions.append(learner_substep[offset:end].sum())
        behavior_decisions.append(behavior_substep[offset:end].sum())
        offset = end
    learner = torch.stack(learner_decisions)
    behavior = torch.stack(behavior_decisions)
    mapping = torch.as_tensor(decision_trajectory, dtype=torch.long)
    substep_mapping = torch.repeat_interleave(mapping, torch.as_tensor(decision_substeps))
    return learner, behavior, mapping, substep_mapping


def _parameter_delta(model: torch.nn.Module, reference: torch.nn.Module) -> dict[str, float | int]:
    squared = 0.0
    max_abs = 0.0
    changed = 0
    count = 0
    for candidate, root in zip(model.parameters(), reference.parameters(), strict=True):
        delta = candidate.detach().to(torch.float32) - root.detach().to(torch.float32)
        squared += float(torch.sum(delta * delta).item())
        max_abs = max(max_abs, float(delta.abs().max().item()))
        changed += int(torch.count_nonzero(delta).item())
        count += delta.numel()
    return {
        "parameter_l2": math.sqrt(squared),
        "parameter_max_abs": max_abs,
        "changed_parameters": changed,
        "parameter_count": count,
    }


def trajectory_group_grpo_update(
    model: torch.nn.Module,
    root_reference: torch.nn.Module,
    trajectories: list[dict[str, Any]],
    *,
    clip_epsilon: float = 0.2,
    learning_rate: float = 1e-5,
    advantage_epsilon: float = 1e-8,
) -> dict[str, float | int | bool | str]:
    """Run one policy-only trajectory-group GRPO update in FP32."""
    if len(trajectories) < 2:
        raise ValueError("trajectory-group GRPO requires at least two trajectories")
    if not 0.0 < clip_epsilon < 1.0 or not math.isfinite(clip_epsilon):
        raise ValueError("clip_epsilon must be finite and between zero and one")
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    returns = [float(trajectory["terminal_return"]) for trajectory in trajectories]
    advantages, group_stats = normalize_group_returns(returns, epsilon=advantage_epsilon)
    learner, behavior, decision_mapping, substep_mapping = recompute_logprobs_by_decision(
        model, trajectories
    )
    decision_advantages, expanded_advantages, expected_decision_mapping = expand_group_advantages(
        trajectories, advantages
    )
    if not torch.equal(decision_mapping, expected_decision_mapping):
        raise AssertionError("logical credit mapping changed during learner recomputation")
    if expanded_advantages.numel() != substep_mapping.numel():
        raise AssertionError("expanded logical credit does not cover all substeps")
    delta = learner - behavior.detach()
    ratio = torch.exp(delta)
    if not _finite(ratio):
        raise ValueError("GRPO importance ratios are non-finite")
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratio * decision_advantages.detach(), clipped * decision_advantages.detach())
    policy_loss = -surrogate.mean()
    zero_variance = bool(group_stats["zero_variance"])
    gradient_norm = torch.tensor(0.0)
    update_seconds_started = time.perf_counter()
    if zero_variance:
        # Evaluate diagnostics but do not let a homogeneous group mutate policy.
        loss = policy_loss.detach() * 0.0
        optimizer_steps = 0
    else:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        optimizer.zero_grad(set_to_none=True)
        loss = policy_loss
        if not _finite(loss):
            raise ValueError("GRPO policy loss is non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not _finite(gradient_norm):
            raise ValueError("GRPO gradient norm is non-finite")
        optimizer.step()
        model.eval()
        optimizer_steps = 1
    update_seconds = time.perf_counter() - update_seconds_started
    delta_metrics = _parameter_delta(model, root_reference)
    metrics: dict[str, float | int | bool | str] = {
        "algorithm": "trajectory_group_grpo",
        "precision": "FP32",
        "policy_only": True,
        "value_loss": 0.0,
        "optimizer_steps": optimizer_steps,
        "update_seconds": update_seconds,
        "group_size": len(trajectories),
        "zero_variance_group": zero_variance,
        "return_mean": group_stats["return_mean"],
        "return_std": group_stats["return_std"],
        "logical_decisions": int(learner.numel()),
        "substeps": int(substep_mapping.numel()),
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "gradient_norm": float(gradient_norm.detach().item()),
        "ratio_mean": float(ratio.detach().mean().item()),
        "ratio_min": float(ratio.detach().min().item()),
        "ratio_max": float(ratio.detach().max().item()),
        "clip_fraction": float(
            ((ratio.detach() < 1.0 - clip_epsilon) | (ratio.detach() > 1.0 + clip_epsilon))
            .to(torch.float32)
            .mean()
            .item()
        ),
        "approx_kl_behavior": float((behavior.detach() - learner.detach()).mean().item()),
        "credit_shared_per_logical_decision": True,
        "credit_shared_across_substeps": True,
        **delta_metrics,
    }
    if not all(_finite(value) for value in metrics.values() if isinstance(value, (float, int))):
        raise ValueError("GRPO metrics are non-finite")
    return metrics


def save_grpo_candidate_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model_metadata: dict[str, Any],
    *,
    root_sha256: str,
    config: dict[str, Any],
    diagnostics: dict[str, Any],
    experiment: str = "AR-019",
) -> str:
    """Write a strict inference checkpoint without persisting the rollout."""
    arch_config = {
        key: value
        for key, value in model_metadata.items()
        if key not in {"inference_config", "training_config", "static_feature_contract"}
    }
    payload = {
        "format": TORCH_INFERENCE_FORMAT,
        "arch_config": arch_config,
        "inference_config": dict(model_metadata["inference_config"]),
        "training_config": dict(model_metadata.get("training_config", {})),
        "static_card_features": (
            getattr(model, "card_feat").detach().to(device="cpu", dtype=torch.float32).clone()
            if getattr(model, "card_feat", None) is not None
            else None
        ),
        "static_feature_contract": model_metadata.get("static_feature_contract"),
        "state_dict": {
            key: value.detach().to(device="cpu", dtype=torch.float32).clone()
            for key, value in model.state_dict().items()
        },
        "autoresearch": {
            "experiment": experiment,
            "root_sha256": root_sha256,
            "config": config,
            "diagnostics": diagnostics,
            "rollout_persistence": "none; compact tensors were in-memory only",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return sha256_file(path)


def _git_commit() -> str | None:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_stage4_trajectory_group(
    *,
    model: torch.nn.Module,
    encoder: DateBoundEncoder,
    deck: list[int],
    deck_content_hash: str,
    deck_source_file_hash: str,
    model_hash: str,
    games: int = 4,
    seed: int = 19019,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect K complete agent trajectories against the recurrent current mirror."""
    if games != 4:
        raise ValueError("AR-019 requires exactly K=4 trajectories")
    mirror = _StatefulMirror(model, encoder, np.random.default_rng(seed + 1000))
    trajectories: list[dict[str, Any]] = []
    started = time.perf_counter()
    for game_index in range(games):
        episode_id = f"trajectory-group-{game_index:03d}"
        bundle: list[dict[str, Any]] = []
        env = CabtEnv(
            agent_deck=deck,
            opponent_deck=deck,
            opponent_fn=mirror,
            encoder=encoder,
            seed=seed + game_index,
            max_steps=4000,
            reset_hook=lambda _attempt, episode_id=episode_id: mirror.reset_episode(episode_id),
        )
        try:
            rows = collect_episode(
                env,
                model,
                encoder,
                episode_id,
                "current_vs_current_true_recurrent",
                seed + game_index,
                deck_content_hash,
                deck_source_file_hash,
                model_hash,
                torch.Generator(device="cpu").manual_seed(seed + game_index),
                bundle,
                on_episode_reset=lambda agent_side: mirror.set_side(1 - agent_side),
            )
        finally:
            env.close()
        trajectory = trajectory_from_bundle(rows, bundle)
        trajectories.append(trajectory)
    elapsed = time.perf_counter() - started
    decisions = sum(int(trajectory["logical_decisions"]) for trajectory in trajectories)
    substeps = sum(int(trajectory["substeps"]) for trajectory in trajectories)
    returns = [float(trajectory["terminal_return"]) for trajectory in trajectories]
    return trajectories, {
        "behavior_snapshot_sha256": model_hash,
        "games": games,
        "collection_seconds": elapsed,
        "games_per_second": games / elapsed if elapsed else None,
        "logical_decisions": decisions,
        "substeps": substeps,
        "decisions_per_second": decisions / elapsed if elapsed else None,
        "substeps_per_second": substeps / elapsed if elapsed else None,
        "returns": returns,
        "trajectory_summaries": [
            {
                "episode_id": trajectory["episode_id"],
                "terminal_return": trajectory["terminal_return"],
                "logical_decisions": trajectory["logical_decisions"],
                "substeps": trajectory["substeps"],
            }
            for trajectory in trajectories
        ],
    }


__all__ = [
    "APPROVED_STAGE4_ROOT_SHA256",
    "DEFAULT_OUTPUT",
    "GRPO_FORMAT",
    "collect_stage4_trajectory_group",
    "expand_group_advantages",
    "normalize_group_returns",
    "recompute_logprobs_by_decision",
    "save_grpo_candidate_checkpoint",
    "trajectory_from_bundle",
    "trajectory_group_grpo_update",
]
