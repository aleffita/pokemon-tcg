"""Lateral prospective branch planner for MLX.

The planner consumes already-computed trunk context/memory plus real branch
tokens.  It does not replace or modify the current TokenTransformer or TBPTT
path.  Q/K attention uses the shared four-axis RoPE-ND coordinate contract,
while attention and objective reductions are performed in FP32.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import pickle

import mlx.core as mx
import mlx.nn as nn

from rl.prospective_schema import (
    BRANCH_POLICY_SCORE,
    BRANCH_VALID,
    EXPECTED_PRIZES,
    KO_LOGIT,
    KO_PROBABILITY,
    N_PROSPECTIVE_AXES,
    PRIZE_LOGIT,
    SCALAR_RETURN,
    SCALAR_VALUE,
    TERMINAL_LOGIT,
    TERMINAL_PROBABILITY,
    UNCERTAINTY,
    UNCERTAINTY_LOGIT,
    ProspectivePlannerConfig,
    prospective_checkpoint_metadata,
)


FP16_MIN_FINITE = -65_504.0


def rope_nd_angles(
    coordinates: mx.array,
    *,
    head_dim: int,
    base: float = 10_000.0,
) -> tuple[mx.array, mx.array]:
    """Return FP32 cosine/sine tables for four-axis rotary attention.

    The returned shape is ``[B, L, 4, head_dim / 8]``.  Each coordinate axis
    owns an equal, even-width slice of every attention head.
    """

    if coordinates.ndim != 3 or coordinates.shape[-1] != N_PROSPECTIVE_AXES:
        raise ValueError(
            "coordinates must have shape [B,L,"
            f"{N_PROSPECTIVE_AXES}], got {coordinates.shape}"
        )
    if head_dim % (2 * N_PROSPECTIVE_AXES) != 0:
        raise ValueError(
            f"head_dim must be divisible by {2 * N_PROSPECTIVE_AXES}"
        )
    axis_dim = head_dim // N_PROSPECTIVE_AXES
    pair_count = axis_dim // 2
    pair_index = mx.arange(pair_count, dtype=mx.float32)
    inverse_frequency = mx.power(
        mx.array(base, dtype=mx.float32),
        -(2.0 * pair_index / float(axis_dim)),
    )
    angles = (
        coordinates.astype(mx.float32)[..., None]
        * inverse_frequency.reshape(1, 1, 1, pair_count)
    )
    return mx.cos(angles), mx.sin(angles)


def apply_rope_nd(
    query: mx.array,
    key: mx.array,
    coordinates: mx.array,
    *,
    base: float = 10_000.0,
) -> tuple[mx.array, mx.array]:
    """Apply shared RoPE-ND to ``[B,H,L,Dh]`` query and key tensors."""

    if query.shape != key.shape or query.ndim != 4:
        raise ValueError("query and key must have equal [B,H,L,Dh] shapes")
    if coordinates.shape[:2] != (query.shape[0], query.shape[2]):
        raise ValueError("coordinates do not match query/key batch and length")
    head_dim = query.shape[-1]
    axis_dim = head_dim // N_PROSPECTIVE_AXES
    cosine, sine = rope_nd_angles(coordinates, head_dim=head_dim, base=base)
    cosine = cosine[:, None, ...]
    sine = sine[:, None, ...]

    def rotate(value: mx.array) -> mx.array:
        axes = value.reshape(
            value.shape[0],
            value.shape[1],
            value.shape[2],
            N_PROSPECTIVE_AXES,
            axis_dim // 2,
            2,
        ).astype(mx.float32)
        even = axes[..., 0]
        odd = axes[..., 1]
        rotated = mx.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine),
            axis=-1,
        )
        return rotated.reshape(value.shape).astype(value.dtype)

    return rotate(query), rotate(key)


class FP32LayerNormMLX(nn.Module):
    """FP16 affine parameters with explicit FP32 mean/variance reductions."""

    def __init__(self, dimensions: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = mx.ones((dimensions,), dtype=mx.float16)
        self.bias = mx.zeros((dimensions,), dtype=mx.float16)
        self.eps = float(eps)

    def __call__(self, value: mx.array) -> mx.array:
        source_dtype = value.dtype
        value_fp32 = value.astype(mx.float32)
        mean = mx.mean(value_fp32, axis=-1, keepdims=True)
        centered = value_fp32 - mean
        variance = mx.mean(centered * centered, axis=-1, keepdims=True)
        normalized = centered * mx.rsqrt(variance + self.eps)
        affine = (
            normalized * self.weight.astype(mx.float32)
            + self.bias.astype(mx.float32)
        )
        return affine.astype(source_dtype)


class ProspectiveAttentionMLX(nn.Module):
    """Custom multi-head attention with Q/K RoPE-ND."""

    def __init__(self, config: ProspectivePlannerConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.head_dim = config.d_model // config.nhead
        self.rope_base = config.rope_base
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=True)

    def __call__(
        self,
        value: mx.array,
        coordinates: mx.array,
        attention_mask: mx.array,
    ) -> mx.array:
        batch_size, length, _ = value.shape

        def project(layer: nn.Linear) -> mx.array:
            return (
                layer(value)
                .reshape(batch_size, length, self.nhead, self.head_dim)
                .transpose(0, 2, 1, 3)
            )

        query = project(self.q_proj)
        key = project(self.k_proj)
        projected_value = project(self.v_proj)
        query, key = apply_rope_nd(
            query, key, coordinates, base=self.rope_base
        )
        scores = mx.matmul(
            query.astype(mx.float32),
            key.astype(mx.float32).transpose(0, 1, 3, 2),
        ) * (self.head_dim ** -0.5)
        if attention_mask.shape not in {
            (batch_size, 1, length, length),
            (batch_size, self.nhead, length, length),
        }:
            raise ValueError(
                "attention_mask must have shape [B,1,L,L] or [B,H,L,L]"
            )
        scores = scores + attention_mask.astype(mx.float32)
        probabilities = mx.softmax(scores, axis=-1).astype(value.dtype)
        context = mx.matmul(probabilities, projected_value)
        context = (
            context.transpose(0, 2, 1, 3)
            .reshape(batch_size, length, self.d_model)
            .astype(value.dtype)
        )
        return self.out_proj(context).astype(value.dtype)


class ProspectivePlannerLayerMLX(nn.Module):
    """Pre-norm prospective attention block."""

    def __init__(self, config: ProspectivePlannerConfig) -> None:
        super().__init__()
        self.norm1 = FP32LayerNormMLX(config.d_model)
        self.attention = ProspectiveAttentionMLX(config)
        self.norm2 = FP32LayerNormMLX(config.d_model)
        self.ff1 = nn.Linear(config.d_model, config.ff_dim)
        self.ff2 = nn.Linear(config.ff_dim, config.d_model)

    def __call__(
        self,
        value: mx.array,
        coordinates: mx.array,
        attention_mask: mx.array,
    ) -> mx.array:
        value = value + self.attention(
            self.norm1(value), coordinates, attention_mask
        )
        feed_forward = self.ff2(nn.gelu(self.ff1(self.norm2(value))))
        return (value + feed_forward).astype(value.dtype)


class ProspectivePlannerMLX(nn.Module):
    """Lateral tree planner producing one prediction set per branch token."""

    def __init__(self, config: ProspectivePlannerConfig | None = None) -> None:
        super().__init__()
        self.config = config or ProspectivePlannerConfig()
        self.config.validate()
        self.layers = [
            ProspectivePlannerLayerMLX(self.config)
            for _ in range(self.config.nlayers)
        ]
        self.policy_head = nn.Linear(self.config.d_model, 1)
        self.return_head = nn.Linear(self.config.d_model, 1)
        self.value_head = nn.Linear(self.config.d_model, 1)
        self.ko_head = nn.Linear(self.config.d_model, 1)
        self.prize_head = nn.Linear(self.config.d_model, 1)
        self.terminal_head = nn.Linear(self.config.d_model, 1)
        self.uncertainty_head = nn.Linear(self.config.d_model, 1)
        self.set_dtype(mx.float16)

    def get_config(self) -> dict:
        return self.config.to_dict()

    def __call__(
        self,
        context: mx.array,
        branch_tokens: mx.array,
        coordinates: mx.array,
        attention_mask: mx.array,
        branch_valid: mx.array,
    ) -> dict[str, mx.array]:
        if context.ndim != 3 or branch_tokens.ndim != 3:
            raise ValueError("context and branch_tokens must be [B,L,D]")
        if context.shape[0] != branch_tokens.shape[0]:
            raise ValueError("context and branch_tokens batch sizes differ")
        if context.shape[2] != self.config.d_model:
            raise ValueError("context hidden dimension differs from config")
        if branch_tokens.shape[2] != self.config.d_model:
            raise ValueError("branch token hidden dimension differs from config")
        batch_size, branch_count, _ = branch_tokens.shape
        total_length = context.shape[1] + branch_count
        if coordinates.shape != (
            batch_size,
            total_length,
            N_PROSPECTIVE_AXES,
        ):
            raise ValueError("coordinates must cover context and branch tokens")
        if branch_valid.shape != (batch_size, branch_count):
            raise ValueError("branch_valid must have shape [B,T]")

        parameter_dtype = self.policy_head.weight.dtype
        hidden = mx.concatenate(
            (
                context.astype(parameter_dtype),
                branch_tokens.astype(parameter_dtype),
            ),
            axis=1,
        )
        for layer in self.layers:
            hidden = layer(hidden, coordinates, attention_mask)
        branches = hidden[:, context.shape[1] :, :]

        def raw(head: nn.Linear) -> mx.array:
            return head(branches).squeeze(-1).astype(parameter_dtype)

        policy_score = raw(self.policy_head)
        return_logit = raw(self.return_head)
        value_logit = raw(self.value_head)
        ko_logit = raw(self.ko_head)
        prize_logit = raw(self.prize_head)
        terminal_logit = raw(self.terminal_head)
        uncertainty_logit = raw(self.uncertainty_head)

        valid = branch_valid.astype(mx.bool_)

        def masked(value: mx.array, invalid: float = 0.0) -> mx.array:
            return mx.where(
                valid,
                value,
                mx.array(invalid, dtype=value.dtype),
            )

        # Nonlinear transforms run in FP32 and return to the FP16 activation
        # contract.  Consumers must retain branch_valid for all reductions.
        scalar_return = mx.tanh(return_logit.astype(mx.float32)).astype(
            parameter_dtype
        )
        scalar_value = mx.tanh(value_logit.astype(mx.float32)).astype(
            parameter_dtype
        )
        ko_probability = mx.sigmoid(ko_logit.astype(mx.float32)).astype(
            parameter_dtype
        )
        expected_prizes = (
            self.config.max_prizes
            * mx.sigmoid(prize_logit.astype(mx.float32))
        ).astype(parameter_dtype)
        terminal_probability = mx.sigmoid(
            terminal_logit.astype(mx.float32)
        ).astype(parameter_dtype)
        uncertainty = (
            mx.logaddexp(
                mx.array(0.0, dtype=mx.float32),
                uncertainty_logit.astype(mx.float32),
            )
            + self.config.uncertainty_floor
        ).astype(parameter_dtype)

        return {
            BRANCH_POLICY_SCORE: masked(policy_score, FP16_MIN_FINITE),
            SCALAR_RETURN: masked(scalar_return),
            SCALAR_VALUE: masked(scalar_value),
            KO_LOGIT: masked(ko_logit),
            KO_PROBABILITY: masked(ko_probability),
            PRIZE_LOGIT: masked(prize_logit),
            EXPECTED_PRIZES: masked(expected_prizes),
            TERMINAL_LOGIT: masked(terminal_logit),
            TERMINAL_PROBABILITY: masked(terminal_probability),
            UNCERTAINTY_LOGIT: masked(uncertainty_logit),
            UNCERTAINTY: masked(uncertainty),
            BRANCH_VALID: valid,
        }


@dataclass(frozen=True)
class GroupRelativeLossResult:
    """FP32 prospective objective components and their honest method name."""

    loss: mx.array
    policy_loss: mx.array
    kl_loss: mx.array
    clip_fraction: mx.array
    mode: str


def _masked_mean(value: mx.array, mask: mx.array) -> mx.array:
    value_fp32 = value.astype(mx.float32)
    weights = mask.astype(mx.float32)
    return mx.sum(value_fp32 * weights) / mx.maximum(
        mx.sum(weights), mx.array(1.0, dtype=mx.float32)
    )


def group_relative_advantages(
    returns: mx.array,
    group_ids: mx.array,
    *,
    valid_mask: mx.array | None = None,
    eps: float = 1e-6,
) -> mx.array:
    """Normalize real branch returns within each rollout sibling group."""

    if returns.ndim != 1 or group_ids.shape != returns.shape:
        raise ValueError("returns and group_ids must have equal [N] shape")
    valid = (
        mx.ones(returns.shape, dtype=mx.bool_)
        if valid_mask is None
        else valid_mask.astype(mx.bool_)
    )
    if valid.shape != returns.shape:
        raise ValueError("valid_mask must match returns")
    values = returns.astype(mx.float32)
    same_group = group_ids[:, None] == group_ids[None, :]
    group_mask = same_group & valid[:, None] & valid[None, :]
    weights = group_mask.astype(mx.float32)
    counts = mx.maximum(
        mx.sum(weights, axis=1), mx.array(1.0, dtype=mx.float32)
    )
    means = mx.sum(weights * values[None, :], axis=1) / counts
    centered = values - means
    variances = mx.sum(
        weights * centered[None, :] * centered[None, :], axis=1
    ) / counts
    advantages = centered / mx.sqrt(variances + eps)
    advantages = mx.where(valid, advantages, mx.zeros_like(advantages))
    return mx.stop_gradient(advantages.astype(mx.float32))


def group_relative_objective(
    current_logprobs: mx.array,
    advantages: mx.array,
    *,
    valid_mask: mx.array | None = None,
    behavior_logprobs: mx.array | None = None,
    has_behavior_logprob: mx.array | None = None,
    reference_logprobs: mx.array | None = None,
    has_reference_logprob: mx.array | None = None,
    clip_ratio: float = 0.2,
    kl_coefficient: float = 0.0,
) -> GroupRelativeLossResult:
    """Compute strict clipped GRPO only when behavior logprobs exist.

    Without behavior logprobs this is explicitly reported as group-relative
    ranking/distillation: no pseudo-ratio and no clipping are manufactured.
    Reference KL is included only for entries with real reference logprobs.
    """

    if current_logprobs.ndim != 1 or advantages.shape != current_logprobs.shape:
        raise ValueError("current_logprobs and advantages must have equal [N] shape")
    if clip_ratio < 0.0 or kl_coefficient < 0.0:
        raise ValueError("clip_ratio and kl_coefficient must be non-negative")
    valid = (
        mx.ones(current_logprobs.shape, dtype=mx.bool_)
        if valid_mask is None
        else valid_mask.astype(mx.bool_)
    )
    current = current_logprobs.astype(mx.float32)
    advantage = mx.stop_gradient(advantages.astype(mx.float32))
    zero = mx.array(0.0, dtype=mx.float32)

    if behavior_logprobs is not None:
        if behavior_logprobs.shape != current_logprobs.shape:
            raise ValueError("behavior_logprobs must match current_logprobs")
        behavior_valid = (
            valid
            if has_behavior_logprob is None
            else valid & has_behavior_logprob.astype(mx.bool_)
        )
        ratio = mx.exp(current - behavior_logprobs.astype(mx.float32))
        clipped = mx.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        surrogate = mx.minimum(ratio * advantage, clipped * advantage)
        policy_loss = -_masked_mean(surrogate, behavior_valid)
        clip_fraction = _masked_mean(
            (mx.abs(ratio - 1.0) > clip_ratio).astype(mx.float32),
            behavior_valid,
        )
        mode = "grpo_strict"
    else:
        policy_loss = -_masked_mean(current * advantage, valid)
        clip_fraction = zero
        mode = "group_relative_ranking_distillation"

    kl_loss = zero
    if reference_logprobs is not None:
        if reference_logprobs.shape != current_logprobs.shape:
            raise ValueError("reference_logprobs must match current_logprobs")
        reference_valid = (
            valid
            if has_reference_logprob is None
            else valid & has_reference_logprob.astype(mx.bool_)
        )
        log_ratio = reference_logprobs.astype(mx.float32) - current
        per_item_kl = mx.exp(log_ratio) - log_ratio - 1.0
        kl_loss = _masked_mean(per_item_kl, reference_valid)
        if behavior_logprobs is None:
            mode = "group_relative_ranking_distillation_with_reference_kl"

    loss = policy_loss + kl_coefficient * kl_loss
    return GroupRelativeLossResult(
        loss=loss.astype(mx.float32),
        policy_loss=policy_loss.astype(mx.float32),
        kl_loss=kl_loss.astype(mx.float32),
        clip_fraction=clip_fraction.astype(mx.float32),
        mode=mode,
    )


def build_prospective_planner_mlx(
    config: ProspectivePlannerConfig | None = None,
) -> ProspectivePlannerMLX:
    """Build the isolated FP16 MLX prospective planner."""

    return ProspectivePlannerMLX(config)


def prospective_mlx_checkpoint_payload(
    model: ProspectivePlannerMLX,
) -> dict:
    """Build the strict lateral checkpoint consumed by the PyTorch converter."""

    mx.eval(model.parameters())
    invalid = {
        path: str(parameter.dtype)
        for path, parameter in nn.utils.tree_flatten(model.parameters())
        if parameter.dtype != mx.float16
    }
    if invalid:
        raise ValueError(f"prospective planner has non-FP16 parameters: {invalid}")
    return {
        **prospective_checkpoint_metadata(model.config),
        "model": model.parameters(),
    }


def save_prospective_mlx_checkpoint(
    path: str,
    model: ProspectivePlannerMLX,
) -> None:
    """Atomically save the isolated planner without touching a trunk checkpoint."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "wb") as handle:
        pickle.dump(prospective_mlx_checkpoint_payload(model), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


__all__ = [
    "FP32LayerNormMLX",
    "GroupRelativeLossResult",
    "ProspectiveAttentionMLX",
    "ProspectivePlannerLayerMLX",
    "ProspectivePlannerMLX",
    "apply_rope_nd",
    "build_prospective_planner_mlx",
    "group_relative_advantages",
    "group_relative_objective",
    "prospective_mlx_checkpoint_payload",
    "rope_nd_angles",
    "save_prospective_mlx_checkpoint",
]
