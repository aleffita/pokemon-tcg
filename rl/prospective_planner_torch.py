"""PyTorch FP16 mirror of the lateral RoPE-ND prospective planner.

This module is intentionally disconnected from ``agent/main.py`` and the
current TokenTransformer.  It implements only the versioned prospective
planner contract shared with the MLX training-side module.
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.prospective_schema import (
    BRANCH_POLICY_SCORE,
    BRANCH_VALID,
    EXPECTED_PRIZES,
    KO_LOGIT,
    KO_PROBABILITY,
    N_PROSPECTIVE_AXES,
    PRIZE_LOGIT,
    PROSPECTIVE_COORD_SCHEMA_VERSION,
    PROSPECTIVE_PLANNER_VERSION,
    SCALAR_RETURN,
    SCALAR_VALUE,
    TERMINAL_LOGIT,
    TERMINAL_PROBABILITY,
    UNCERTAINTY,
    UNCERTAINTY_LOGIT,
    ProspectivePlannerConfig,
)


PROSPECTIVE_TORCH_CHECKPOINT_FORMAT = "ptcg-prospective-torch-fp16-v1"


def _config_from_dict(raw: dict[str, Any]) -> ProspectivePlannerConfig:
    if not isinstance(raw, dict):
        raise ValueError("prospective planner config must be an object")
    config = ProspectivePlannerConfig(**raw)
    config.validate()
    return config


def _validate_checkpoint_contract(payload: dict[str, Any]) -> ProspectivePlannerConfig:
    if payload.get("planner_version") != PROSPECTIVE_PLANNER_VERSION:
        raise ValueError(
            f"unsupported planner version {payload.get('planner_version')!r}"
        )
    if payload.get("coord_schema_version") != PROSPECTIVE_COORD_SCHEMA_VERSION:
        raise ValueError(
            "unsupported prospective coordinate schema "
            f"{payload.get('coord_schema_version')!r}"
        )
    config = _config_from_dict(payload.get("config"))
    if config.version != payload["planner_version"]:
        raise ValueError("planner version disagrees with serialized config")
    if config.coord_schema_version != payload["coord_schema_version"]:
        raise ValueError("coordinate schema disagrees with serialized config")
    return config


class ProspectiveAttentionTorch(nn.Module):
    """Multi-head attention with four-axis rotary query/key coordinates."""

    def __init__(self, config: ProspectivePlannerConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.head_dim = config.d_model // config.nhead
        self.axis_dim = self.head_dim // N_PROSPECTIVE_AXES
        self.rope_base = float(config.rope_base)
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        inv_frequency = self.rope_base ** (
            -torch.arange(0, self.axis_dim, 2, dtype=torch.float32)
            / self.axis_dim
        )
        # Keep the canonical frequencies outside module state so a model-wide
        # FP16 cast cannot quantize the FP32 rotary calculation.
        self._inv_frequency_values = tuple(inv_frequency.tolist())

    def _apply_rope_nd(
        self, tensor: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        batch, heads, length, head_dim = tensor.shape
        if head_dim != self.head_dim:
            raise ValueError(
                f"attention head dim {head_dim} != configured {self.head_dim}"
            )
        output_axes = []
        inv_frequency = torch.tensor(
            self._inv_frequency_values,
            dtype=torch.float32,
            device=tensor.device,
        )
        for axis in range(N_PROSPECTIVE_AXES):
            start = axis * self.axis_dim
            stop = start + self.axis_dim
            values = tensor[..., start:stop]
            angle = (
                coordinates[:, None, :, axis].to(torch.float32)[..., None]
                * inv_frequency[None, None, None, :]
            )
            cosine = torch.cos(angle)
            sine = torch.sin(angle)
            even = values[..., 0::2].to(torch.float32)
            odd = values[..., 1::2].to(torch.float32)
            rotated = torch.stack(
                (
                    even * cosine - odd * sine,
                    even * sine + odd * cosine,
                ),
                dim=-1,
            ).flatten(-2)
            output_axes.append(rotated.to(values.dtype))
        output = torch.cat(output_axes, dim=-1)
        return output.reshape(batch, heads, length, self.head_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        coordinates: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape

        def project(layer: nn.Linear) -> torch.Tensor:
            return (
                layer(hidden)
                .reshape(batch, length, self.nhead, self.head_dim)
                .transpose(1, 2)
            )

        query = self._apply_rope_nd(project(self.q_proj), coordinates)
        key = self._apply_rope_nd(project(self.k_proj), coordinates)
        value = project(self.v_proj)
        scores = torch.matmul(
            query.to(torch.float32), key.to(torch.float32).transpose(-2, -1)
        ) / math.sqrt(self.head_dim)
        scores = scores + attention_mask.to(torch.float32)
        probability = torch.softmax(scores, dim=-1)
        probability = torch.nan_to_num(
            probability, nan=0.0, posinf=0.0, neginf=0.0
        ).to(value.dtype)
        attended = torch.matmul(probability, value)
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .reshape(batch, length, self.d_model)
        )
        return self.out_proj(attended)


class FP32LayerNormTorch(nn.Module):
    """FP16 affine parameters with explicit FP32 normalization reductions."""

    def __init__(self, dimensions: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(dimensions, dtype=torch.float16)
        )
        self.bias = nn.Parameter(
            torch.zeros(dimensions, dtype=torch.float16)
        )
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        source_dtype = value.dtype
        value_fp32 = value.to(torch.float32)
        mean = torch.mean(value_fp32, dim=-1, keepdim=True)
        centered = value_fp32 - mean
        variance = torch.mean(centered * centered, dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        affine = (
            normalized * self.weight.to(torch.float32)
            + self.bias.to(torch.float32)
        )
        return affine.to(source_dtype)


class ProspectivePlannerLayerTorch(nn.Module):
    """Pre-normalized prospective Transformer block."""

    def __init__(self, config: ProspectivePlannerConfig) -> None:
        super().__init__()
        self.norm1 = FP32LayerNormTorch(config.d_model)
        self.attention = ProspectiveAttentionTorch(config)
        self.norm2 = FP32LayerNormTorch(config.d_model)
        self.ff1 = nn.Linear(config.d_model, config.ff_dim)
        self.ff2 = nn.Linear(config.ff_dim, config.d_model)

    def forward(
        self,
        hidden: torch.Tensor,
        coordinates: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = hidden + self.attention(
            self.norm1(hidden), coordinates, attention_mask
        )
        hidden = hidden + self.ff2(F.gelu(self.ff1(self.norm2(hidden))))
        return hidden


class ProspectivePlannerTorch(nn.Module):
    """Lateral branch planner with versioned four-axis RoPE-ND coordinates."""

    def __init__(
        self, config: ProspectivePlannerConfig | None = None
    ) -> None:
        super().__init__()
        self.config = config or ProspectivePlannerConfig()
        self.config.validate()
        self.layers = nn.ModuleList(
            [
                ProspectivePlannerLayerTorch(self.config)
                for _ in range(self.config.nlayers)
            ]
        )
        self.policy_head = nn.Linear(self.config.d_model, 1)
        self.return_head = nn.Linear(self.config.d_model, 1)
        self.value_head = nn.Linear(self.config.d_model, 1)
        self.ko_head = nn.Linear(self.config.d_model, 1)
        self.prize_head = nn.Linear(self.config.d_model, 1)
        self.terminal_head = nn.Linear(self.config.d_model, 1)
        self.uncertainty_head = nn.Linear(self.config.d_model, 1)
        self.to(dtype=torch.float16)

    def get_config(self) -> dict[str, Any]:
        return self.config.to_dict()

    def forward(
        self,
        context: torch.Tensor,
        branch_tokens: torch.Tensor,
        coordinates: torch.Tensor,
        attention_mask: torch.Tensor,
        branch_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if context.ndim != 3 or branch_tokens.ndim != 3:
            raise ValueError("context and branch_tokens must have [B,L,D] shape")
        if context.shape[0] != branch_tokens.shape[0]:
            raise ValueError("context and branch_tokens batch sizes differ")
        if context.shape[2] != self.config.d_model:
            raise ValueError("context d_model does not match planner config")
        if branch_tokens.shape[2] != self.config.d_model:
            raise ValueError("branch token d_model does not match planner config")
        batch, context_length, _ = context.shape
        branch_count = branch_tokens.shape[1]
        total_length = context_length + branch_count
        if coordinates.shape != (batch, total_length, N_PROSPECTIVE_AXES):
            raise ValueError(
                "coordinates must have shape "
                f"{(batch, total_length, N_PROSPECTIVE_AXES)}"
            )
        if attention_mask.shape != (batch, 1, total_length, total_length):
            raise ValueError(
                "attention_mask must have shape "
                f"{(batch, 1, total_length, total_length)}"
            )
        if branch_valid.shape != (batch, branch_count):
            raise ValueError(
                f"branch_valid must have shape {(batch, branch_count)}"
            )
        if coordinates.dtype not in (torch.int32, torch.int64):
            raise TypeError("coordinates must be an integer tensor")
        if torch.any(coordinates < 0):
            raise ValueError("prospective coordinates must be non-negative")
        if not torch.all(
            torch.isneginf(attention_mask) | (attention_mask == 0)
        ):
            raise ValueError("attention_mask must contain only 0 and -inf")

        parameter_dtype = self.policy_head.weight.dtype
        hidden = torch.cat(
            (
                context.to(parameter_dtype),
                branch_tokens.to(parameter_dtype),
            ),
            dim=1,
        )
        for layer in self.layers:
            hidden = layer(hidden, coordinates, attention_mask)
        branch_hidden = hidden[:, context_length:]

        policy_score = self.policy_head(branch_hidden).squeeze(-1)
        return_logit = self.return_head(branch_hidden).squeeze(-1)
        value_logit = self.value_head(branch_hidden).squeeze(-1)
        ko_logit = self.ko_head(branch_hidden).squeeze(-1)
        prize_logit = self.prize_head(branch_hidden).squeeze(-1)
        terminal_logit = self.terminal_head(branch_hidden).squeeze(-1)
        uncertainty_logit = self.uncertainty_head(branch_hidden).squeeze(-1)

        scalar_return = torch.tanh(
            return_logit.to(torch.float32)
        ).to(parameter_dtype)
        scalar_value = torch.tanh(
            value_logit.to(torch.float32)
        ).to(parameter_dtype)
        ko_probability = torch.sigmoid(
            ko_logit.to(torch.float32)
        ).to(parameter_dtype)
        expected_prizes = (
            torch.sigmoid(prize_logit.to(torch.float32))
            * self.config.max_prizes
        ).to(parameter_dtype)
        terminal_probability = torch.sigmoid(
            terminal_logit.to(torch.float32)
        ).to(parameter_dtype)
        uncertainty = (
            F.softplus(uncertainty_logit.to(torch.float32))
            + self.config.uncertainty_floor
        ).to(parameter_dtype)

        valid = branch_valid.to(torch.bool)
        zero = torch.zeros((), dtype=parameter_dtype, device=hidden.device)
        minimum = torch.full(
            (), -65504.0, dtype=parameter_dtype, device=hidden.device
        )
        outputs = {
            BRANCH_POLICY_SCORE: torch.where(valid, policy_score, minimum),
            SCALAR_RETURN: torch.where(valid, scalar_return, zero),
            SCALAR_VALUE: torch.where(valid, scalar_value, zero),
            KO_LOGIT: torch.where(valid, ko_logit, zero),
            KO_PROBABILITY: torch.where(valid, ko_probability, zero),
            PRIZE_LOGIT: torch.where(valid, prize_logit, zero),
            EXPECTED_PRIZES: torch.where(valid, expected_prizes, zero),
            TERMINAL_LOGIT: torch.where(valid, terminal_logit, zero),
            TERMINAL_PROBABILITY: torch.where(
                valid, terminal_probability, zero
            ),
            UNCERTAINTY_LOGIT: torch.where(
                valid, uncertainty_logit, zero
            ),
            UNCERTAINTY: torch.where(valid, uncertainty, zero),
            BRANCH_VALID: valid,
        }
        return outputs


def _flatten_checkpoint_tree(
    tree: Any, prefix: str = ""
) -> dict[str, np.ndarray]:
    if isinstance(tree, dict):
        flattened: dict[str, np.ndarray] = {}
        for key, value in tree.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_checkpoint_tree(value, name))
        return flattened
    if isinstance(tree, (list, tuple)):
        flattened = {}
        for index, value in enumerate(tree):
            name = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(_flatten_checkpoint_tree(value, name))
        return flattened
    return {prefix: np.asarray(tree)}


def _strict_state_from_arrays(
    model: ProspectivePlannerTorch,
    arrays: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    expected = model.state_dict()
    missing = set(expected) - set(arrays)
    unexpected = set(arrays) - set(expected)
    if missing or unexpected:
        raise ValueError(
            f"prospective planner state mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    converted: dict[str, torch.Tensor] = {}
    for key, target in expected.items():
        source = np.asarray(arrays[key])
        if np.issubdtype(source.dtype, np.floating) and source.dtype != np.float16:
            raise ValueError(
                f"dtype mismatch for {key}: source {source.dtype} != float16"
            )
        tensor = torch.from_numpy(source)
        if tuple(tensor.shape) != tuple(target.shape):
            raise ValueError(
                f"shape mismatch for {key}: source {tuple(tensor.shape)} "
                f"!= target {tuple(target.shape)}"
            )
        converted[key] = tensor.to(torch.float16)
    return converted


def prospective_torch_checkpoint_payload(
    model: ProspectivePlannerTorch,
) -> dict[str, Any]:
    state_dict = model.state_dict()
    non_fp16 = {
        key: str(value.dtype)
        for key, value in state_dict.items()
        if value.is_floating_point() and value.dtype != torch.float16
    }
    if non_fp16:
        raise ValueError(f"prospective planner has non-FP16 state: {non_fp16}")
    return {
        "format": PROSPECTIVE_TORCH_CHECKPOINT_FORMAT,
        "planner_version": PROSPECTIVE_PLANNER_VERSION,
        "coord_schema_version": PROSPECTIVE_COORD_SCHEMA_VERSION,
        "config": model.get_config(),
        "state_dict": state_dict,
    }


def save_prospective_torch_checkpoint(
    path: str | Path,
    model: ProspectivePlannerTorch,
) -> None:
    torch.save(prospective_torch_checkpoint_payload(model), path)


def load_prospective_torch_checkpoint(
    path: str | Path,
) -> tuple[ProspectivePlannerTorch, ProspectivePlannerConfig]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("prospective PyTorch checkpoint must be an object")
    if payload.get("format") != PROSPECTIVE_TORCH_CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported prospective torch format {payload.get('format')!r}"
        )
    config = _validate_checkpoint_contract(payload)
    model = ProspectivePlannerTorch(config).to(torch.float16)
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("prospective PyTorch checkpoint has no state_dict")
    arrays = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"state_dict[{key!r}] is not a tensor")
        if value.is_floating_point() and value.dtype != torch.float16:
            raise ValueError(
                f"state_dict[{key!r}] is {value.dtype}, expected float16"
            )
        arrays[key] = value.detach().cpu().numpy()
    model.load_state_dict(_strict_state_from_arrays(model, arrays), strict=True)
    model.eval()
    return model, config


def convert_mlx_prospective_checkpoint(
    mlx_path: str | Path,
    torch_path: str | Path,
) -> ProspectivePlannerConfig:
    """Strictly convert the lateral MLX payload with identical parameter paths."""
    with open(mlx_path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("prospective MLX checkpoint must be an object")
    config = _validate_checkpoint_contract(payload)
    model_tree = payload.get("model")
    if model_tree is None:
        raise ValueError("prospective MLX checkpoint has no model")
    model = ProspectivePlannerTorch(config).to(torch.float16)
    arrays = _flatten_checkpoint_tree(model_tree)
    model.load_state_dict(_strict_state_from_arrays(model, arrays), strict=True)
    model.eval()
    save_prospective_torch_checkpoint(torch_path, model)
    return config


__all__ = [
    "FP32LayerNormTorch",
    "PROSPECTIVE_TORCH_CHECKPOINT_FORMAT",
    "ProspectiveAttentionTorch",
    "ProspectivePlannerLayerTorch",
    "ProspectivePlannerTorch",
    "convert_mlx_prospective_checkpoint",
    "load_prospective_torch_checkpoint",
    "prospective_torch_checkpoint_payload",
    "save_prospective_torch_checkpoint",
]
