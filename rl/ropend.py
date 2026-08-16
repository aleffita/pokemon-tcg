"""Small RoPE-ND operator for the recurrent Stage 4 PyTorch policy.

The current policy represents one logical decision as a token set and carries
temporal content between decisions in recurrent scratch registers.  RoPE-ND
therefore uses the registers as the relational anchor (coordinate zero) and
assigns the current turn/decision/substep coordinate to state and option
tokens.  Learned content embeddings are unchanged.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn


ROPEND_VERSION = 1
DEFAULT_AXES = ("turn", "logical_decision", "substep")
DEFAULT_PAIR_COUNTS = (4, 4, 4)


def validate_ropend_config(config: dict[str, Any], head_dim: int) -> dict[str, Any]:
    """Validate and canonicalize the serialized RoPE-ND architecture contract."""
    if not isinstance(config, dict):
        raise TypeError("RoPE-ND config must be an object")
    version = int(config.get("version", ROPEND_VERSION))
    if version != ROPEND_VERSION:
        raise ValueError(f"unsupported RoPE-ND version {version}")
    axes = tuple(str(value) for value in config.get("axes", DEFAULT_AXES))
    pair_counts = tuple(int(value) for value in config.get("pair_counts", DEFAULT_PAIR_COUNTS))
    if axes != DEFAULT_AXES:
        raise ValueError(f"unsupported RoPE-ND axes {axes!r}; expected {DEFAULT_AXES!r}")
    if len(pair_counts) != len(axes) or any(value < 0 for value in pair_counts):
        raise ValueError("RoPE-ND pair_counts must be non-negative and match axes")
    rotary_dim = 2 * sum(pair_counts)
    if rotary_dim <= 0 or rotary_dim > int(head_dim):
        raise ValueError(
            f"RoPE-ND rotary_dim={rotary_dim} must be in [2, head_dim={head_dim}]"
        )
    base = float(config.get("base", 10_000.0))
    if not math.isfinite(base) or base <= 1.0:
        raise ValueError("RoPE-ND base must be finite and greater than one")
    init_scale = float(config.get("init_scale", 0.0))
    if not math.isfinite(init_scale):
        raise ValueError("RoPE-ND init_scale must be finite")
    anchor = str(config.get("scratch_anchor", "zero"))
    if anchor != "zero":
        raise ValueError("only zero-anchored recurrent scratch coordinates are supported")
    return {
        "version": version,
        "axes": list(axes),
        "pair_counts": list(pair_counts),
        "base": base,
        "init_scale": init_scale,
        "scratch_anchor": anchor,
        "rotary_dim": rotary_dim,
    }


def default_ropend_config(*, init_scale: float = 0.0) -> dict[str, Any]:
    return {
        "version": ROPEND_VERSION,
        "axes": list(DEFAULT_AXES),
        "pair_counts": list(DEFAULT_PAIR_COUNTS),
        "base": 10_000.0,
        "init_scale": float(init_scale),
        "scratch_anchor": "zero",
    }


def temporal_coordinates(
    observation: dict[str, torch.Tensor],
    *,
    state_tokens: int,
    scratch_tokens: int,
    option_tokens: int,
) -> torch.Tensor:
    """Build [B,S,3] coordinates from existing Stage 4 scalar inputs.

    ``cls_scalars`` already carries normalized turn, turnActionCount and the
    number of picks in the current multi-select.  No encoder/schema expansion
    is required.  Current state/options share the current coordinate while
    recurrent scratch registers stay at the origin, exposing their relative
    phase to every attention layer.
    """
    scalars = observation["cls_scalars"]
    if scalars.ndim != 2 or scalars.shape[1] < 18:
        raise ValueError("RoPE-ND requires batched Stage 4 cls_scalars with width >= 18")
    current = torch.stack(
        (
            scalars[:, 0] * 50.0,
            scalars[:, 1] * 20.0,
            scalars[:, 17] * 5.0,
        ),
        dim=-1,
    )
    batch = scalars.shape[0]
    state = current[:, None, :].expand(batch, int(state_tokens), -1)
    scratch = current.new_zeros(batch, int(scratch_tokens), len(DEFAULT_AXES))
    options = current[:, None, :].expand(batch, int(option_tokens), -1)
    return torch.cat((state, scratch, options), dim=1)


def _axis_frequencies(
    pair_count: int,
    *,
    base: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if pair_count == 0:
        return torch.empty(0, dtype=dtype, device=device)
    index = torch.arange(pair_count, dtype=dtype, device=device)
    return torch.pow(torch.as_tensor(base, dtype=dtype, device=device), -index / pair_count)


def apply_ropend(
    tensor: torch.Tensor,
    coordinates: torch.Tensor,
    axis_scales: torch.Tensor,
    pair_counts: Sequence[int],
    *,
    base: float = 10_000.0,
) -> torch.Tensor:
    """Apply block-diagonal multidimensional rotary phase to [B,H,S,Dh]."""
    if tensor.ndim != 4:
        raise ValueError("RoPE-ND tensor must have shape [B,H,S,Dh]")
    if coordinates.ndim != 3 or coordinates.shape[:2] != (tensor.shape[0], tensor.shape[2]):
        raise ValueError("RoPE-ND coordinates must have shape [B,S,A]")
    if coordinates.shape[-1] != len(pair_counts) or axis_scales.numel() != len(pair_counts):
        raise ValueError("RoPE-ND axis dimensions do not match pair_counts")
    rotary_dim = 2 * sum(int(value) for value in pair_counts)
    if rotary_dim > tensor.shape[-1]:
        raise ValueError("RoPE-ND allocation exceeds attention head dimension")

    pieces: list[torch.Tensor] = []
    offset = 0
    for axis, pair_count_value in enumerate(pair_counts):
        pair_count = int(pair_count_value)
        width = 2 * pair_count
        if width == 0:
            continue
        block = tensor[..., offset : offset + width].reshape(
            tensor.shape[0], tensor.shape[1], tensor.shape[2], pair_count, 2
        )
        frequency = _axis_frequencies(
            pair_count, base=base, dtype=tensor.dtype, device=tensor.device
        )
        angle = (
            coordinates[..., axis].to(dtype=tensor.dtype)[:, None, :, None]
            * axis_scales[axis].to(dtype=tensor.dtype)
            * frequency[None, None, None, :]
        )
        cos, sin = torch.cos(angle), torch.sin(angle)
        even, odd = block[..., 0], block[..., 1]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        pieces.append(rotated.reshape(*tensor.shape[:-1], width))
        offset += width
    if offset < tensor.shape[-1]:
        pieces.append(tensor[..., offset:])
    return torch.cat(pieces, dim=-1)


def encoder_forward_ropend(
    encoder: nn.TransformerEncoder,
    sequence: torch.Tensor,
    padding_mask: torch.Tensor,
    coordinates: torch.Tensor,
    axis_scales: torch.Tensor,
    pair_counts: Sequence[int],
    *,
    base: float = 10_000.0,
) -> torch.Tensor:
    """Run the existing encoder weights with RoPE-ND inserted after Q/K projection."""
    if sequence.ndim != 3 or padding_mask.shape != sequence.shape[:2]:
        raise ValueError("RoPE-ND encoder inputs have incompatible shapes")
    x = sequence
    allowed_keys = (~padding_mask)[:, None, None, :]
    for layer in encoder.layers:
        if layer.norm_first:
            raise ValueError("RoPE-ND Stage 4 path expects post-norm encoder layers")
        attention = layer.self_attn
        embed_dim = x.shape[-1]
        heads = int(attention.num_heads)
        head_dim = embed_dim // heads
        projected = nn.functional.linear(x, attention.in_proj_weight, attention.in_proj_bias)
        q, k, v = projected.chunk(3, dim=-1)
        q = q.view(x.shape[0], x.shape[1], heads, head_dim).transpose(1, 2)
        k = k.view(x.shape[0], x.shape[1], heads, head_dim).transpose(1, 2)
        v = v.view(x.shape[0], x.shape[1], heads, head_dim).transpose(1, 2)
        q = apply_ropend(q, coordinates, axis_scales, pair_counts, base=base)
        k = apply_ropend(k, coordinates, axis_scales, pair_counts, base=base)
        attended = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed_keys,
            dropout_p=float(attention.dropout) if layer.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view_as(x)
        attended = nn.functional.linear(
            attended, attention.out_proj.weight, attention.out_proj.bias
        )
        x = layer.norm1(x + layer.dropout1(attended))
        feedforward = layer.linear2(
            layer.dropout(layer.activation(layer.linear1(x)))
        )
        x = layer.norm2(x + layer.dropout2(feedforward))
    if encoder.norm is not None:
        x = encoder.norm(x)
    return x


__all__ = [
    "DEFAULT_AXES",
    "DEFAULT_PAIR_COUNTS",
    "ROPEND_VERSION",
    "apply_ropend",
    "default_ropend_config",
    "encoder_forward_ropend",
    "temporal_coordinates",
    "validate_ropend_config",
]
