"""Behavioral-cloning trainer — MLX version (Apple Silicon native).

Same architecture as bc_train.py but uses MLX instead of PyTorch.
Faster on M1/M2 via native Metal GPU (no MPS NaN bug).

FP16-native: numeric features stay float16 end-to-end.
Gradient accumulation: --accum-steps K accumulates K microbatches before update.

Usage:
  uv run tcg-train data/bc_data/bc_2026_07_21 \
      --d-model 128 --static --split-heads --epochs 8 --batch 128
"""

import argparse
import hashlib
import json
import os
import queue
import shutil
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from rl.encoder.card_features import get_card_table
from rl.encoder.enc_constants import OPT_WK
from rl.encoder.encoding import TokenEncoder
from rl.lr_schedule import lr_at
from rl.policy_mlx import build_token_net_mlx
from rl.train_config import load_config

WK_LO, WK_HI = OPT_WK, OPT_WK + 3


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class FP32StateMuon(optim.Muon):
    """Muon with FP32 momentum and FP16 parameter storage."""

    def init_single(self, parameter: mx.array, state: dict) -> None:
        state["v"] = mx.zeros(parameter.shape, dtype=mx.float32)

    def apply_single(
        self, gradient: mx.array, parameter: mx.array, state: dict
    ) -> mx.array:
        updated = super().apply_single(
            gradient.astype(mx.float32), parameter.astype(mx.float32), state
        )
        return updated.astype(parameter.dtype)


class FP32StateAdamW(optim.AdamW):
    """AdamW with FP32 moments and FP16 parameter storage."""

    def init_single(self, parameter: mx.array, state: dict) -> None:
        state["m"] = mx.zeros(parameter.shape, dtype=mx.float32)
        state["v"] = mx.zeros(parameter.shape, dtype=mx.float32)

    def apply_single(
        self, gradient: mx.array, parameter: mx.array, state: dict
    ) -> mx.array:
        updated = super().apply_single(
            gradient.astype(mx.float32), parameter.astype(mx.float32), state
        )
        return updated.astype(parameter.dtype)


class PathSafeMultiOptimizer(optim.MultiOptimizer):
    """MultiOptimizer merge that supports parameter trees containing lists."""

    def apply_gradients(self, gradients: dict, parameters: dict):
        merged_leaves = []
        for optimizer, optimizer_grads in zip(
            self.optimizers, self._split_dictionary(gradients)
        ):
            updated = optimizer.apply_gradients(optimizer_grads, parameters)
            merged_leaves.extend(nn.utils.tree_flatten(updated))
        return nn.utils.tree_unflatten(merged_leaves)


_ADAMW_PARAMETER_PREFIXES = (
    "card_emb.",
    "type_emb.",
    "sel_type_emb.",
    "sel_ctx_emb.",
    "opt_verb_emb.",
    "attack_emb.",
    "type_query.",
    "type_bias.",
    "opt_head.",
    "submit_head.",
    "value_head.",
)


def _use_muon_parameter(path: str, parameter: mx.array) -> bool:
    """Route only hidden 2D projection/Transformer weights to Muon."""
    if parameter.ndim != 2 or not path.endswith(".weight"):
        return False
    return not path.startswith(_ADAMW_PARAMETER_PREFIXES)


def _optimizer_contract(cfg) -> dict:
    """Return the serializable optimizer identity used for safe resume."""
    return {
        "name": cfg.optimizer,
        "muon_momentum": float(cfg.muon_momentum),
        "muon_weight_decay": float(cfg.muon_weight_decay),
        "adamw_betas": [float(value) for value in cfg.adamw_betas],
        "adamw_eps": float(cfg.adamw_eps),
        "adamw_weight_decay": float(cfg.adamw_weight_decay),
        "state_dtype": "float32",
        "parameter_dtype": "float16",
        "routing_version": 1,
    }


def _scheduler_contract(cfg, total_steps: int, warmup_steps: int) -> dict:
    """Return the immutable identity of one scheduler phase."""
    return {
        "schedule": cfg.lr_schedule,
        "base_lr": float(cfg.lr),
        "warmup_steps": int(warmup_steps),
        "min_ratio": float(cfg.lr_min_ratio),
        "total_steps": int(total_steps),
    }


def _build_optimizer(cfg) -> optim.MultiOptimizer:
    """Build the configured Muon/AdamW optimizer topology."""
    if cfg.optimizer != "muon_adamw":
        raise ValueError(f"unsupported optimizer contract: {cfg.optimizer!r}")
    if len(cfg.adamw_betas) != 2:
        raise ValueError("adamw_betas must contain exactly two values")
    muon = FP32StateMuon(
        learning_rate=cfg.lr,
        momentum=cfg.muon_momentum,
        weight_decay=cfg.muon_weight_decay,
    )
    adamw = FP32StateAdamW(
        learning_rate=cfg.lr,
        betas=[float(value) for value in cfg.adamw_betas],
        eps=cfg.adamw_eps,
        weight_decay=cfg.adamw_weight_decay,
    )
    return PathSafeMultiOptimizer([muon, adamw], filters=[_use_muon_parameter])


def _cross_entropy_sum(logits: mx.array, labels: mx.array) -> mx.array:
    """Return an FP32 sum so accumulation can normalize exactly once."""
    return nn.losses.cross_entropy(
        logits.astype(mx.float32), labels, reduction="sum"
    ).astype(mx.float32)


def _sequential_tbptt_loss(
    model: nn.Module,
    observations: dict[str, mx.array],
    labels: mx.array,
    memory_in: mx.array | None,
    decision_lengths: list[int] | None = None,
) -> tuple[mx.array, mx.array]:
    """Unroll one temporal lane while advancing memory once per decision."""
    if decision_lengths is None:
        decision_lengths = [1] * int(labels.shape[0])
    return _batched_sequential_tbptt_loss(
        model,
        [observations],
        [labels],
        [decision_lengths],
        [memory_in],
    )


def _batched_sequential_tbptt_loss(
    model: nn.Module,
    lane_observations: list[dict[str, mx.array]],
    lane_labels: list[mx.array],
    lane_decision_lengths: list[list[int]],
    lane_memory_in: list[mx.array | None],
    *,
    return_logits: bool = False,
):
    """Unroll independent lanes in parallel without sharing recurrent memory."""
    lane_count = len(lane_observations)
    if not (
        lane_count
        == len(lane_labels)
        == len(lane_decision_lengths)
        == len(lane_memory_in)
    ):
        raise ValueError("TBPTT lane inputs must have identical lengths")
    if lane_count == 0:
        raise ValueError("TBPTT temporal batches must contain at least one lane")

    for labels, decision_lengths in zip(lane_labels, lane_decision_lengths):
        if not decision_lengths or any(length <= 0 for length in decision_lengths):
            raise ValueError("every TBPTT decision must contain at least one row")
        if sum(decision_lengths) != int(labels.shape[0]):
            raise ValueError("TBPTT decision lengths do not cover all lane rows")

    loss_sum = mx.array(0.0, dtype=mx.float32)
    lane_memories = list(lane_memory_in)
    lane_offsets = [0] * lane_count
    lane_logits: list[list[mx.array]] = [[] for _ in range(lane_count)]
    max_decisions = max(len(lengths) for lengths in lane_decision_lengths)

    for decision_index in range(max_decisions):
        active_lanes = [
            lane_index
            for lane_index, lengths in enumerate(lane_decision_lengths)
            if decision_index < len(lengths)
        ]
        decision_observations: list[dict[str, mx.array]] = []
        decision_labels: list[mx.array] = []
        repeated_memories: list[mx.array] = []
        row_counts: list[int] = []

        for lane_index in active_lanes:
            row_count = lane_decision_lengths[lane_index][decision_index]
            row_start = lane_offsets[lane_index]
            row_stop = row_start + row_count
            decision_observations.append(
                {
                    key: value[row_start:row_stop]
                    for key, value in lane_observations[lane_index].items()
                }
            )
            decision_labels.append(lane_labels[lane_index][row_start:row_stop])
            memory = lane_memories[lane_index]
            if memory is None:
                memory = model.learned_init.reshape(
                    1, model.scratch_tokens, model.d
                )
            repeated_memories.append(
                mx.broadcast_to(
                    memory,
                    (row_count, model.scratch_tokens, model.d),
                )
            )
            row_counts.append(row_count)
            lane_offsets[lane_index] = row_stop

        combined_observation = {
            key: mx.concatenate(
                [observation[key] for observation in decision_observations],
                axis=0,
            )
            for key in decision_observations[0]
        }
        combined_labels = mx.concatenate(decision_labels, axis=0)
        combined_memory = mx.concatenate(repeated_memories, axis=0)
        logits, _, memory_out = model.logits_value(
            combined_observation, memory_in=combined_memory
        )
        loss_sum = loss_sum + _cross_entropy_sum(logits, combined_labels)

        row_offset = 0
        for lane_index, row_count in zip(active_lanes, row_counts):
            if return_logits:
                lane_logits[lane_index].append(
                    logits[row_offset : row_offset + row_count]
                )
            # Every subrow of one multi-select decision saw the same memory_in.
            # Commit only the final subrow's output to the next decision.
            lane_memories[lane_index] = memory_out[
                row_offset + row_count - 1 : row_offset + row_count
            ]
            row_offset += row_count

    if any(memory is None for memory in lane_memories):
        raise ValueError("every TBPTT lane must advance at least one decision")
    combined_final_memory = mx.concatenate(
        [memory for memory in lane_memories if memory is not None],
        axis=0,
    )
    if return_logits:
        return (
            loss_sum,
            combined_final_memory,
            [mx.concatenate(parts, axis=0) for parts in lane_logits],
        )
    return loss_sum, combined_final_memory


@dataclass(frozen=True)
class _TBPTTChunk:
    """One ordered decision chunk from an episode-side trajectory."""

    group_index: int
    decisions: tuple[np.ndarray, ...]
    is_new_group: bool

    @property
    def row_count(self) -> int:
        return sum(len(rows) for rows in self.decisions)


def _build_tbptt_decision_groups(
    episode_meta: np.ndarray, train_rows: int
) -> list[list[np.ndarray]]:
    """Group ordered rows into decisions, preserving episode and side isolation."""
    row_groups = _build_tbptt_groups(episode_meta, train_rows)
    decision_groups: list[list[np.ndarray]] = []
    step_ids = episode_meta["step_id"]
    for rows in row_groups:
        decisions: list[np.ndarray] = []
        current_rows: list[int] = []
        current_step = None
        seen_steps: set[int] = set()
        for row in rows:
            step = int(step_ids[int(row)])
            if current_step is None or step == current_step:
                current_rows.append(int(row))
                current_step = step
                continue
            if step in seen_steps:
                raise ValueError(
                    "episode_meta.step_id is non-contiguous within an episode-side"
                )
            seen_steps.add(int(current_step))
            decisions.append(np.asarray(current_rows, dtype=np.int64))
            current_rows = [int(row)]
            current_step = step
        if current_rows:
            decisions.append(np.asarray(current_rows, dtype=np.int64))
        decision_groups.append(decisions)
    return decision_groups


def _build_tbptt_plan(
    decision_groups: list[list[np.ndarray]],
    chunk_size: int,
    row_budget: int,
) -> list[list[_TBPTTChunk]]:
    """Pack independent trajectory chunks under an exact row budget."""
    if chunk_size <= 0:
        raise ValueError("TBPTT chunk_size must be positive")
    if row_budget <= 0:
        raise ValueError("TBPTT row_budget must be positive")

    # A chunk is bounded by both temporal horizon and physical row budget.
    # Decisions remain indivisible because all autoregressive rows of one
    # engine decision must read the same memory_in.
    chunks_by_group: list[list[_TBPTTChunk]] = []
    for group_index, decisions in enumerate(decision_groups):
        group_chunks: list[_TBPTTChunk] = []
        current: list[np.ndarray] = []
        current_rows = 0
        for decision in decisions:
            decision_rows = len(decision)
            if decision_rows > row_budget:
                raise ValueError(
                    "one engine decision exceeds the physical TBPTT row budget: "
                    f"group={group_index}, rows={decision_rows}, "
                    f"row_budget={row_budget}. Increase batch_size to at least "
                    "the largest autoregressive decision."
                )
            if current and (
                len(current) >= chunk_size
                or current_rows + decision_rows > row_budget
            ):
                group_chunks.append(
                    _TBPTTChunk(
                        group_index=group_index,
                        decisions=tuple(current),
                        is_new_group=(len(group_chunks) == 0),
                    )
                )
                current = []
                current_rows = 0
            current.append(decision)
            current_rows += decision_rows
        if current:
            group_chunks.append(
                _TBPTTChunk(
                    group_index=group_index,
                    decisions=tuple(current),
                    is_new_group=(len(group_chunks) == 0),
                )
            )
        chunks_by_group.append(group_chunks)

    plan: list[list[_TBPTTChunk]] = []
    max_chunks = max((len(chunks) for chunks in chunks_by_group), default=0)
    for chunk_index in range(max_chunks):
        temporal_batch: list[_TBPTTChunk] = []
        batch_rows = 0
        for group_chunks in chunks_by_group:
            if chunk_index >= len(group_chunks):
                continue
            chunk = group_chunks[chunk_index]
            if temporal_batch and batch_rows + chunk.row_count > row_budget:
                plan.append(temporal_batch)
                temporal_batch = []
                batch_rows = 0
            temporal_batch.append(chunk)
            batch_rows += chunk.row_count
            if batch_rows > row_budget:
                raise AssertionError("TBPTT planner exceeded its physical row budget")
        if temporal_batch:
            plan.append(temporal_batch)
    return plan


def _to_fp32_grads(grads):
    """Materialize gradient leaves in FP32 before accumulation."""
    return nn.utils.tree_map(
        lambda gradient: (
            gradient.astype(mx.float32) if gradient is not None else gradient
        ),
        grads,
    )


def _validate_optimizer_state_dtypes(state: dict) -> None:
    """Reject moment buffers that violate the FP32 optimizer contract."""
    moments = [
        value
        for path, value in nn.utils.tree_flatten(state)
        if path.endswith(".m") or path.endswith(".v")
    ]
    if not moments:
        raise ValueError("optimizer state contains no moment buffers")
    invalid = {str(value.dtype) for value in moments if value.dtype != mx.float32}
    if invalid:
        raise ValueError(f"optimizer moments must be FP32, found {sorted(invalid)}")


def _format_compact_duration(seconds: float | None) -> str:
    """Format a duration for the human-readable ETA field."""
    if seconds is None:
        return "--"
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{seconds:02d}s"


def _ceil_div(numerator: int, denominator: int) -> int:
    """Return ceil(numerator / denominator) for non-negative work counts."""
    if numerator < 0:
        raise ValueError(f"numerator must be non-negative, got {numerator}")
    if denominator <= 0:
        raise ValueError(f"denominator must be positive, got {denominator}")
    return (numerator + denominator - 1) // denominator


def _standard_microbatch_count(
    slab_bounds: list[tuple[int, int]], batch_size: int
) -> int:
    """Count the batches actually yielded across all slab boundaries."""
    return sum(_ceil_div(stop - start, batch_size) for start, stop in slab_bounds)


def _build_tbptt_groups(episode_meta: np.ndarray, train_rows: int) -> list[np.ndarray]:
    """Build the exact ordered row groups consumed by the TBPTT generator."""
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    episode_ids = episode_meta["episode_id"][:train_rows]
    sides = episode_meta["side"][:train_rows]
    for row_index, (episode_id, side) in enumerate(zip(episode_ids, sides)):
        groups[(str(episode_id), int(side))].append(row_index)
    return [np.asarray(groups[key], dtype=np.int64) for key in sorted(groups)]


def _tbptt_microbatch_count(groups: list[np.ndarray], chunk_size: int) -> int:
    """Count chunks yielded by TBPTT, including each partial final chunk."""
    return sum(_ceil_div(len(rows), chunk_size) for rows in groups)


def read_rows(arr: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Rows [start:stop) — buffered sequential read for memmap, slice for in-RAM."""
    if not isinstance(arr, np.memmap):
        return np.asarray(arr[start:stop])
    rowel = int(np.prod(arr.shape[1:], dtype=np.int64))
    flat = np.fromfile(
        arr.filename,
        dtype=arr.dtype,
        count=(stop - start) * rowel,
        offset=arr.offset + start * rowel * arr.dtype.itemsize,
    )
    return flat.reshape((stop - start,) + arr.shape[1:])


def _bool_arg(s: str) -> bool:
    """Parse 'true'/'false'/'1'/'0'/'yes'/'no' for boolean CLI args."""
    return s.lower() in ("true", "1", "yes")


def main() -> None:
    p = argparse.ArgumentParser(
        description="BC trainer — MLX Metal GPU. All fields overridable via CLI or config JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Config hierarchy: CLI args > --config file > configs/train_config.json > defaults",
    )
    p.add_argument(
        "data",
        nargs="?",
        default=None,
        help="Dataset directory (default: most recent in data_dir)",
    )
    p.add_argument("--config", default=None, help="Path to JSON config file")
    # Architecture
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--nhead", type=int, default=None)
    p.add_argument("--nlayers", type=int, default=None)
    p.add_argument("--ff", type=int, default=None, help="FFN width (default 4*d_model)")
    p.add_argument(
        "--static",
        type=_bool_arg,
        default=None,
        metavar="true|false",
        help="Static card features (default: config or true)",
    )
    p.add_argument(
        "--split-heads",
        type=_bool_arg,
        default=None,
        metavar="true|false",
        help="Dedicated value/submit heads",
    )
    p.add_argument(
        "--structured",
        type=_bool_arg,
        default=None,
        metavar="true|false",
        help="Verb-conditioned action head",
    )
    p.add_argument(
        "--scratch-registers",
        type=int,
        default=None,
        help="Number of scratch/workspace tokens (default: config or 4)",
    )
    p.add_argument(
        "--value-atoms",
        type=int,
        default=None,
        help="Categorical value head atom count",
    )
    p.add_argument(
        "--value-vmax",
        type=float,
        default=None,
        help="Categorical value head max absolute value",
    )
    # Training
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument(
        "--accum-steps",
        type=int,
        default=None,
        help="Gradient accumulation microbatches",
    )
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lr-schedule", choices=["cosine", "linear", "none"], default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--lr-min-ratio", type=float, default=None)
    p.add_argument("--max-grad-norm", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--optimizer", choices=["muon_adamw"], default=None)
    p.add_argument("--optimizer-state", choices=["reset", "resume"], default=None)
    p.add_argument("--scheduler-state", choices=["reset", "resume"], default=None)
    p.add_argument("--scheduler-total-steps", type=int, default=None)
    p.add_argument("--muon-momentum", type=float, default=None)
    p.add_argument("--muon-weight-decay", type=float, default=None)
    p.add_argument("--adamw-betas", type=float, nargs=2, default=None)
    p.add_argument("--adamw-eps", type=float, default=None)
    p.add_argument("--adamw-weight-decay", type=float, default=None)
    p.add_argument(
        "--tbptt-chunk", type=int, default=None, help="TBPTT chunk size (0=disabled)"
    )
    # Trainer options
    p.add_argument(
        "--compile",
        type=_bool_arg,
        default=None,
        metavar="true|false",
        help="mx.compile the loss function",
    )
    p.add_argument(
        "--prefetch",
        type=_bool_arg,
        default=None,
        metavar="true|false",
        help="Prefetch next slab on CPU while GPU trains",
    )
    p.add_argument("--log-interval", type=int, default=None)
    # Data
    p.add_argument("--val-frac", type=float, default=None)
    p.add_argument("--val-batch-size", type=int, default=None)
    p.add_argument("--slab-rows", type=int, default=None)
    p.add_argument(
        "--max-rows", type=int, default=None, help="Max training rows (0=all)"
    )
    p.add_argument(
        "--bc-would-ko",
        type=_bool_arg,
        default=None,
        metavar="true|false",
        help="Require or disable the dataset would-KO contract",
    )
    p.add_argument("--bc-wk-nvar", type=int, default=None)
    p.add_argument("--zero-wouldko", action="store_true", default=None)
    p.add_argument("--dedup", action="store_true", default=None)
    # Output
    p.add_argument("--out", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument(
        "--phase-id",
        default=None,
        help="Stable orchestration phase identity stored in every checkpoint",
    )
    p.add_argument(
        "--export-final",
        default=None,
        help=(
            "Explicitly copy the best checkpoint to this path after the run. "
            "No implicit model/bc_model write occurs when omitted."
        ),
    )
    p.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=None,
        help="Save a numbered checkpoint every N epochs (default: config or 1)",
    )
    a = p.parse_args()

    # Load config: CLI > config file > defaults
    # Every TrainConfig field is overridable from CLI
    _CLI_MAP = {
        "d_model": "d_model",
        "nhead": "nhead",
        "nlayers": "nlayers",
        "ff": "ff_dim",
        "static": "static",
        "split_heads": "split_heads",
        "structured": "structured",
        "scratch_registers": "scratch_registers",
        "value_atoms": "value_atoms",
        "value_vmax": "value_vmax",
        "epochs": "epochs",
        "batch": "batch_size",
        "accum_steps": "accum_steps",
        "lr": "lr",
        "lr_schedule": "lr_schedule",
        "warmup_steps": "warmup_steps",
        "lr_min_ratio": "lr_min_ratio",
        "max_grad_norm": "max_grad_norm",
        "seed": "seed",
        "optimizer": "optimizer",
        "optimizer_state": "optimizer_state",
        "scheduler_state": "scheduler_state",
        "scheduler_total_steps": "scheduler_total_steps",
        "muon_momentum": "muon_momentum",
        "muon_weight_decay": "muon_weight_decay",
        "adamw_betas": "adamw_betas",
        "adamw_eps": "adamw_eps",
        "adamw_weight_decay": "adamw_weight_decay",
        "tbptt_chunk": "tbptt_chunk",
        "compile": "compile",
        "prefetch": "prefetch",
        "log_interval": "log_interval",
        "val_frac": "val_frac",
        "val_batch_size": "val_batch_size",
        "slab_rows": "slab_rows",
        "max_rows": "max_rows",
        "bc_would_ko": "bc_would_ko",
        "bc_wk_nvar": "bc_wk_nvar",
        "checkpoint_every_epochs": "checkpoint_every_epochs",
    }
    cli = {}
    for cli_attr, cfg_key in _CLI_MAP.items():
        val = getattr(a, cli_attr, None)
        if val is not None:
            cli[cfg_key] = val
    cfg = load_config(cli_overrides=cli, config_path=a.config)

    # Apply config values (config defaults > hardcoded defaults for flags)
    a.d_model = a.d_model if a.d_model is not None else cfg.d_model
    a.nhead = a.nhead if a.nhead is not None else cfg.nhead
    a.nlayers = a.nlayers if a.nlayers is not None else cfg.nlayers
    a.ff = a.ff if a.ff is not None else cfg.ff_dim
    a.static = a.static if a.static is not None else cfg.static
    a.split_heads = a.split_heads if a.split_heads is not None else cfg.split_heads
    a.structured = a.structured if a.structured is not None else cfg.structured
    a.scratch_registers = (
        a.scratch_registers
        if a.scratch_registers is not None
        else cfg.scratch_registers
    )
    a.value_atoms = a.value_atoms if a.value_atoms is not None else cfg.value_atoms
    a.value_vmax = a.value_vmax if a.value_vmax is not None else cfg.value_vmax
    a.epochs = a.epochs if a.epochs is not None else cfg.epochs
    a.batch = a.batch if a.batch is not None else cfg.batch_size
    a.accum_steps = a.accum_steps if a.accum_steps is not None else cfg.accum_steps
    a.lr = a.lr if a.lr is not None else cfg.lr
    a.lr_schedule = a.lr_schedule if a.lr_schedule is not None else cfg.lr_schedule
    a.warmup_steps = a.warmup_steps if a.warmup_steps is not None else cfg.warmup_steps
    a.lr_min_ratio = a.lr_min_ratio if a.lr_min_ratio is not None else cfg.lr_min_ratio
    a.max_grad_norm = (
        a.max_grad_norm if a.max_grad_norm is not None else cfg.max_grad_norm
    )
    a.seed = a.seed if a.seed is not None else cfg.seed
    a.optimizer = a.optimizer if a.optimizer is not None else cfg.optimizer
    a.optimizer_state = (
        a.optimizer_state if a.optimizer_state is not None else cfg.optimizer_state
    )
    a.scheduler_state = (
        a.scheduler_state if a.scheduler_state is not None else cfg.scheduler_state
    )
    a.scheduler_total_steps = (
        a.scheduler_total_steps
        if a.scheduler_total_steps is not None
        else cfg.scheduler_total_steps
    )
    a.muon_momentum = (
        a.muon_momentum if a.muon_momentum is not None else cfg.muon_momentum
    )
    a.muon_weight_decay = (
        a.muon_weight_decay
        if a.muon_weight_decay is not None
        else cfg.muon_weight_decay
    )
    a.adamw_betas = (
        a.adamw_betas if a.adamw_betas is not None else cfg.adamw_betas
    )
    a.adamw_eps = a.adamw_eps if a.adamw_eps is not None else cfg.adamw_eps
    a.adamw_weight_decay = (
        a.adamw_weight_decay
        if a.adamw_weight_decay is not None
        else cfg.adamw_weight_decay
    )
    a.tbptt_chunk = a.tbptt_chunk if a.tbptt_chunk is not None else cfg.tbptt_chunk
    a.compile = a.compile if a.compile is not None else cfg.compile
    a.prefetch = a.prefetch if a.prefetch is not None else cfg.prefetch
    a.log_interval = a.log_interval if a.log_interval is not None else cfg.log_interval
    a.val_frac = a.val_frac if a.val_frac is not None else cfg.val_frac
    a.val_batch_size = (
        a.val_batch_size if a.val_batch_size is not None else cfg.val_batch_size
    )
    a.slab_rows = a.slab_rows if a.slab_rows is not None else cfg.slab_rows
    a.max_rows = a.max_rows if a.max_rows is not None else cfg.max_rows
    a.checkpoint_every_epochs = (
        a.checkpoint_every_epochs
        if a.checkpoint_every_epochs is not None
        else cfg.checkpoint_every_epochs
    )
    if a.checkpoint_every_epochs <= 0:
        raise ValueError(
            f"checkpoint_every_epochs must be positive, got {a.checkpoint_every_epochs}"
        )
    a.out = a.out or "model/checkpoint/bc_best_mlx.pkl"

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs("model/bc_model", exist_ok=True)

    # MLX auto-detecta GPU (Metal) — sem device management!
    print(f"[bc-train-mlx] device={mx.default_device()}", flush=True)

    # Auto-detect dataset: use most recent if not specified
    if a.data is None:
        data_dir = Path(cfg.data_dir)
        datasets = sorted(
            [
                d
                for d in data_dir.iterdir()
                if d.is_dir() and (d / "__labels__.npy").exists()
            ]
        )
        if not datasets:
            print(f"[bc-train-mlx] ERROR: No datasets found in {data_dir}")
            return
        a.data = str(datasets[-1])
        print(f"[bc-train-mlx] Auto-detected dataset: {a.data}", flush=True)

    # --- data loading (igual ao PyTorch) ---
    mmapped = os.path.isdir(a.data)
    if mmapped:
        d = {
            f[:-4]: np.load(os.path.join(a.data, f), mmap_mode="r")
            for f in sorted(os.listdir(a.data))
            if f.endswith(".npy")
        }
    else:
        z = np.load(a.data)
        d = {k: z[k] for k in z.files}

    dataset_manifest: dict = {}
    manifest_path = os.path.join(a.data, "dataset_manifest.json") if mmapped else None
    if manifest_path and os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            dataset_manifest = json.load(handle)
        if not isinstance(dataset_manifest, dict):
            raise ValueError("dataset_manifest.json must contain an object")
    manifest_would_ko = dataset_manifest.get("would_ko")
    if manifest_would_ko is not None and not isinstance(manifest_would_ko, dict):
        raise ValueError("dataset manifest would_ko contract must be an object")
    if manifest_would_ko is not None:
        dataset_would_ko = bool(manifest_would_ko.get("enabled", False))
        dataset_would_ko_nvar = int(manifest_would_ko.get("n_var", 0))
        dataset_would_ko_status = str(
            manifest_would_ko.get("status", "missing")
        )
        if dataset_would_ko != bool(cfg.bc_would_ko):
            raise ValueError(
                "training config and dataset disagree on would-KO: "
                f"config={bool(cfg.bc_would_ko)}, "
                f"dataset={dataset_would_ko}. Rebuild the dataset with the "
                "current run sheet."
            )
        if dataset_would_ko:
            if dataset_would_ko_nvar != int(cfg.bc_wk_nvar):
                raise ValueError(
                    "training config and dataset disagree on bc_wk_nvar: "
                    f"config={int(cfg.bc_wk_nvar)}, "
                    f"dataset={dataset_would_ko_nvar}"
                )
            if dataset_would_ko_status != "computed":
                raise ValueError(
                    "would-KO dataset is not computation-complete: "
                    f"status={dataset_would_ko_status!r}"
                )
    elif mmapped and bool(cfg.bc_would_ko):
        raise ValueError(
            "bc_would_ko=true requires a dataset_manifest.json with a verified "
            "would_ko contract"
        )
    else:
        # Legacy/unmanifested corpora cannot prove prospective features.
        dataset_would_ko = False
        dataset_would_ko_nvar = int(cfg.bc_wk_nvar)
        dataset_would_ko_status = "unverified-disabled"

    effective_would_ko = bool(dataset_would_ko and not a.zero_wouldko)
    inference_provenance = (
        "verified-dataset-manifest"
        if manifest_would_ko is not None
        else "unmanifested-corpus-would-ko-disabled"
    )
    if a.zero_wouldko:
        inference_provenance = "explicit-zero-would-ko"
    if a.zero_wouldko:
        print(
            "[bc-train-mlx] would-KO features explicitly zeroed for this run; "
            "checkpoint inference will disable would-KO",
            flush=True,
        )
    N = int(d["__labels__"].shape[0])

    # Apply max_rows limit (for smoke testing)
    if a.max_rows and a.max_rows > 0:
        N = min(N, a.max_rows)
        print(
            f"[bc-train-mlx] limited to {a.max_rows} rows (max_rows={a.max_rows})",
            flush=True,
        )

    labels = read_rows(d["__labels__"], 0, N)
    keys = [key for key in d if not key.startswith("__") and key != "episode_meta"]
    obs_np = {k: d[k] for k in keys}
    group_np = d.get("__group__")

    # Dedup
    if a.dedup:
        has_group = "__group__" in d
        if not has_group:
            raise SystemExit("--dedup needs __group__")
        for _s in range(0, N, 1 << 20):
            _e = min(N, _s + (1 << 20))
            _g = read_rows(group_np, _s, _e)
            labels[_s:_e] = _g[np.arange(_e - _s), labels[_s:_e]]
    y = labels.astype(np.int32)
    is_attack = (
        read_rows(d["__is_attack__"], 0, N).astype(bool)
        if "__is_attack__" in d
        else np.zeros(N, dtype=bool)
    )

    # D.4: Episode-level val split (snap to episode boundary if metadata available)
    nval = max(1, int(N * a.val_frac))
    v0 = N - nval
    episode_meta_all: np.ndarray | None = None
    meta_path = os.path.join(a.data, "episode_meta.npy") if mmapped else None
    if meta_path and os.path.exists(meta_path):
        try:
            meta = np.load(meta_path, mmap_mode="r")
            episode_meta_all = meta
            new_ep = meta["new_episode"]
            boundaries = np.where(new_ep)[0]
            valid_boundaries = boundaries[boundaries <= N - nval]
            if len(valid_boundaries) > 0:
                v0 = int(valid_boundaries[-1])
                nval = N - v0
                print(
                    f"[bc-train-mlx] D.4: episode-level split at row {v0} "
                    f"({v0} train, {nval} val)",
                    flush=True,
                )
            else:
                print(
                    f"[bc-train-mlx] D.4: no episode boundary before {N - nval}, "
                    f"using tail split at {v0}",
                    flush=True,
                )
        except Exception as e:
            print(
                f"[bc-train-mlx] D.4: failed to load episode_meta ({e}), "
                f"using tail split at {v0}",
                flush=True,
            )
    idx = np.arange(N)
    vi, ti = idx[v0:], idx[:v0]

    # Val cache
    val_np = {k: read_rows(obs_np[k], v0, N) for k in keys}
    gv_np = read_rows(group_np, v0, N) if group_np is not None else None
    oa_v = val_np["opt_attr"]
    vi_atk = is_attack[vi]
    vi_ko = (oa_v[..., WK_LO] >= 0.5).any(axis=1)

    ct = get_card_table()
    enc = TokenEncoder(ct)
    int_keys = set(enc.int_keys)

    # --- model (MLX!) ---
    net_cfg = {
        "arch": "transformer2",
        "d_model": a.d_model,
        "nhead": a.nhead,
        "nlayers": a.nlayers,
        "ff": a.ff,
        "static": a.static,
        "structured": a.structured,
        "split_heads": a.split_heads,
        "scratch_registers": a.scratch_registers,
        "value_atoms": a.value_atoms,
        "value_vmax": a.value_vmax,
    }
    model = build_token_net_mlx(ct, net_cfg)

    # Resume from checkpoint
    state: dict = {}
    start_epoch = 0
    best = 0.0
    gstep = 0
    if a.resume:
        import pickle

        with open(a.resume, "rb") as f:
            state = pickle.load(f)
        model_params = state["model"]
        if isinstance(model_params, dict):
            model.update(model_params)
        start_epoch = int(state.get("epoch", -1)) + 1
        best = float(state.get("best_val_acc", state.get("val_acc", 0.0)))
        # Global model-update history is provenance, not scheduler position.
        gstep = int(state.get("gstep", 0))
        # Validate arch_config if present (backward compat with old checkpoints)
        saved_cfg = state.get("arch_config")
        if saved_cfg is not None:
            cur_cfg = model.get_config()
            mismatches = []
            for k, v in saved_cfg.items():
                if k != "dtype" and k in cur_cfg and cur_cfg[k] != v:
                    mismatches.append(f"{k}: saved={v} current={cur_cfg[k]}")
            if mismatches:
                raise ValueError(
                    "checkpoint architecture mismatch: " + ", ".join(mismatches)
                )
            else:
                print("[bc-train-mlx] arch_config validated OK")
        else:
            print(
                "[bc-train-mlx] WARNING: no arch_config in checkpoint (old format) — "
                "proceeding without validation"
            )
        print(
            f"[bc-train-mlx] resumed from {a.resume} (epoch {start_epoch}, "
            f"val_acc={best:.4f}, gstep={gstep})"
        )

    # Parameters and forward activations use FP16. Loss, reductions, gradient
    # accumulation, and optimizer moments are promoted explicitly to FP32.
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    parameter_dtypes = {
        str(parameter.dtype)
        for _, parameter in nn.utils.tree_flatten(model.parameters())
    }
    if parameter_dtypes != {"mlx.core.float16"}:
        raise RuntimeError(f"model parameters are not strictly FP16: {parameter_dtypes}")

    # ``--epochs`` is the number of epochs for THIS invocation. The checkpoint
    # epoch is an absolute history counter used only to continue numbering.
    run_epochs = int(a.epochs)
    if run_epochs <= 0:
        raise ValueError(f"epochs must be positive for this run, got {run_epochs}")
    print(
        f"[bc-train-mlx] this run: {run_epochs} epoch(s) "
        f"(global start={start_epoch + 1})",
        flush=True,
    )

    # Optimizer phase is explicit and independent from model-weight resume.
    optimizer = _build_optimizer(a)
    optimizer_contract = _optimizer_contract(a)
    trainable_leaves = nn.utils.tree_flatten(model.trainable_parameters())
    muon_parameter_count = sum(
        parameter.size
        for path, parameter in trainable_leaves
        if _use_muon_parameter(path, parameter)
    )
    adamw_parameter_count = sum(
        parameter.size
        for path, parameter in trainable_leaves
        if not _use_muon_parameter(path, parameter)
    )
    if a.resume and a.optimizer_state == "resume":
        saved_contract = state.get("optimizer_contract")
        if saved_contract != optimizer_contract:
            raise ValueError(
                "cannot resume optimizer state with a different contract: "
                f"saved={saved_contract!r}, current={optimizer_contract!r}"
            )
        saved_opt_state = state.get("optimizer")
        if saved_opt_state is None:
            raise ValueError("checkpoint has no optimizer state to resume")
        optimizer.state = saved_opt_state
        print("[bc-train-mlx] optimizer phase resumed from checkpoint")
    else:
        optimizer.init(model.trainable_parameters())
        print("[bc-train-mlx] optimizer phase initialized from zero")
    optimizer.learning_rate = a.lr
    mx.eval(optimizer.state)
    _validate_optimizer_state_dtypes(optimizer.state)
    print(
        f"[bc-train-mlx] optimizer routing: Muon={muon_parameter_count:,} "
        f"hidden-matrix params; AdamW={adamw_parameter_count:,} "
        "embedding/head/vector params",
        flush=True,
    )

    nparams = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    tag = (
        f"d{a.d_model}L{a.nlayers}h{a.nhead}"
        f"{' +static' if a.static else ''}"
        f"{' +split' if a.split_heads else ''}"
        f"{' +struct' if a.structured else ''}"
        f"{' +compile' if a.compile else ''}"
        f"{' +prefetch' if a.prefetch else ''}"
        f"{' +accum' if a.accum_steps > 1 else ''}"
    )
    print(
        f"[bc-train-mlx] {tag} params={nparams:,} N={N} train={len(ti)} val={len(vi)} "
        f"batch={a.batch} accum_steps={a.accum_steps}",
        flush=True,
    )

    # --- batch generator (C.1: FP16-native numeric features) ---
    def batches(
        arrs: dict, grp: np.ndarray | None, base: int, order: np.ndarray, bs: int
    ):
        for i in range(0, len(order), bs):
            b = order[i : i + bs]
            ob = {
                k: mx.array(
                    np.asarray(arrs[k][b]).astype(
                        np.int32 if k in int_keys else np.float16
                    )
                )
                for k in keys
            }
            if a.dedup:
                gb = mx.array(np.asarray(grp[b]), dtype=mx.int32)
                canon = (gb == mx.arange(gb.shape[1])[None, :]).astype(mx.float16)
                ob["action_mask"] = ob["action_mask"] * canon
            if a.zero_wouldko:
                attr = np.asarray(ob["opt_attr"]).copy()
                attr[..., WK_LO:WK_HI] = 0.0
                ob["opt_attr"] = mx.array(attr, dtype=mx.float16)
            yield ob, mx.array(y[base + b].astype(np.int32))

    # --- loss + grad function ---
    grad_fn = mx.value_and_grad(
        lambda model, ob, yb: _cross_entropy_sum(
            model.logits_value(ob)[0], yb
        ),
        argnums=0,
    )

    # F.3: TBPTT loss + grad function (accepts memory, returns memory_out)
    def _tbptt_loss_fn(
        model, lane_observations, lane_labels, decision_lengths, lane_memory_in
    ):
        return _batched_sequential_tbptt_loss(
            model,
            lane_observations,
            lane_labels,
            decision_lengths,
            lane_memory_in,
        )

    _tbptt_grad_fn = mx.value_and_grad(_tbptt_loss_fn, argnums=0)

    def tbptt_loss_and_grad(
        model, lane_observations, lane_labels, decision_lengths, lane_memory_in
    ):
        """Forward with memory, cross-entropy loss, backward through model params only."""
        (loss, memory_out), grads = _tbptt_grad_fn(
            model,
            lane_observations,
            lane_labels,
            decision_lengths,
            lane_memory_in,
        )
        return loss, grads, memory_out

    # --- exact work plan and optimizer-step schedule (C.4 / F.3) ---
    slab_bounds = [(s, min(s + a.slab_rows, v0)) for s in range(0, v0, a.slab_rows)]
    tbptt_meta_path = os.path.join(a.data, "episode_meta.npy") if mmapped else None
    _use_tbptt = bool(
        a.tbptt_chunk > 0 and tbptt_meta_path and os.path.exists(tbptt_meta_path)
    )
    _tbptt_groups: list[np.ndarray] = []
    _tbptt_decision_groups: list[list[np.ndarray]] = []
    _tbptt_plan: list[list[_TBPTTChunk]] = []
    _val_tbptt_plan: list[list[_TBPTTChunk]] = []
    if _use_tbptt:
        tbptt_meta = np.load(tbptt_meta_path, mmap_mode="r")
        _tbptt_groups = _build_tbptt_groups(tbptt_meta, v0)
        _tbptt_decision_groups = _build_tbptt_decision_groups(tbptt_meta, v0)
        _tbptt_plan = _build_tbptt_plan(
            _tbptt_decision_groups,
            chunk_size=a.tbptt_chunk,
            row_budget=a.batch,
        )
        if episode_meta_all is None:
            raise ValueError("TBPTT validation requires episode_meta")
        validation_meta = episode_meta_all[v0:N]
        validation_decision_groups = _build_tbptt_decision_groups(
            validation_meta, nval
        )
        _val_tbptt_plan = _build_tbptt_plan(
            validation_decision_groups,
            chunk_size=a.tbptt_chunk,
            row_budget=a.val_batch_size,
        )
        microbatches_per_epoch = len(_tbptt_plan)
        progress_mode = (
            f"TBPTT temporal batches (decisions/chunk<={a.tbptt_chunk}, "
            f"rows/batch<={a.batch:,}, groups={len(_tbptt_groups):,})"
        )
    else:
        microbatches_per_epoch = _standard_microbatch_count(slab_bounds, a.batch)
        progress_mode = f"shuffled batches (batch<={a.batch:,})"

    if microbatches_per_epoch <= 0:
        raise ValueError("training split produced no microbatches")
    compile_active = bool(a.compile and not _use_tbptt)
    if a.compile and _use_tbptt:
        print(
            "[bc-train-mlx] compile disabled for variable-length sequential TBPTT",
            flush=True,
        )

    optimizer_steps_per_epoch = _ceil_div(microbatches_per_epoch, a.accum_steps)
    run_microbatches = run_epochs * microbatches_per_epoch
    run_optimizer_steps = run_epochs * optimizer_steps_per_epoch
    run_start_gstep = gstep
    if not a.resume and (
        a.optimizer_state == "resume" or a.scheduler_state == "resume"
    ):
        raise ValueError("phase state cannot be resumed without --resume")

    optimizer_phase_step = (
        int(state.get("optimizer_phase_step", 0))
        if a.resume and a.optimizer_state == "resume"
        else 0
    )
    if a.resume and a.scheduler_state == "resume":
        if "scheduler_phase_step" not in state or "scheduler_total_steps" not in state:
            raise ValueError("checkpoint has no scheduler phase to resume")
        scheduler_phase_step = int(state["scheduler_phase_step"])
        scheduler_total_steps = int(state["scheduler_total_steps"])
        if (
            a.scheduler_total_steps > 0
            and a.scheduler_total_steps != scheduler_total_steps
        ):
            raise ValueError(
                "scheduler_total_steps cannot change while resuming a phase: "
                f"saved={scheduler_total_steps}, configured={a.scheduler_total_steps}"
            )
    else:
        scheduler_phase_step = 0
        scheduler_total_steps = (
            int(a.scheduler_total_steps)
            if a.scheduler_total_steps > 0
            else run_optimizer_steps
        )
    if scheduler_total_steps <= 0:
        raise ValueError("scheduler_total_steps must be positive")
    if (
        a.resume
        and a.scheduler_state == "resume"
        and scheduler_phase_step + run_optimizer_steps > scheduler_total_steps
    ):
        raise ValueError(
            "resumed scheduler phase does not have enough remaining steps: "
            f"position={scheduler_phase_step}, requested={run_optimizer_steps}, "
            f"total={scheduler_total_steps}. Reset the scheduler or choose the "
            "full phase horizon before the original run."
        )
    warmup_steps = min(a.warmup_steps, max(1, scheduler_total_steps // 5))
    scheduler_contract = _scheduler_contract(
        a, scheduler_total_steps, warmup_steps
    )
    if a.resume and a.scheduler_state == "resume":
        saved_scheduler_contract = state.get("scheduler_contract")
        if saved_scheduler_contract != scheduler_contract:
            raise ValueError(
                "cannot resume scheduler phase with a different contract: "
                f"saved={saved_scheduler_contract!r}, "
                f"current={scheduler_contract!r}"
            )
    run_end_epoch = start_epoch + run_epochs
    print(
        f"[bc-train-mlx] global epoch range: {start_epoch + 1}-{run_end_epoch} "
        f"(local epochs={run_epochs})",
        flush=True,
    )
    print(
        f"[bc-train-mlx] progress plan: {progress_mode}; "
        f"microbatches/epoch={microbatches_per_epoch:,}; "
        f"optimizer_steps/epoch={optimizer_steps_per_epoch:,}; "
        f"run_microbatches={run_microbatches:,}; "
        f"run_optimizer_steps={run_optimizer_steps:,}; "
        f"global_step={run_start_gstep:,}->{run_start_gstep + run_optimizer_steps:,}; "
        f"scheduler_phase={scheduler_phase_step:,}/"
        f"{scheduler_total_steps:,} ({a.scheduler_state})",
        flush=True,
    )

    def _checkpoint_payload(epoch: int, val_acc: float) -> dict:
        static_features = getattr(model, "_card_feat_np", None)
        static_contract = None
        if static_features is not None:
            static_array = np.asarray(static_features, dtype=np.float16)
            static_contract = {
                "dtype": str(static_array.dtype),
                "shape": list(static_array.shape),
                "sha256": _sha256_bytes(static_array.tobytes(order="C")),
                "card_csv_sha256": _sha256_file(ct.csv_path),
            }
        return {
            "model": model.parameters(),
            "optimizer": optimizer.state,
            "optimizer_contract": optimizer_contract,
            "optimizer_phase_step": optimizer_phase_step,
            "arch_config": model.get_config(),
            "static_card_features": static_features,
            "static_feature_contract": static_contract,
            # The JSON config is a transient run sheet. Checkpoints retain the
            # resolved training provenance and the exact inference contract so
            # exported agents never consult a possibly changed JSON file.
            "run_config": {
                **cfg.to_dict(),
                "data_path": a.data,
                "output_path": a.out,
                "zero_wouldko": bool(a.zero_wouldko),
            },
            "inference_config": {
                "version": 1,
                "seed": int(a.seed),
                "bc_would_ko": effective_would_ko,
                "bc_wk_nvar": int(dataset_would_ko_nvar),
                "provenance": inference_provenance,
            },
            "dataset_manifest": dataset_manifest,
            "dataset_build_fingerprint": dataset_manifest.get(
                "build_fingerprint"
            ),
            "phase_id": a.phase_id,
            "epoch": epoch,
            "gstep": gstep,
            "val_acc": val_acc,
            "best_val_acc": best,
            "seed": a.seed,
            "dataset_path": a.data,
            "accum_steps": a.accum_steps,
            "microbatches_per_epoch": microbatches_per_epoch,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "scheduler_phase_step": scheduler_phase_step,
            "scheduler_total_steps": scheduler_total_steps,
            "scheduler_contract": scheduler_contract,
            "scheduler_state": a.scheduler_state,
        }

    def _save_checkpoint(path: str, epoch: int, val_acc: float) -> None:
        import pickle

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        mx.eval(model.parameters(), optimizer.state)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(_checkpoint_payload(epoch, val_acc), f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def _periodic_checkpoint_path(epoch: int) -> str:
        out_path = Path(a.out)
        return str(
            out_path.with_name(
                f"{out_path.stem}_epoch_{epoch + 1:04d}{out_path.suffix}"
            )
        )

    def _latest_checkpoint_path() -> str:
        out_path = Path(a.out)
        return str(
            out_path.with_name(f"{out_path.stem}_latest{out_path.suffix}")
        )

    # --- graph-safe gradient clipping (C.3) ---
    def clip_grads(grads, max_norm):
        """Clip gradients in MLX graph (no float() calls)."""
        if max_norm <= 0:
            return grads
        flat = [g.reshape(-1) for _, g in nn.utils.tree_flatten(grads) if g is not None]
        if not flat:
            return grads
        gn = mx.sqrt(sum(mx.sum(g**2) for g in flat))
        scale = mx.where(gn > max_norm, max_norm / mx.maximum(gn, 1e-6), 1.0)
        grads = nn.utils.tree_map(lambda g: (g * scale) if g is not None else g, grads)
        mx.eval(grads)
        return grads

    # --- train step with gradient accumulation (C.2) ---
    def train_step_accum(ob: dict, yb: mx.array):
        """Forward + backward for one microbatch using an FP32 loss sum."""
        loss, grads = grad_fn(model, ob, yb)
        mx.eval(loss)
        loss_val = float(loss)
        return loss_val, _to_fp32_grads(grads)

    def optimizer_step(grads, n_examples):
        """Normalize accumulated grads, clip, update optimizer, advance gstep."""
        nonlocal gstep, optimizer_phase_step, scheduler_phase_step
        if n_examples <= 0:
            raise ValueError("optimizer steps require at least one example")
        gstep += 1
        optimizer_phase_step += 1
        scheduler_phase_step += 1
        # Normalize by total examples (FP32 reduction)
        grads = nn.utils.tree_map(
            lambda g: (g / n_examples) if g is not None else g, grads
        )
        # Clip (C.3: graph-safe, no float())
        grads = clip_grads(grads, a.max_grad_norm)
        # LR schedule on optimizer step (C.4)
        if a.lr_schedule != "none":
            optimizer.learning_rate = lr_at(
                scheduler_phase_step,
                scheduler_total_steps,
                a.lr,
                a.lr_schedule,
                warmup_steps,
                a.lr_min_ratio,
            )
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        mx.eval(optimizer.state)

    # Compile stable shuffled-batch forward/backward; clipping stays outside.
    if compile_active:
        from functools import partial

        _state = [model.state, optimizer.state]

        @partial(mx.compile, inputs=_state, outputs=_state)
        def compiled_step(ob, yb):
            """Compiled forward + backward; clipping runs after accumulation."""
            loss, grads = grad_fn(model, ob, yb)
            return loss, grads

        print(f"[bc-train-mlx] compiled train_step with state capture", flush=True)

    # ---- prefetch helper (stream overlap: CPU loads next slab while GPU trains) ----
    def _slab_generator(slab_indices, ep_seed):
        """Yield (slab_i, si, sd, sg, perm) for each slab."""
        for slab_i, si in enumerate(slab_indices):
            s0, e0 = slab_bounds[si]
            sd = {k: read_rows(obs_np[k], s0, e0) for k in keys}
            sg = (
                read_rows(group_np, s0, e0)
                if (a.dedup and group_np is not None)
                else None
            )
            perm = np.random.default_rng([a.seed, ep_seed, s0]).permutation(e0 - s0)
            yield slab_i, si, s0, e0, sd, sg, perm

    def _prefetch_slabs(slab_indices, ep_seed, q: queue.Queue):
        """Background thread: load slabs ahead of the GPU."""
        for item in _slab_generator(slab_indices, ep_seed):
            q.put(item)
        q.put(None)

    # ---- F.3: TBPTT batch generator ----
    def _load_temporal_batch(
        temporal_batch: list[_TBPTTChunk],
        arrays: dict,
        *,
        label_base: int,
    ):
        """Materialize one pre-planned temporal batch with lane-local row order."""
        lane_observations: list[dict[str, mx.array]] = []
        lane_labels: list[mx.array] = []
        lane_decision_lengths: list[list[int]] = []
        lane_rows: list[np.ndarray] = []
        for chunk in temporal_batch:
            chunk_arr = np.concatenate(chunk.decisions)
            lane_rows.append(chunk_arr)
            lane_observations.append(
                {
                    key: mx.array(
                        np.asarray(arrays[key][chunk_arr]).astype(
                            np.int32 if key in int_keys else np.float16
                        )
                    )
                    for key in keys
                }
            )
            if a.zero_wouldko:
                attr = np.asarray(lane_observations[-1]["opt_attr"]).copy()
                attr[..., WK_LO:WK_HI] = 0.0
                lane_observations[-1]["opt_attr"] = mx.array(
                    attr, dtype=mx.float16
                )
            lane_labels.append(
                mx.array(y[label_base + chunk_arr].astype(np.int32))
            )
            lane_decision_lengths.append(
                [len(decision_rows) for decision_rows in chunk.decisions]
            )
        return (
            lane_observations,
            lane_labels,
            lane_decision_lengths,
            lane_rows,
        )

    def _tbptt_batches():
        """Yield the exact packed temporal batches used by the work plan."""
        for temporal_batch in _tbptt_plan:
            lane_observations, lane_labels, lane_decision_lengths, _ = (
                _load_temporal_batch(
                    temporal_batch,
                    obs_np,
                    label_base=0,
                )
            )
            yield (
                temporal_batch,
                lane_observations,
                lane_labels,
                lane_decision_lengths,
            )

    # ---- training loop ----
    _running_loss: float = 0.0
    _running_n: int = 0
    _compile_pending: bool = compile_active
    _micro_count: int = 0
    _accum_grads = None
    _accum_examples: int = 0
    _accum_loss_sum: float = 0.0
    _tbptt_memories: dict[int, mx.array] = {}

    validation_batches = (
        len(_val_tbptt_plan)
        if _use_tbptt
        else _ceil_div(nval, a.val_batch_size)
    )
    train_t0 = time.time()

    for ep in range(start_epoch, run_end_epoch):
        mx.reset_peak_memory()
        model.train()
        ep_t0: float = time.time()
        ep_step: int = 0  # optimizer steps in this epoch
        ep_micro: int = 0  # microbatches processed in this epoch
        _running_loss = 0.0
        _running_n = 0
        _micro_count = 0
        _accum_grads = None
        _accum_examples = 0
        _accum_loss_sum = 0.0
        _tbptt_memories = {}  # reset every episode-side lane at epoch start
        local_epoch = ep - start_epoch + 1
        print(
            f"[bc-train-mlx] === epoch {ep + 1} (run {local_epoch}/{run_epochs}) ===",
            flush=True,
        )

        # One phase-aware task per epoch. The bar is reset for validation and
        # checkpointing, so 100% always means the displayed phase is complete.
        _progress_bar = Progress(
            SpinnerColumn(),
            TextColumn(
                "[bold blue]Epoch {task.fields[epoch]} "
                "(run {task.fields[local_epoch]}/{task.fields[total_epochs]})"
            ),
            TextColumn("[bold magenta]{task.fields[phase]}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.fields[unit]}"),
            TextColumn("Opt: {task.fields[opt]}"),
            TextColumn("Loss: {task.fields[loss]}"),
            TextColumn("LR: {task.fields[lr]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        _progress_bar.start()
        _progress_task = _progress_bar.add_task(
            "epoch",
            epoch=ep + 1,
            local_epoch=local_epoch,
            total_epochs=run_epochs,
            phase="train",
            total=microbatches_per_epoch,
            unit=f"micro 0/{microbatches_per_epoch:,}",
            opt=f"0/{optimizer_steps_per_epoch:,}",
            loss="--",
            lr=f"{float(optimizer.learning_rate):.2e}",
        )

        # Flatten all batches from all slabs into a single generator
        # so we can accumulate across slab boundaries
        def _all_batches():
            """Yield (ob, yb) from all slabs in this epoch."""
            if mmapped:
                perm_slabs = np.random.default_rng([a.seed, ep]).permutation(
                    len(slab_bounds)
                )
                if a.prefetch:
                    q: queue.Queue = queue.Queue(maxsize=1)
                    t = threading.Thread(
                        target=_prefetch_slabs, args=(perm_slabs, ep, q), daemon=True
                    )
                    t.start()
                    slab_iter = iter(lambda: q.get(), None)
                else:
                    slab_iter = _slab_generator(perm_slabs, ep)
                for slab_i, si, s0, e0, sd, sg, perm in slab_iter:
                    slab_t: float = time.time()
                    load_t = 0.0
                    if not a.prefetch:
                        print(
                            f"[bc-train-mlx]   slab {slab_i + 1}/{len(slab_bounds)} "
                            f"(rows {s0:,}-{e0:,}) loading...",
                            end="",
                            flush=True,
                        )
                        load_t = time.time() - slab_t
                        print(f" {load_t:.1f}s", flush=True)
                    else:
                        print(
                            f"[bc-train-mlx]   slab {slab_i + 1}/{len(slab_bounds)} "
                            f"(rows {s0:,}-{e0:,}) prefetched",
                            flush=True,
                        )
                    for ob, yb in batches(sd, sg, s0, perm, a.batch):
                        yield ob, yb
                    del sd, sg
                    slab_loss = _running_loss / max(_running_n, 1)
                    print(
                        f"[bc-train-mlx]   slab {slab_i + 1}/{len(slab_bounds)} done "
                        f"train_loss={slab_loss:.4f} ({time.time() - ep_t0:.0f}s total, "
                        f"{load_t:.1f}s load)",
                        flush=True,
                    )
            else:
                g = np.random.default_rng([a.seed, ep])
                order = g.permutation(len(ti))
                for ob, yb in batches(obs_np, group_np, 0, order, a.batch):
                    yield ob, yb

        if _use_tbptt:
            print(
                f"[bc-train-mlx] F.3: TBPTT enabled "
                f"(decisions/chunk={a.tbptt_chunk}, row_budget={a.batch:,}, "
                f"microbatches={microbatches_per_epoch:,}, "
                f"optimizer_steps={optimizer_steps_per_epoch:,})",
                flush=True,
            )

        _batch_iter = _tbptt_batches() if _use_tbptt else _all_batches()

        for _batch_tuple in _batch_iter:
            optimizer_updated = False
            is_final_microbatch = ep_micro + 1 == microbatches_per_epoch
            if _use_tbptt:
                (
                    temporal_batch,
                    lane_observations,
                    lane_labels,
                    lane_decision_lengths,
                ) = _batch_tuple
                lane_memory_in = [
                    (
                        None
                        if chunk.is_new_group
                        else _tbptt_memories[chunk.group_index]
                    )
                    for chunk in temporal_batch
                ]
                micro_n = sum(len(labels) for labels in lane_labels)
            else:
                ob, yb = _batch_tuple  # standard path: 2-tuple
                micro_n = len(yb)

            # Forward + backward this microbatch
            if compile_active and _compile_pending:
                print(
                    "[bc-train-mlx]   compiling (first call, may take several minutes)...",
                    end="",
                    flush=True,
                )
                _compile_t = time.time()

            if a.accum_steps > 1:
                # F.3: TBPTT path uses memory-aware forward
                if _use_tbptt:
                    loss, grads, mem_out = tbptt_loss_and_grad(
                        model,
                        lane_observations,
                        lane_labels,
                        lane_decision_lengths,
                        lane_memory_in,
                    )
                    grads = _to_fp32_grads(grads)
                    mx.eval(loss, grads, mem_out)
                    loss_val = float(loss)
                    detached_memories = mx.stop_gradient(mem_out)
                    mx.eval(detached_memories)
                    for lane_index, chunk in enumerate(temporal_batch):
                        _tbptt_memories[chunk.group_index] = detached_memories[
                            lane_index : lane_index + 1
                        ]
                else:
                    loss_val, grads = train_step_accum(ob, yb)
                if not np.isfinite(loss_val):
                    raise FloatingPointError(
                        f"non-finite training loss at epoch={ep + 1}, "
                        f"microbatch={ep_micro + 1}"
                    )
                if _accum_grads is None:
                    _accum_grads = grads
                    _accum_examples = micro_n
                    _accum_loss_sum = loss_val
                else:
                    _accum_grads = nn.utils.tree_map(
                        lambda a, b: (
                            (a + b)
                            if (a is not None and b is not None)
                            else (a if a is not None else b)
                        ),
                        _accum_grads,
                        grads,
                    )
                    _accum_examples += micro_n
                    _accum_loss_sum += loss_val
                _micro_count += 1

                # Accumulate in FP32 for numerical stability
                if _micro_count % a.accum_steps == 0 or is_final_microbatch:
                    optimizer_step(_accum_grads, _accum_examples)
                    ep_step += 1
                    optimizer_updated = True
                    _running_loss += _accum_loss_sum
                    _running_n += _accum_examples
                    _accum_grads = None
                    _accum_examples = 0
                    _accum_loss_sum = 0.0
            else:
                # No accumulation: single microbatch = full step
                if _use_tbptt:
                    loss, grads, mem_out = tbptt_loss_and_grad(
                        model,
                        lane_observations,
                        lane_labels,
                        lane_decision_lengths,
                        lane_memory_in,
                    )
                    grads = _to_fp32_grads(grads)
                    mx.eval(loss, grads, mem_out)
                    loss_val = float(loss)
                    detached_memories = mx.stop_gradient(mem_out)
                    mx.eval(detached_memories)
                    for lane_index, chunk in enumerate(temporal_batch):
                        _tbptt_memories[chunk.group_index] = detached_memories[
                            lane_index : lane_index + 1
                        ]
                elif compile_active:
                    loss, grads = compiled_step(ob, yb)
                    grads = _to_fp32_grads(grads)
                    mx.eval(loss, grads)
                    loss_val = float(loss)
                else:
                    loss, grads = grad_fn(model, ob, yb)
                    grads = _to_fp32_grads(grads)
                    mx.eval(loss, grads)
                    loss_val = float(loss)
                if not np.isfinite(loss_val):
                    raise FloatingPointError(
                        f"non-finite training loss at epoch={ep + 1}, "
                        f"microbatch={ep_micro + 1}"
                    )
                optimizer_step(grads, micro_n)
                ep_step += 1
                optimizer_updated = True
                _running_loss += loss_val
                _running_n += micro_n

            ep_micro += 1
            if _compile_pending:
                print(f" done ({time.time() - _compile_t:.0f}s)", flush=True)
                _compile_pending = False

            # Include the current, not-yet-stepped accumulation window so the
            # displayed loss never drops to a misleading 0.0000 between
            # optimizer updates.
            display_loss_sum = _running_loss + _accum_loss_sum
            display_examples = _running_n + _accum_examples
            avg = display_loss_sum / max(display_examples, 1)
            elapsed_s = time.time() - ep_t0
            microbatches_left = max(0, microbatches_per_epoch - ep_micro)
            train_eta_s = (elapsed_s / max(ep_micro, 1)) * microbatches_left
            elapsed_str = _format_compact_duration(elapsed_s)
            train_eta_str = _format_compact_duration(train_eta_s)
            _progress_bar.update(
                _progress_task,
                completed=ep_micro,
                unit=f"micro {ep_micro:,}/{microbatches_per_epoch:,}",
                opt=f"{ep_step:,}/{optimizer_steps_per_epoch:,}",
                loss=f"{avg:.4f}",
                lr=f"{float(optimizer.learning_rate):.2e}",
            )

            if (
                a.log_interval > 0
                and optimizer_updated
                and ep_step % a.log_interval == 0
            ):
                print(
                    f"[bc-train-mlx]   opt_step "
                    f"{ep_step:,}/{optimizer_steps_per_epoch:,} "
                    f"(run {gstep - run_start_gstep:,}/{run_optimizer_steps:,}) "
                    f"micro={ep_micro:,}/{microbatches_per_epoch:,} "
                    f"loss={avg:.4f} lr={float(optimizer.learning_rate):.2e} "
                    f"elapsed={elapsed_str} train_ETA={train_eta_str}",
                    flush=True,
                )

        if ep_micro != microbatches_per_epoch:
            _progress_bar.stop()
            raise RuntimeError(
                f"progress plan mismatch: expected {microbatches_per_epoch:,} "
                f"microbatches, processed {ep_micro:,}"
            )
        if ep_step != optimizer_steps_per_epoch:
            _progress_bar.stop()
            raise RuntimeError(
                f"optimizer-step plan mismatch: expected "
                f"{optimizer_steps_per_epoch:,}, completed {ep_step:,}"
            )
        if _accum_grads is not None or _accum_examples != 0:
            _progress_bar.stop()
            raise RuntimeError("gradient accumulation was not flushed at epoch end")

        print(
            f"[bc-train-mlx] training complete: epoch {ep + 1}, "
            f"microbatches={ep_micro:,}, optimizer_steps={ep_step:,}, "
            f"gstep={gstep:,}; peak_memory={mx.get_peak_memory() / (1024 ** 3):.2f} GiB; "
            "starting validation",
            flush=True,
        )
        _progress_bar.reset(
            _progress_task,
            total=validation_batches,
            completed=0,
            epoch=ep + 1,
            local_epoch=local_epoch,
            total_epochs=run_epochs,
            phase="validate",
            unit=f"batch 0/{validation_batches:,}",
            opt=f"{ep_step:,}/{optimizer_steps_per_epoch:,}",
            loss="--",
            lr="--",
        )

        # ---- validation ----
        model.eval()
        preds: list[np.ndarray] = []
        am_all: list[np.ndarray] = []
        vloss: float = 0.0
        tot: int = 0
        if _use_tbptt:
            validation_memories: dict[int, mx.array] = {}
            validation_rows: list[np.ndarray] = []
            validation_logits: list[np.ndarray] = []
            for val_batch, temporal_batch in enumerate(
                _val_tbptt_plan, start=1
            ):
                (
                    lane_observations,
                    lane_labels,
                    lane_decision_lengths,
                    lane_rows,
                ) = _load_temporal_batch(
                    temporal_batch,
                    val_np,
                    label_base=v0,
                )
                lane_memory_in = [
                    (
                        None
                        if chunk.is_new_group
                        else validation_memories[chunk.group_index]
                    )
                    for chunk in temporal_batch
                ]
                _, memory_out, lane_logits = _batched_sequential_tbptt_loss(
                    model,
                    lane_observations,
                    lane_labels,
                    lane_decision_lengths,
                    lane_memory_in,
                    return_logits=True,
                )
                mx.eval(memory_out, lane_logits)
                detached_memories = mx.stop_gradient(memory_out)
                mx.eval(detached_memories)
                for lane_index, chunk in enumerate(temporal_batch):
                    validation_memories[chunk.group_index] = detached_memories[
                        lane_index : lane_index + 1
                    ]
                    validation_rows.append(lane_rows[lane_index])
                    validation_logits.append(
                        np.asarray(lane_logits[lane_index], dtype=np.float32)
                    )
                _progress_bar.update(
                    _progress_task,
                    completed=val_batch,
                    unit=f"temporal {val_batch:,}/{validation_batches:,}",
                )

            row_order = np.concatenate(validation_rows)
            row_permutation = np.argsort(row_order)
            if not np.array_equal(
                row_order[row_permutation], np.arange(nval)
            ):
                raise RuntimeError(
                    "temporal validation did not cover every validation row exactly once"
                )
            lg_np = np.concatenate(validation_logits, axis=0)[row_permutation]
            yb_np = y[vi]
        else:
            logits_parts: list[np.ndarray] = []
            for val_batch, (ob, _) in enumerate(
                batches(
                    val_np,
                    gv_np,
                    v0,
                    np.arange(nval),
                    a.val_batch_size,
                ),
                start=1,
            ):
                lg, _, _ = model.logits_value(ob)
                logits_parts.append(np.asarray(lg, dtype=np.float32))
                _progress_bar.update(
                    _progress_task,
                    completed=val_batch,
                    unit=f"batch {val_batch:,}/{validation_batches:,}",
                )
            lg_np = np.concatenate(logits_parts, axis=0)
            yb_np = y[vi]

        # Proper cross-entropy: -(logit[label] - logsumexp(logits))
        logsumexp = np.logaddexp.reduce(lg_np, axis=1)
        ce = -(lg_np[np.arange(len(yb_np)), yb_np] - logsumexp)
        vloss = float(ce.sum())
        tot = len(yb_np)
        top3 = np.argsort(-lg_np, axis=1)[:, :3]
        correct = np.argmax(lg_np, axis=1) == yb_np
        in_top3 = np.array([yb_np[i] in top3[i] for i in range(len(yb_np))])
        preds.append(
            np.stack([correct.astype(float), in_top3.astype(float)], axis=1)
        )
        am_all.append(np.argmax(lg_np, axis=1))

        _progress_bar.reset(
            _progress_task,
            total=1,
            completed=0,
            epoch=ep + 1,
            local_epoch=local_epoch,
            total_epochs=run_epochs,
            phase="metrics",
            unit="aggregating",
            opt=f"{ep_step:,}/{optimizer_steps_per_epoch:,}",
            loss="--",
            lr="--",
        )
        pr = np.concatenate(preds)
        c1, c3 = pr[:, 0], pr[:, 1]
        acc: float = float(c1.mean())
        t3: float = float(c3.mean())
        atk: float = float(c1[vi_atk].mean()) if vi_atk.sum() > 0 else 0.0
        ko: float = float(c1[vi_ko].mean()) if vi_ko.sum() > 0 else 0.0
        eq: float = acc
        if gv_np is not None:
            am_cat = np.concatenate(am_all)
            yv_group = gv_np[np.arange(len(vi)), y[vi]]
            eq = float((gv_np[np.arange(len(vi)), am_cat] == yv_group).mean())

        _progress_bar.reset(
            _progress_task,
            total=1,
            completed=0,
            epoch=ep + 1,
            local_epoch=local_epoch,
            total_epochs=run_epochs,
            phase="checkpoint",
            unit="saving",
            opt=f"{ep_step:,}/{optimizer_steps_per_epoch:,}",
            loss=f"{_running_loss / max(_running_n, 1):.4f}",
            lr=f"{float(optimizer.learning_rate):.2e}",
        )
        # Complete checkpoint: save model, optimizer, arch_config, scheduler, seed (C.5)
        is_best = acc > best or not os.path.exists(a.out)
        best = max(best, acc)
        if is_best:
            _save_checkpoint(a.out, ep, acc)

        # The rolling latest checkpoint makes an interrupted long run
        # resumable without creating one file per epoch.
        latest_path = _latest_checkpoint_path()
        _save_checkpoint(latest_path, ep, acc)

        # Numbered snapshots are retained at the configured cadence. The final
        # epoch is always retained as well.
        if local_epoch % a.checkpoint_every_epochs == 0 or ep == run_end_epoch - 1:
            periodic_path = _periodic_checkpoint_path(ep)
            _save_checkpoint(periodic_path, ep, acc)
            print(
                f"[bc-train-mlx] checkpoint saved: {periodic_path} "
                f"(epoch {ep + 1}, val_acc={acc:.4f})",
                flush=True,
            )
        print(
            f"[bc-train-mlx] rolling checkpoint: {latest_path}",
            flush=True,
        )

        ep_time: float = time.time() - ep_t0
        elapsed: float = time.time() - train_t0
        completed: int = ep - start_epoch + 1
        remaining: int = run_end_epoch - ep - 1
        eta_s: float = (elapsed / max(completed, 1)) * remaining
        eta_str = _format_compact_duration(eta_s)
        _progress_bar.update(
            _progress_task,
            total=1,
            completed=1,
            phase="done",
            unit="epoch complete",
        )
        print(
            f"[bc-train-mlx] epoch {ep + 1} complete: "
            f"val_acc={acc:.4f} equiv={eq:.4f} top3={t3:.4f} "
            f"atk={atk:.4f} ko={ko:.4f} loss={vloss / max(tot, 1):.4f} "
            f"t={_format_compact_duration(ep_time)} run_ETA={eta_str} "
            f"gstep={gstep:,}",
            flush=True,
        )

        _progress_bar.stop()

    # Export is explicit so smoke runs and alternative training phases cannot
    # overwrite the submission model as an unrelated side effect.
    if a.export_final:
        if not a.out or not os.path.exists(a.out):
            raise RuntimeError("cannot export final model: best checkpoint is missing")
        os.makedirs(os.path.dirname(a.export_final) or ".", exist_ok=True)
        shutil.copy2(a.out, a.export_final)
        print(
            f"[bc-train-mlx] best checkpoint exported to {a.export_final}",
            flush=True,
        )
    print(
        f"[bc-train-mlx] RESULT: best_val_acc={best:.4f} params={nparams:,} gstep={gstep}",
        flush=True,
    )


if __name__ == "__main__":
    main()
