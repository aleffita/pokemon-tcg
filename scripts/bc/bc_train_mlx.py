"""Behavioral-cloning trainer — MLX version (Apple Silicon native).

Same architecture as bc_train.py but uses MLX instead of PyTorch.
Faster on M1/M2 via native Metal GPU (no MPS NaN bug).

FP16-native: numeric features stay float16 end-to-end.
Gradient accumulation: --accum-steps K accumulates K microbatches before update.

Data comes from the Parquet catalog (one Parquet file per day, registered in
model/results.db by scripts/bc/build_bc_from_zips.py), streamed with
pyarrow.dataset -- there is no more multi-.npy mmap directory format.

Usage:
  uv run tcg-train --days 2026-07-30,2026-08-01 --config configs/train_config.json
  uv run tcg-train --last-n-days 5 --config configs/train_config.json
  uv run tcg-train --all-days --config configs/train_config.json
  uv run tcg-train --config configs/train_config.json  # uses training_days from config
"""

import argparse
import hashlib
import json
import os
import shutil
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pyarrow.dataset as pads
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
from rl.results_db import ResultsDB
from rl.train_config import load_config

WK_LO, WK_HI = OPT_WK, OPT_WK + 3

# Auxiliary-head target columns written by the Fase 3a dataset builder.
# aux_valid gates which rows actually have a computed aux target (some rows,
# e.g. near-terminal or non-attack decisions, may not carry every signal).
_AUX_COLUMNS = ["aux_ko", "aux_prize_delta", "aux_terminal", "aux_return", "aux_valid"]

# One pyarrow I/O batch is the physical read unit (no more slab_rows). This is
# independent of the trainer's --batch (microbatch) size; the streaming
# re-chunker (`_stream_train_microbatches`) re-slices I/O batches into exact
# `cfg.batch_size` microbatches regardless of how pyarrow partitions row
# groups, so the microbatch/optimizer-step accounting stays exact.
_IO_BATCH_ROWS = 32768

# Metadata/aux columns that are NOT part of the encoder's token schema (i.e.
# not in TokenEncoder.shapes) but are still real Parquet columns written by
# build_bc_from_zips.py. Each maps to the numpy dtype it should be read back
# as. "y" (label) and "opt_group" (dedup canonicalization, __group__ in the
# old .npy format) are handled explicitly wherever they're read, since their
# shape depends on the encoder (opt_group reuses action_mask's shape).
_META_COLUMN_DTYPES: dict[str, type] = {
    "y": np.int32,
    "is_attack": np.bool_,
    "episode_id": np.int64,
    "side": np.int32,
    "step_id": np.int32,
    "decision_id": np.int32,
    "substep": np.int32,
    "new_episode": np.bool_,
    "terminal": np.bool_,
    "reward": np.float32,
    "aux_ko": np.int32,
    "aux_prize_delta": np.float32,
    "aux_terminal": np.bool_,
    "aux_return": np.float32,
    "aux_valid": np.int32,
}


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
    """MultiOptimizer merge that supports parameter trees containing lists.

    Also fixes the base MultiOptimizer's `_split_dictionary` which silently
    drops parameters that don't match any filter (rather than routing them to
    the fallback optimizer, which is the last one by convention). Here every
    unfiltered leaf goes to the last optimizer.
    """

    def _split_dictionary(self, gradients: dict):
        if len(self.optimizers) == 1:
            return [gradients]
        parts = [[] for _ in range(len(self.optimizers))]
        for k, g in nn.utils.tree_flatten(gradients):
            matched = False
            for i, fn in enumerate(self.filters):
                if fn(k, g):
                    parts[i].append((k, g))
                    matched = True
                    break
            if not matched:
                parts[-1].append((k, g))
        # tree_unflatten([]) returns [] which breaks Optimizer.init on empty
        # splits (structured heads may be dormant with structured=False). Force
        # {} in that case so downstream .init() treats it as an empty dict tree.
        return [nn.utils.tree_unflatten(p) if p else {} for p in parts]

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

# Structured verb heads get their own AdamW group with a heavier weight decay
# so rare verbs collapse toward the shared opt_head fallback instead of drifting.
_STRUCTURED_PARAMETER_PREFIXES = (
    "type_query.",
    "type_bias.",
)


def _use_muon_parameter(path: str, parameter: mx.array) -> bool:
    """Route only hidden 2D projection/Transformer weights to Muon."""
    if parameter.ndim != 2 or not path.endswith(".weight"):
        return False
    return not path.startswith(_ADAMW_PARAMETER_PREFIXES)


def _use_structured_adamw_parameter(path: str, parameter: mx.array) -> bool:
    """Route the verb-conditioned structured heads to a high-decay AdamW."""
    return path.startswith(_STRUCTURED_PARAMETER_PREFIXES)


def _optimizer_contract(cfg) -> dict:
    """Return the serializable optimizer identity used for safe resume."""
    return {
        "name": cfg.optimizer,
        "muon_momentum": float(cfg.muon_momentum),
        "muon_weight_decay": float(cfg.muon_weight_decay),
        "adamw_betas": [float(value) for value in cfg.adamw_betas],
        "adamw_eps": float(cfg.adamw_eps),
        "adamw_weight_decay": float(cfg.adamw_weight_decay),
        "structured_weight_decay": float(cfg.structured_weight_decay),
        "state_dtype": "float32",
        "parameter_dtype": "float16",
        "routing_version": 2,
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
    structured_adamw = FP32StateAdamW(
        learning_rate=cfg.lr,
        betas=[float(value) for value in cfg.adamw_betas],
        eps=cfg.adamw_eps,
        weight_decay=cfg.structured_weight_decay,
    )
    adamw = FP32StateAdamW(
        learning_rate=cfg.lr,
        betas=[float(value) for value in cfg.adamw_betas],
        eps=cfg.adamw_eps,
        weight_decay=cfg.adamw_weight_decay,
    )
    # Filter order matches optimizer order: 2D-hidden -> Muon, verb heads ->
    # structured AdamW, everything else -> default AdamW fallback.
    return PathSafeMultiOptimizer(
        [muon, structured_adamw, adamw],
        filters=[_use_muon_parameter, _use_structured_adamw_parameter],
    )


def _cross_entropy_sum(logits: mx.array, labels: mx.array) -> mx.array:
    """Return an FP32 sum so accumulation can normalize exactly once."""
    return nn.losses.cross_entropy(
        logits.astype(mx.float32), labels, reduction="sum"
    ).astype(mx.float32)


def _aux_loss(
    aux_dict: dict[str, mx.array],
    aux_targets: dict[str, mx.array],
    weights: dict[str, float],
) -> mx.array:
    """Compute the weighted mean of the four auxiliary-head losses.

    ``aux_targets`` carries an ``aux_valid`` row mask -- only rows with
    aux_valid=1 contribute. Returns a scalar FP32 loss in MEAN form (i.e.
    already divided by the valid-row count), 0.0 if no row in the batch is
    valid. Callers that accumulate loss in SUM form (to match
    ``_cross_entropy_sum``) must rescale by the row count themselves.
    """
    ko_logit = aux_dict["ko_logit"].astype(mx.float32)
    prize_pred = aux_dict["prize_pred"].astype(mx.float32)
    terminal_logit = aux_dict["terminal_logit"].astype(mx.float32)
    return_pred = aux_dict["return_pred"].astype(mx.float32)

    ko_tgt = aux_targets["aux_ko"].astype(mx.float32)
    prize_tgt = aux_targets["aux_prize_delta"].astype(mx.float32)
    terminal_tgt = aux_targets["aux_terminal"].astype(mx.float32)
    return_tgt = aux_targets["aux_return"].astype(mx.float32)
    valid = aux_targets["aux_valid"].astype(mx.float32)
    valid_sum = mx.maximum(mx.sum(valid), mx.array(1.0, dtype=mx.float32))

    ko_bce = (
        mx.logaddexp(mx.array(0.0, dtype=mx.float32), ko_logit) - ko_tgt * ko_logit
    )
    terminal_bce = (
        mx.logaddexp(mx.array(0.0, dtype=mx.float32), terminal_logit)
        - terminal_tgt * terminal_logit
    )
    prize_mse = (prize_pred - prize_tgt) ** 2
    return_mse = (return_pred - return_tgt) ** 2

    loss = (
        weights["ko"] * mx.sum(valid * ko_bce) / valid_sum
        + weights["prize"] * mx.sum(valid * prize_mse) / valid_sum
        + weights["terminal"] * mx.sum(valid * terminal_bce) / valid_sum
        + weights["return"] * mx.sum(valid * return_mse) / valid_sum
    )
    return loss


def _aux_metrics(
    aux_dict: dict[str, np.ndarray], aux_targets: dict[str, np.ndarray]
) -> dict[str, float]:
    """Per-head, unweighted, masked-mean metrics for reporting (numpy, host-side).

    Unlike ``_aux_loss`` (which returns one combined weighted scalar for
    backprop), this keeps every head separate so val logs can show
    ``aux_ko_bce``/``aux_prize_mse``/``aux_terminal_bce``/``aux_return_mse``
    independently of the configured loss weights.
    """
    valid = aux_targets["aux_valid"].astype(np.float32)
    valid_sum = max(float(valid.sum()), 1.0)
    ko_logit = aux_dict["ko_logit"].astype(np.float32)
    terminal_logit = aux_dict["terminal_logit"].astype(np.float32)
    prize_pred = aux_dict["prize_pred"].astype(np.float32)
    return_pred = aux_dict["return_pred"].astype(np.float32)
    ko_tgt = aux_targets["aux_ko"].astype(np.float32)
    terminal_tgt = aux_targets["aux_terminal"].astype(np.float32)
    prize_tgt = aux_targets["aux_prize_delta"].astype(np.float32)
    return_tgt = aux_targets["aux_return"].astype(np.float32)

    ko_bce = np.logaddexp(0.0, ko_logit) - ko_tgt * ko_logit
    terminal_bce = np.logaddexp(0.0, terminal_logit) - terminal_tgt * terminal_logit
    prize_mse = (prize_pred - prize_tgt) ** 2
    return_mse = (return_pred - return_tgt) ** 2

    return {
        "aux_ko_bce": float((valid * ko_bce).sum() / valid_sum),
        "aux_prize_mse": float((valid * prize_mse).sum() / valid_sum),
        "aux_terminal_bce": float((valid * terminal_bce).sum() / valid_sum),
        "aux_return_mse": float((valid * return_mse).sum() / valid_sum),
    }


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
    return_aux: bool = False,
    lane_aux_targets: list[dict[str, mx.array]] | None = None,
    aux_weights: dict[str, float] | None = None,
):
    """Unroll independent lanes in parallel without sharing recurrent memory.

    When ``aux_weights`` is given (together with a matching
    ``lane_aux_targets`` -- one dict of per-lane aux-column arrays per lane,
    row-aligned with that lane's ``lane_labels``/``lane_observations``),
    every decision forward calls ``model.logits_value_aux`` instead of
    ``model.logits_value``. The auxiliary predictions and targets from every
    decision across the whole temporal batch are accumulated and scored once
    at the end with ``_aux_loss`` (a single global masked mean over every
    valid row in the batch), then rescaled to SUM form -- matching
    ``_cross_entropy_sum`` -- and folded into the returned loss.
    """
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
    if (lane_aux_targets is None) != (aux_weights is None):
        raise ValueError(
            "lane_aux_targets and aux_weights must be provided together"
        )
    if lane_aux_targets is not None and len(lane_aux_targets) != lane_count:
        raise ValueError("lane_aux_targets must have one entry per lane")

    for labels, decision_lengths in zip(lane_labels, lane_decision_lengths):
        if not decision_lengths or any(length <= 0 for length in decision_lengths):
            raise ValueError("every TBPTT decision must contain at least one row")
        if sum(decision_lengths) != int(labels.shape[0]):
            raise ValueError("TBPTT decision lengths do not cover all lane rows")

    aux_active = aux_weights is not None
    loss_sum = mx.array(0.0, dtype=mx.float32)
    lane_memories = list(lane_memory_in)
    lane_offsets = [0] * lane_count
    lane_logits: list[list[mx.array]] = [[] for _ in range(lane_count)]
    aux_pred_parts: dict[str, list[mx.array]] = defaultdict(list)
    aux_target_parts: dict[str, list[mx.array]] = defaultdict(list)
    total_aux_rows = 0
    max_decisions = max(len(lengths) for lengths in lane_decision_lengths)

    for decision_index in range(max_decisions):
        active_lanes = [
            lane_index
            for lane_index, lengths in enumerate(lane_decision_lengths)
            if decision_index < len(lengths)
        ]
        decision_observations: list[dict[str, mx.array]] = []
        decision_labels: list[mx.array] = []
        decision_aux_targets: list[dict[str, mx.array]] = []
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
            if aux_active:
                decision_aux_targets.append(
                    {
                        key: value[row_start:row_stop]
                        for key, value in lane_aux_targets[lane_index].items()
                    }
                )
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
        if aux_active:
            logits, _, memory_out, aux_dict = model.logits_value_aux(
                combined_observation, memory_in=combined_memory
            )
            for aux_key, aux_value in aux_dict.items():
                aux_pred_parts[aux_key].append(aux_value)
            for aux_key in _AUX_COLUMNS:
                aux_target_parts[aux_key].append(
                    mx.concatenate(
                        [target[aux_key] for target in decision_aux_targets],
                        axis=0,
                    )
                )
            total_aux_rows += int(combined_labels.shape[0])
        else:
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

    aux_dict_all: dict[str, mx.array] | None = None
    aux_targets_all: dict[str, mx.array] | None = None
    if aux_active and total_aux_rows > 0:
        aux_dict_all = {
            key: mx.concatenate(parts, axis=0)
            for key, parts in aux_pred_parts.items()
        }
        aux_targets_all = {
            key: mx.concatenate(parts, axis=0)
            for key, parts in aux_target_parts.items()
        }
        aux_mean = _aux_loss(aux_dict_all, aux_targets_all, aux_weights)
        # Rescale mean -> sum form so it composes with the CE sum above and
        # the caller's single division by total example count at the
        # optimizer-step boundary yields mean(CE) + mean(aux), consistent
        # with the non-TBPTT scaling choice.
        loss_sum = loss_sum + aux_mean * total_aux_rows

    if any(memory is None for memory in lane_memories):
        raise ValueError("every TBPTT lane must advance at least one decision")
    combined_final_memory = mx.concatenate(
        [memory for memory in lane_memories if memory is not None],
        axis=0,
    )

    result: list = [loss_sum, combined_final_memory]
    if return_logits:
        result.append([mx.concatenate(parts, axis=0) for parts in lane_logits])
    if return_aux:
        if not aux_active:
            raise ValueError("return_aux requires aux_weights/lane_aux_targets")
        result.append(aux_dict_all)
        result.append(aux_targets_all)
    return tuple(result)


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
    episode_meta, train_rows: int
) -> list[list[np.ndarray]]:
    """Group ordered rows into decisions, preserving episode and side isolation.

    ``episode_meta`` only needs bracket-key access to parallel "episode_id" /
    "side" / "step_id" arrays of length >= train_rows -- a plain
    ``dict[str, np.ndarray]`` works exactly like the old structured-dtype
    ``episode_meta.npy`` array did here.
    """
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


def _build_tbptt_groups(episode_meta, train_rows: int) -> list[np.ndarray]:
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


def _read_batch_column(
    batch, name: str, shapes: dict[str, tuple], int_keys: set[str]
) -> np.ndarray:
    """One Parquet column of a RecordBatch -> its original (rows, *shape) array.

    Encoder-schema columns were written by build_bc_from_zips.py's
    ``_rows_to_table`` as either a plain scalar column (encoder shape ==
    (1,)) or a flattened FixedSizeList (row-major) for any larger shape.
    "opt_group" is the Parquet name for the old .npy-era "__group__" array
    and reuses action_mask's shape. Anything else is a flat per-row
    metadata/aux column (see ``_META_COLUMN_DTYPES``).
    """
    col = batch.column(name)
    if name == "opt_group":
        shape, dtype = shapes["action_mask"], np.int32
    elif name in shapes:
        shape = shapes[name]
        dtype = np.int32 if name in int_keys else np.float32
    else:
        shape, dtype = None, _META_COLUMN_DTYPES.get(name, np.float32)

    if shape is None:
        return col.to_numpy(zero_copy_only=False).astype(dtype)
    if shape == (1,):
        return col.to_numpy(zero_copy_only=False).astype(dtype).reshape(-1, 1)
    flat_len = int(np.prod(shape))
    flat = col.flatten().to_numpy(zero_copy_only=False).astype(dtype)
    assert flat.shape[0] == batch.num_rows * flat_len
    return flat.reshape((batch.num_rows, *shape))


def _materialize_columns(
    dataset,
    columns: list[str],
    row_filter,
    shapes: dict[str, tuple],
    int_keys: set[str],
    *,
    batch_size: int = _IO_BATCH_ROWS,
) -> dict[str, np.ndarray]:
    """Fully read ``columns`` for rows matching ``row_filter`` into RAM.

    Used for validation (always) and for TBPTT training (chunking needs
    ordered access across a whole episode-side trajectory, which a row-by-row
    Parquet stream cannot give cheaply -- see CLAUDE.md's TBPTT section).
    """
    parts: dict[str, list[np.ndarray]] = {c: [] for c in columns}
    for io_batch in dataset.to_batches(
        columns=columns, filter=row_filter, batch_size=batch_size
    ):
        if io_batch.num_rows == 0:
            continue
        for c in columns:
            parts[c].append(_read_batch_column(io_batch, c, shapes, int_keys))
    return {
        c: (
            np.concatenate(values, axis=0)
            if values
            else np.zeros(
                (0,) + shapes.get(c, ()),
                dtype=_META_COLUMN_DTYPES.get(c, np.float32),
            )
        )
        for c, values in parts.items()
    }


def _scan_tbptt_locations(
    dataset,
    row_filter,
    shapes: dict[str, tuple],
    int_keys: set[str],
    *,
    batch_size: int = _IO_BATCH_ROWS,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Pass 1 of the TBPTT streaming loader: read ONLY the metadata columns
    needed to build the TBPTT plan (episode_id/side/step_id and optional dedup
    label group), and record every kept row's physical location so pass 2 can
    fetch data columns without re-scanning.

    Returns
    -------
    episode_meta:
        dict[str, np.ndarray] with the same shape as the old materialized
        ``train_meta`` -- keys ``episode_id``, ``side``, ``step_id``.
    file_indices, row_group_indices, offsets:
        Parallel int32 arrays, one entry per kept row, giving the origin
        ``(fragment_index, row_group_index, offset_within_row_group)``. Pass 2
        uses these to load only the row_groups it needs.
    file_paths:
        Ordered fragment paths; ``file_indices[i]`` indexes into this list.

    Memory: ``episode_meta`` is 3 int arrays over the whole split (~24 bytes
    per row -> 330MB for 13.7M rows, the largest expected corpus). Locations
    add another 12 bytes/row -> 165MB. Combined ~500MB, vs ~5+GB for a full
    materialization.
    """
    columns = ["episode_id", "side", "step_id"]
    fragments = list(dataset.get_fragments())
    file_paths = [str(getattr(fr, "path", "?")) for fr in fragments]

    ep_parts: list[np.ndarray] = []
    side_parts: list[np.ndarray] = []
    step_parts: list[np.ndarray] = []
    file_parts: list[np.ndarray] = []
    rg_parts: list[np.ndarray] = []
    off_parts: list[np.ndarray] = []

    for file_idx, fragment in enumerate(fragments):
        pq_file = fragment.metadata
        n_row_groups = pq_file.num_row_groups
        for rg_idx in range(n_row_groups):
            # Load this row_group's metadata columns only, applying the split filter.
            rg_fragment = fragment.subset(row_group_ids=[rg_idx])
            for io_batch in rg_fragment.to_batches(
                columns=columns, filter=row_filter, batch_size=batch_size
            ):
                if io_batch.num_rows == 0:
                    continue
                # NOTE: to_batches with a filter drops non-matching rows, so
                # we CANNOT infer the intra-row-group offset from the batch's
                # position alone. Re-scan the same row_group without a filter
                # to recover ordinal positions of kept rows.
                pass
            # Simpler and cheaper: load the row_group WITHOUT a filter and
            # test the filter against a compiled predicate over episode_id.
            # Fragment.subset(row_group_ids=[k]).to_batches(columns=[...]) reads
            # the whole rg once, and the eid filter is a fast np.isin.
            eids_rg = None
            sides_rg = None
            steps_rg = None
            for io_batch in fragment.subset(
                row_group_ids=[rg_idx]
            ).to_batches(columns=columns, batch_size=batch_size):
                eb = io_batch.column("episode_id").to_numpy(zero_copy_only=False)
                sb = io_batch.column("side").to_numpy(zero_copy_only=False)
                tb = io_batch.column("step_id").to_numpy(zero_copy_only=False)
                eids_rg = eb if eids_rg is None else np.concatenate([eids_rg, eb])
                sides_rg = sb if sides_rg is None else np.concatenate([sides_rg, sb])
                steps_rg = tb if steps_rg is None else np.concatenate([steps_rg, tb])
            if eids_rg is None or len(eids_rg) == 0:
                continue
            # Apply the pyarrow filter as a compiled boolean over ``eids_rg``.
            # pyarrow filters can be arbitrary expressions, but for TBPTT the
            # only two we build are ``episode_id.isin([...])`` (train/val
            # split). Evaluate that at the numpy layer to avoid a second
            # parquet read pass.
            mask = _apply_episode_filter(row_filter, eids_rg)
            if not mask.any():
                continue
            offsets_in_rg = np.nonzero(mask)[0].astype(np.int32)
            n_kept = len(offsets_in_rg)
            ep_parts.append(eids_rg[mask].astype(np.int64))
            side_parts.append(sides_rg[mask].astype(np.int32))
            step_parts.append(steps_rg[mask].astype(np.int32))
            file_parts.append(np.full(n_kept, file_idx, dtype=np.int32))
            rg_parts.append(np.full(n_kept, rg_idx, dtype=np.int32))
            off_parts.append(offsets_in_rg)

    if not ep_parts:
        empty_meta = {
            "episode_id": np.empty(0, dtype=np.int64),
            "side": np.empty(0, dtype=np.int32),
            "step_id": np.empty(0, dtype=np.int32),
        }
        return (
            empty_meta,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            file_paths,
        )

    episode_meta = {
        "episode_id": np.concatenate(ep_parts),
        "side": np.concatenate(side_parts),
        "step_id": np.concatenate(step_parts),
    }
    return (
        episode_meta,
        np.concatenate(file_parts),
        np.concatenate(rg_parts),
        np.concatenate(off_parts),
        file_paths,
    )


_TBPTT_FILTER_CACHE: dict[int, np.ndarray] = {}


def _apply_episode_filter(row_filter, episode_ids: np.ndarray) -> np.ndarray:
    """Evaluate the TBPTT split filter against a numpy array of episode_ids.

    The split filter built by the trainer is always ``episode_id.isin([...])``
    (train_filter / val_filter -- see the ``pads.field("episode_id").isin(...)``
    calls in ``main``). Extract the id set once from the expression's repr
    and evaluate ``np.isin`` at the numpy layer so pass 1 avoids a second
    parquet read. Any other filter shape is a contract violation and raises
    (no silent fallback).
    """
    key = id(row_filter)
    cached_ids = _TBPTT_FILTER_CACHE.get(key)
    if cached_ids is None:
        raise RuntimeError(
            "TBPTT episode filter was not registered before use. "
            "Call sites must populate _TBPTT_FILTER_CACHE[id(filter)] "
            "with the exact episode_id numpy array used to build the "
            "pyarrow is_in expression -- pyarrow does not expose its "
            "value_set as a public property, and its repr abbreviates "
            "long lists, so re-parsing the repr is unreliable."
        )
    return np.isin(episode_ids, cached_ids)


class _ParquetRowGroupCache:
    """Hierarchical KV cache of decoded parquet row_groups.

    Design (mirrors an LLM KV cache with hot/transient zones + spill tier):

    * **Zone HOT (60% of resident)** — pinned. An entry is promoted here once
      its per-entry hit count crosses ``_HOT_PROMOTION_HITS``. Hot entries
      are immune to memory-pressure eviction; the intent is that things we
      touch repeatedly (row_groups iterated every epoch) sit rent-free.
    * **Zone TRANSIENT (40% of resident)** — pure LRU. New entries land here
      and rotate under memory pressure. Rotation criterion is LRU because
      within the transient window recency and frequency are correlated.
    * **Opt-step protection** — during an ``in_opt_step()`` context, evictions
      are suppressed. MLX's forward+backward activation peak briefly pushes
      host memory above the water-mark; we don't want the row_group we just
      finished reading to be evicted mid-step and re-decoded on the next.
    * **SSD tier** — an evicted transient entry is spilled to
      ``<scratch>/<key>.npz`` before its RAM buffer is released. A subsequent
      miss checks the SSD tier first; a hit there re-hydrates directly from
      disk instead of re-decoding the parquet row_group (cheaper on
      compressed columnar data, and on unified-memory Macs the .npz reads
      go through the block cache anyway).
    * **Device-aware pressure probe** — on unified memory (M-series /
      ``mx.metal``) we probe host memory percent because RAM and VRAM share
      one pool. On x86 + discrete accelerator (A100) we would probe VRAM
      via ``mx.cuda`` or, absent MLX support, delegate to ``torch.cuda``.
      The probe returns ``True`` when the OS is about to swap or the
      accelerator is about to OOM.

    Keyed by ``(file_idx, row_group_idx)``. Each value is a dict of
    ``column_name -> np.ndarray`` already reshaped by ``_read_batch_column``.
    """

    # macOS starts compressing / swapping around 85% of system memory used
    # (Activity Monitor's "yellow" memory-pressure band). On unified memory
    # this is our combined RAM+VRAM ceiling.
    _HIGH_WATERMARK_PCT = 85.0
    # Number of hits an entry must accumulate before promotion to the hot
    # zone. Chosen so a row_group touched at least twice (typical from
    # multiple chunks per row_group in the TBPTT plan) can survive rotation.
    _HOT_PROMOTION_HITS = 2
    # Fraction of the current resident set reserved for hot pinning. Values
    # near 1.0 approach full pinning (no rotation); near 0.0 approach pure
    # LRU with no hot layer. 0.6 keeps enough working set warm without
    # freezing the whole cache.
    _HOT_ZONE_FRACTION = 0.6

    def __init__(
        self,
        file_paths: list[str],
        columns: list[str],
        shapes: dict[str, tuple],
        int_keys: set[str],
        *,
        ssd_spill_dir: str | None = None,
    ) -> None:
        import pyarrow.parquet as pq  # local to avoid startup cost when unused
        from collections import OrderedDict
        self._pq = pq
        self._file_paths = file_paths
        # ParquetFile handles: opened lazily and thread-safe for concurrent
        # read_row_group() calls (pyarrow's C++ core takes its own locks).
        # Wrapping ``_open`` in ``_lock`` still needed because the dict
        # ``self._pq_files`` is populated in check-then-insert.
        self._pq_files: dict[int, "pq.ParquetFile"] = {}
        self._columns = list(columns)
        self._shapes = shapes
        self._int_keys = int_keys
        # Two OrderedDicts model the tiers. ``_transient`` is LRU (oldest at
        # front, newest at back). ``_hot`` order does not matter for
        # eviction but preserves insertion order for a stable report.
        self._transient: OrderedDict[
            tuple[int, int], dict[str, np.ndarray]
        ] = OrderedDict()
        self._hot: OrderedDict[
            tuple[int, int], dict[str, np.ndarray]
        ] = OrderedDict()
        # Per-entry hit counters. Kept for evicted entries too so a later
        # re-load can restore its hot status without waiting for the counter
        # to climb again.
        self._hits_by_key: dict[tuple[int, int], int] = {}
        # SSD spill tier. When an entry rotates out of RAM we write it as a
        # single .npz alongside its key. On miss we check disk before
        # re-decoding parquet. Directory is per-cache so train and val do
        # not share spill files (they may hold different column subsets).
        if ssd_spill_dir is not None:
            os.makedirs(ssd_spill_dir, exist_ok=True)
        self._ssd_spill_dir = ssd_spill_dir
        self._ssd_keys: set[tuple[int, int]] = set()
        # Fine-grained lock: guards _transient / _hot / _hits_by_key / stats.
        # The pyarrow decode itself releases the GIL and runs unlocked so
        # the prefetch thread and the main thread genuinely overlap I/O
        # with GPU work.
        self._lock = threading.Lock()
        # Reentrant depth counter so nested ``in_opt_step`` contexts work
        # (e.g. gradient accumulation calls the forward path more than once
        # per optimizer step, each protected). Non-zero => suppress
        # eviction.
        self._opt_step_depth = 0
        # Instrumentation.
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._promotions = 0
        self._ssd_hits = 0
        self._ssd_spills = 0
        self._bytes_loaded = 0
        try:
            import psutil as _psutil  # noqa: F401 -- probe availability
            self._pressure_probe = True
        except ImportError:
            self._pressure_probe = False

    # ---- opt-step protection ------------------------------------------------
    @contextmanager
    def in_opt_step(self):
        """Suppress eviction while the caller is inside a forward/backward /
        optimizer.update. Reentrant. Exit resumes normal pressure eviction.
        """
        with self._lock:
            self._opt_step_depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._opt_step_depth = max(0, self._opt_step_depth - 1)

    # ---- pressure probe (device-aware) --------------------------------------
    def _under_pressure(self) -> bool:
        # Unified-memory devices (M-series Macs) share RAM and VRAM; probing
        # host memory percent is the correct signal. For discrete VRAM the
        # right probe is ``mx.metal.get_active_memory()`` divided by the
        # device limit, or ``torch.cuda.mem_get_info()``. Add per-device
        # branches here when a run targets an accelerator with a separate
        # VRAM pool.
        if not self._pressure_probe:
            return False
        import psutil
        return psutil.virtual_memory().percent >= self._HIGH_WATERMARK_PCT

    # ---- pyarrow helpers ----------------------------------------------------
    def _open(self, file_idx: int):
        with self._lock:
            pf = self._pq_files.get(file_idx)
            if pf is None:
                pf = self._pq.ParquetFile(self._file_paths[file_idx])
                self._pq_files[file_idx] = pf
        return pf

    def _load_rg(
        self, file_idx: int, row_group_idx: int
    ) -> dict[str, np.ndarray]:
        pf = self._open(file_idx)
        table = pf.read_row_group(row_group_idx, columns=self._columns)
        rg_bytes = 0
        rg_dict: dict[str, np.ndarray] = {}
        record_batch = table.combine_chunks().to_batches()
        if not record_batch:
            for c in self._columns:
                dtype = (
                    np.int32
                    if c in self._int_keys
                    or c == "opt_group"
                    else _META_COLUMN_DTYPES.get(c, np.float32)
                )
                rg_dict[c] = np.zeros(
                    (0,) + self._shapes.get(c, ()),
                    dtype=dtype,
                )
            return rg_dict
        rb = record_batch[0]
        for c in self._columns:
            rg_dict[c] = _read_batch_column(rb, c, self._shapes, self._int_keys)
            rg_bytes += int(rg_dict[c].nbytes)
        with self._lock:
            self._bytes_loaded += rg_bytes
        return rg_dict

    # ---- SSD spill tier -----------------------------------------------------
    def _ssd_path(self, key: tuple[int, int]) -> str:
        assert self._ssd_spill_dir is not None
        return os.path.join(
            self._ssd_spill_dir, f"rg_{key[0]}_{key[1]}.npz"
        )

    def _spill_to_ssd(
        self, key: tuple[int, int], rg: dict[str, np.ndarray]
    ) -> bool:
        if self._ssd_spill_dir is None:
            return False
        try:
            np.savez(self._ssd_path(key), **rg)
            return True
        except OSError:
            # Disk full or permission trouble -- silently drop the entry.
            return False

    def _load_from_ssd(
        self, key: tuple[int, int]
    ) -> dict[str, np.ndarray] | None:
        if self._ssd_spill_dir is None:
            return None
        if key not in self._ssd_keys:
            return None
        try:
            with np.load(self._ssd_path(key)) as npz:
                return {c: np.array(npz[c]) for c in npz.files}
        except (OSError, KeyError):
            return None

    # ---- eviction -----------------------------------------------------------
    def _rebalance_and_evict_locked(self) -> None:
        """Called under ``self._lock`` from the miss path. Evicts from the
        transient tier until host is below the water-mark, spilling each
        evicted entry to SSD if the spill dir is configured. Hot tier is
        never evicted here -- it's the whole point of promotion. Callers
        must hold the lock.
        """
        if self._opt_step_depth > 0:
            return
        while self._under_pressure() and self._transient:
            key, rg = self._transient.popitem(last=False)
            self._evictions += 1
            # Attempt SSD spill BEFORE releasing the reference so a partial
            # write does not lose the entry silently. Success adds to
            # _ssd_keys, failure just drops the entry.
            if self._spill_to_ssd(key, rg):
                self._ssd_spills += 1
                self._ssd_keys.add(key)

    def _rebalance_zones_locked(self) -> None:
        """Enforce hot/transient split according to ``_HOT_ZONE_FRACTION``.
        Called under lock. If the hot zone has grown past its share, demote
        the least-recently-promoted entries back to transient (they can be
        re-promoted quickly if they stay warm).
        """
        total = len(self._hot) + len(self._transient)
        hot_budget = int(total * self._HOT_ZONE_FRACTION)
        while len(self._hot) > hot_budget and self._hot:
            k, v = self._hot.popitem(last=False)
            # Insert at the newest slot of the transient LRU -- a demoted
            # entry gets a full cycle to prove itself before rotation.
            self._transient[k] = v
            self._transient.move_to_end(k)

    # ---- primary touch (fast path + slow path) ------------------------------
    def _touch(self, key: tuple[int, int]) -> dict[str, np.ndarray]:
        # Fast path A: hot tier hit.
        with self._lock:
            rg = self._hot.get(key)
            if rg is not None:
                self._hits += 1
                self._hits_by_key[key] = self._hits_by_key.get(key, 0) + 1
                return rg
            # Fast path B: transient tier hit -> update recency and check
            # for promotion.
            rg = self._transient.get(key)
            if rg is not None:
                self._transient.move_to_end(key)
                self._hits += 1
                new_hits = self._hits_by_key.get(key, 0) + 1
                self._hits_by_key[key] = new_hits
                if new_hits >= self._HOT_PROMOTION_HITS:
                    del self._transient[key]
                    self._hot[key] = rg
                    self._promotions += 1
                    self._rebalance_zones_locked()
                return rg

        # Slow path 1: try the SSD spill tier before touching parquet again.
        ssd_rg = self._load_from_ssd(key)
        if ssd_rg is not None:
            with self._lock:
                # Concurrent load may have already re-hydrated the entry.
                for tier in (self._hot, self._transient):
                    existing = tier.get(key)
                    if existing is not None:
                        self._hits += 1
                        return existing
                self._transient[key] = ssd_rg
                self._hits += 1
                self._ssd_hits += 1
                self._hits_by_key[key] = self._hits_by_key.get(key, 0) + 1
                self._rebalance_and_evict_locked()
            return ssd_rg

        # Slow path 2: decode from parquet. Runs unlocked so the prefetch
        # thread and the main thread can genuinely overlap.
        rg_loaded = self._load_rg(*key)
        with self._lock:
            for tier in (self._hot, self._transient):
                existing = tier.get(key)
                if existing is not None:
                    self._hits += 1
                    return existing
            self._transient[key] = rg_loaded
            self._misses += 1
            self._hits_by_key[key] = self._hits_by_key.get(key, 0) + 1
            self._rebalance_and_evict_locked()
        return rg_loaded

    def report(self) -> dict[str, int]:
        """Snapshot of cache statistics for end-of-training logs."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "promotions": self._promotions,
                "ssd_hits": self._ssd_hits,
                "ssd_spills": self._ssd_spills,
                "bytes_loaded": self._bytes_loaded,
                "resident_row_groups": len(self._hot) + len(self._transient),
                "resident_hot": len(self._hot),
                "resident_transient": len(self._transient),
                "ssd_resident": len(self._ssd_keys),
            }

    def read_rows(
        self,
        row_indices: np.ndarray,
        file_indices: np.ndarray,
        row_group_indices: np.ndarray,
        offsets: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Fetch every column at the given ``row_indices``.

        ``row_indices`` are the split-local row ids that pass 1 wrote into
        ``episode_meta``. ``file_indices``, ``row_group_indices``, ``offsets``
        are the parallel location arrays returned by pass 1: this method reads
        those parallel arrays at ``row_indices`` to know which row_group and
        offset each output row comes from.
        """
        sel_files = file_indices[row_indices]
        sel_rgs = row_group_indices[row_indices]
        sel_offs = offsets[row_indices]

        n = len(row_indices)
        # Group requested rows by (file, row_group). np.lexsort keeps ordering
        # stable so an argsort round-trip lets us gather then scatter into the
        # caller's requested order.
        order = np.lexsort((sel_offs, sel_rgs, sel_files))
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(n)

        gathered: dict[str, np.ndarray | None] = {c: None for c in self._columns}
        i = 0
        while i < n:
            j = i
            f = int(sel_files[order[i]])
            g = int(sel_rgs[order[i]])
            while (
                j < n
                and int(sel_files[order[j]]) == f
                and int(sel_rgs[order[j]]) == g
            ):
                j += 1
            rg_dict = self._touch((f, g))
            block = order[i:j]
            row_ids_in_rg = sel_offs[block]
            for c in self._columns:
                arr = rg_dict[c][row_ids_in_rg]
                if gathered[c] is None:
                    gathered[c] = np.empty(
                        (n,) + arr.shape[1:], dtype=arr.dtype
                    )
                gathered[c][block] = arr  # writes into sorted-order slots
            i = j

        # Rewind sorted-order slots back to caller-requested order.
        out: dict[str, np.ndarray] = {}
        for c in self._columns:
            data = gathered[c]
            if data is None:
                data = np.zeros(
                    (0,) + self._shapes.get(c, ()),
                    dtype=np.int32
                    if c in self._int_keys
                    or c == "opt_group"
                    else _META_COLUMN_DTYPES.get(c, np.float32),
                )
            out[c] = data[inv_order]
        return out


def _apply_dedup_relabel(chunk: dict[str, np.ndarray]) -> None:
    """Remap chunk["y"] to its canonical dedup-group label, in place.

    Mirrors the old .npy loader's one-shot ``labels = group[arange, labels]``
    pass, applied per chunk instead of once over the whole label array -- the
    remap is purely row-local so the two are exactly equivalent.
    """
    group = chunk.get("opt_group")
    if group is None:
        return
    y = chunk["y"]
    chunk["y"] = group[np.arange(len(y)), y]


def _stream_train_microbatches(
    dataset,
    columns: list[str],
    row_filter,
    shapes: dict[str, tuple],
    int_keys: set[str],
    batch_size: int,
    seed_key: list,
    *,
    io_batch_rows: int = _IO_BATCH_ROWS,
):
    """Yield exactly ``ceil(n_rows / batch_size)`` fixed-size microbatches.

    Reads pyarrow I/O batches (whatever size pyarrow's row-group partitioning
    happens to produce, capped at io_batch_rows) and re-chunks them through a
    carry buffer, so the emitted microbatch boundary never depends on that
    internal partitioning -- the trainer needs an exact microbatch count
    computed independently ahead of time (``_ceil_div(n_train, batch_size)``)
    for the optimizer-step/scheduler accounting to stay exact. Each I/O
    batch's rows are shuffled (seeded on seed_key + io-batch index) before
    entering the carry buffer.
    """
    carry: dict[str, np.ndarray] | None = None
    io_index = 0
    for io_batch in dataset.to_batches(
        columns=columns, filter=row_filter, batch_size=io_batch_rows
    ):
        if io_batch.num_rows == 0:
            continue
        chunk = {
            c: _read_batch_column(io_batch, c, shapes, int_keys) for c in columns
        }
        _apply_dedup_relabel(chunk)
        n = io_batch.num_rows
        perm = np.random.default_rng([*seed_key, io_index]).permutation(n)
        io_index += 1
        chunk = {c: v[perm] for c, v in chunk.items()}
        if carry is not None:
            chunk = {
                c: np.concatenate([carry[c], chunk[c]], axis=0) for c in columns
            }
        total = len(chunk[columns[0]])
        pos = 0
        while total - pos >= batch_size:
            yield {c: chunk[c][pos : pos + batch_size] for c in columns}
            pos += batch_size
        carry = {c: chunk[c][pos:] for c in columns} if pos < total else None
    if carry is not None and len(carry[columns[0]]) > 0:
        yield carry


def _resolve_day_ids(args, cfg, db: "ResultsDB") -> list[int]:
    """Resolve the `days` table ids selected by CLI flags or config."""
    selectors = [bool(args.all_days), bool(args.last_n_days), bool(args.days)]
    if sum(selectors) > 1:
        raise SystemExit("use only one of --days, --last-n-days, --all-days")
    if args.all_days:
        rows = db.conn.execute("SELECT id FROM days ORDER BY date").fetchall()
        return [int(row["id"]) for row in rows]
    if args.last_n_days:
        n = int(args.last_n_days)
        rows = db.conn.execute(
            "SELECT id, date FROM days ORDER BY date DESC LIMIT ?", (n,)
        ).fetchall()
        if len(rows) < n:
            print(
                f"[bc-train-mlx] --last-n-days {n}: only {len(rows)} day(s) "
                f"registered in the catalog; using them all",
                flush=True,
            )
        # Return in chronological (oldest first) order so training walks the
        # meta forward in time, matching the intent of ``--days`` when the
        # user types the list in ISO order.
        return [int(row["id"]) for row in reversed(rows)]
    dates: list[str] = []
    if args.days:
        dates = [d.strip() for d in args.days.split(",") if d.strip()]
    elif cfg.training_days:
        dates = list(cfg.training_days)
    if not dates:
        raise SystemExit(
            "no training days specified; use --days, --last-n-days, --all-days, "
            "or configure training_days"
        )
    day_ids = []
    for date in dates:
        row = db.conn.execute("SELECT id FROM days WHERE date = ?", (date,)).fetchone()
        if row is None:
            raise SystemExit(f"training day not found in catalog: {date!r}")
        day_ids.append(int(row["id"]))
    return day_ids


def _load_dataset_manifest(parquet_path: str) -> dict:
    """Load the sidecar ``{date}.manifest.json`` written next to one Parquet file."""
    p = Path(parquet_path)
    manifest_path = p.with_name(p.stem + ".manifest.json")
    if not manifest_path.is_file():
        raise SystemExit(
            f"missing dataset manifest for {parquet_path}: expected {manifest_path}"
        )
    with open(manifest_path, encoding="utf-8") as handle:
        return json.load(handle)


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
        help=argparse.SUPPRESS,  # removed: see --days/--last-n-days/--all-days
    )
    p.add_argument("--config", default=None, help="Path to JSON config file")
    p.add_argument(
        "--days",
        default=None,
        help="Comma-separated training days, e.g. 2026-07-30,2026-08-01",
    )
    p.add_argument(
        "--last-n-days",
        type=int,
        default=None,
        help="Use the N most recently registered training days",
    )
    p.add_argument(
        "--all-days",
        action="store_true",
        help="Use every day registered in the results catalog",
    )
    p.add_argument(
        "--db",
        default="model/results.db",
        help="Path to the SQLite results catalog (default: model/results.db)",
    )
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
    p.add_argument(
        "--aux-ko-weight",
        type=float,
        default=None,
        help="Weight for the ko_head_aux BCE loss (0=disabled)",
    )
    p.add_argument(
        "--aux-prize-weight",
        type=float,
        default=None,
        help="Weight for the prize_head_aux MSE loss (0=disabled)",
    )
    p.add_argument(
        "--aux-terminal-weight",
        type=float,
        default=None,
        help="Weight for the terminal_head_aux BCE loss (0=disabled)",
    )
    p.add_argument(
        "--aux-return-weight",
        type=float,
        default=None,
        help="Weight for the return_head_aux MSE loss (0=disabled)",
    )
    # Trainer options
    p.add_argument(
        "--compile",
        type=_bool_arg,
        default=None,
        metavar="true|false",
        help="mx.compile the loss function",
    )
    p.add_argument("--log-interval", type=int, default=None)
    # Data
    p.add_argument("--val-frac", type=float, default=None)
    p.add_argument("--val-batch-size", type=int, default=None)
    p.add_argument(
        "--max-rows", type=int, default=None, help="Max training rows (0=all)"
    )
    p.add_argument(
        "--max-rows-per-day",
        type=int,
        default=None,
        help="Cap training rows PER DAY (episode-boundary rounded, 0=off)",
    )
    p.add_argument(
        "--top-elo",
        type=int,
        default=None,
        help=(
            "Filter to episodes played by top-N agents by daily remote elo "
            "(source=remote). Both player_name and opponent_name must be "
            "in the top-N set for that day. 0=off"
        ),
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
        "aux_ko_weight": "aux_ko_weight",
        "aux_prize_weight": "aux_prize_weight",
        "aux_terminal_weight": "aux_terminal_weight",
        "aux_return_weight": "aux_return_weight",
        "compile": "compile",
        "log_interval": "log_interval",
        "val_frac": "val_frac",
        "val_batch_size": "val_batch_size",
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
    # No CLI flag exists for this one (config-only knob); _build_optimizer(a)
    # reads it straight off the namespace like every other optimizer field.
    a.structured_weight_decay = cfg.structured_weight_decay
    a.tbptt_chunk = a.tbptt_chunk if a.tbptt_chunk is not None else cfg.tbptt_chunk
    a.aux_ko_weight = (
        a.aux_ko_weight if a.aux_ko_weight is not None else cfg.aux_ko_weight
    )
    a.aux_prize_weight = (
        a.aux_prize_weight if a.aux_prize_weight is not None else cfg.aux_prize_weight
    )
    a.aux_terminal_weight = (
        a.aux_terminal_weight
        if a.aux_terminal_weight is not None
        else cfg.aux_terminal_weight
    )
    a.aux_return_weight = (
        a.aux_return_weight if a.aux_return_weight is not None else cfg.aux_return_weight
    )
    a.compile = a.compile if a.compile is not None else cfg.compile
    a.log_interval = a.log_interval if a.log_interval is not None else cfg.log_interval
    a.val_frac = a.val_frac if a.val_frac is not None else cfg.val_frac
    a.val_batch_size = (
        a.val_batch_size if a.val_batch_size is not None else cfg.val_batch_size
    )
    a.max_rows = a.max_rows if a.max_rows is not None else cfg.max_rows
    # Trainer-only CLI knobs (no config counterparts). Default 0 = disabled.
    a.max_rows_per_day = int(a.max_rows_per_day) if a.max_rows_per_day else 0
    a.top_elo = int(a.top_elo) if a.top_elo else 0
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

    # Auxiliary-head multi-task loss: active iff any weight is positive. Aux
    # targets come from the parquet dataset (aux_ko, aux_prize_delta,
    # aux_terminal, aux_return, aux_valid) -- see _AUX_COLUMNS / _aux_loss.
    aux_weights = {
        "ko": float(a.aux_ko_weight),
        "prize": float(a.aux_prize_weight),
        "terminal": float(a.aux_terminal_weight),
        "return": float(a.aux_return_weight),
    }
    aux_active = any(w > 0.0 for w in aux_weights.values())
    if aux_active:
        print(
            f"[bc-train-mlx] aux losses active — ko={aux_weights['ko']}, "
            f"prize={aux_weights['prize']}, terminal={aux_weights['terminal']}, "
            f"return={aux_weights['return']}",
            flush=True,
        )

    # MLX auto-detecta GPU (Metal) — sem device management!
    print(f"[bc-train-mlx] device={mx.default_device()}", flush=True)

    if a.data is not None:
        raise SystemExit(
            "the .npy dataset path arg is no longer supported; use "
            "--days/--all-days/--last-n-days or the training_days config field"
        )

    # ---- resolve training days from the SQLite results catalog ----
    db = ResultsDB(a.db)
    try:
        day_ids = _resolve_day_ids(a, cfg, db)
        datasets = db.list_datasets_by_days(day_ids)
    finally:
        db.close()
    if not datasets:
        raise SystemExit(
            f"no datasets registered in {a.db} for the requested training days"
        )
    for row in datasets:
        if not os.path.isfile(row["path"]):
            raise SystemExit(f"dataset parquet file missing on disk: {row['path']}")
    dataset_paths = [row["path"] for row in datasets]
    print(
        "[bc-train-mlx] training days: "
        + ", ".join(f"{row['day_date']}({row['rows']:,} rows)" for row in datasets),
        flush=True,
    )

    # ---- would-KO contract, checked against every selected day's manifest ----
    manifests = [_load_dataset_manifest(p) for p in dataset_paths]
    would_ko_blocks = [m.get("would_ko") or {} for m in manifests]
    would_ko_enabled_flags = {bool(b.get("enabled", False)) for b in would_ko_blocks}
    if len(would_ko_enabled_flags) > 1:
        raise ValueError(
            "selected training days disagree on would-KO ('enabled' differs "
            "across days); pick a homogeneous set of days"
        )
    dataset_would_ko = (
        would_ko_enabled_flags.pop() if would_ko_enabled_flags else False
    )
    if dataset_would_ko != bool(cfg.bc_would_ko):
        raise ValueError(
            "training config and dataset disagree on would-KO: "
            f"config={bool(cfg.bc_would_ko)}, dataset={dataset_would_ko}. "
            "Rebuild the dataset(s) with the current run sheet."
        )
    if dataset_would_ko:
        wk_nvars = {
            int((m.get("build_config") or {}).get("bc_wk_nvar", -1))
            for m in manifests
        }
        if wk_nvars != {int(cfg.bc_wk_nvar)}:
            raise ValueError(
                "training config and dataset(s) disagree on bc_wk_nvar: "
                f"config={int(cfg.bc_wk_nvar)}, dataset(s)={sorted(wk_nvars)}"
            )
        statuses = {str(b.get("status", "missing")) for b in would_ko_blocks}
        if statuses != {"computed"}:
            raise ValueError(
                "would-KO dataset is not computation-complete on every "
                f"selected day: statuses={sorted(statuses)}"
            )
        dataset_would_ko_nvar = int(cfg.bc_wk_nvar)
        dataset_would_ko_status = "computed"
    else:
        dataset_would_ko_nvar = int(cfg.bc_wk_nvar)
        dataset_would_ko_status = "disabled"

    effective_would_ko = bool(dataset_would_ko and not a.zero_wouldko)
    inference_provenance = "verified-dataset-manifest"
    if a.zero_wouldko:
        inference_provenance = "explicit-zero-would-ko"
        print(
            "[bc-train-mlx] would-KO features explicitly zeroed for this run; "
            "checkpoint inference will disable would-KO",
            flush=True,
        )

    dataset_manifest = {
        "days": [
            {
                "date": row["day_date"],
                "path": row["path"],
                "rows": int(row["rows"]),
                "sha256": row["sha256"],
            }
            for row in datasets
        ],
        "would_ko": {
            "enabled": dataset_would_ko,
            "status": dataset_would_ko_status,
            "n_var": dataset_would_ko_nvar,
        },
        "schema_versions": sorted(
            {int(m.get("schema_version", -1)) for m in manifests}
        ),
    }
    dataset_build_fingerprint = hashlib.sha256(
        "|".join(sorted(row["sha256"] for row in datasets)).encode("utf-8")
    ).hexdigest()

    # ---- encoder / token schema (needed to reshape Parquet columns) ----
    ct = get_card_table()
    enc = TokenEncoder(ct)
    int_keys = set(enc.int_keys)
    enc_shapes = dict(enc.shapes)
    keys = sorted(enc_shapes.keys())

    # ---- pyarrow streaming dataset over every selected day's Parquet file ----
    pa_dataset = pads.dataset(dataset_paths, format="parquet")

    # One pass over the "episode_id" column (single int64 column, cheap even
    # for a large multi-day corpus) builds the episode universe for the
    # train/val split and, if max_rows is set, the smoke-test row cap.
    # Parquet has no cheap absolute row-position slice across multiple files,
    # so max_rows caps by whole episode (kept in first-appearance/day order)
    # rather than by exact row count -- fine for its only real use, smoke
    # tests.
    # Scan (episode_id, day_id, player_name, opponent_name) per row so that
    # both --top-elo and --max-rows-per-day can be applied at the episode
    # level BEFORE the train/val split. player_name / opponent_name are
    # sidecar-emitted string columns already present in every parquet built
    # by build_bc_from_zips.py.
    scan_cols = ["episode_id", "day_id"]
    if a.top_elo and a.top_elo > 0:
        scan_cols.extend(["player_name", "opponent_name"])
    scan_chunks: dict[str, list[np.ndarray]] = {c: [] for c in scan_cols}
    for batch in pa_dataset.to_batches(columns=scan_cols):
        if batch.num_rows == 0:
            continue
        for c in scan_cols:
            scan_chunks[c].append(batch.column(c).to_numpy(zero_copy_only=False))
    if not scan_chunks["episode_id"]:
        raise SystemExit("selected training days contain no rows")
    all_eids = np.concatenate(scan_chunks["episode_id"])
    all_day_ids = np.concatenate(scan_chunks["day_id"])
    all_player = (
        np.concatenate(scan_chunks["player_name"])
        if "player_name" in scan_chunks and scan_chunks["player_name"]
        else None
    )
    all_opponent = (
        np.concatenate(scan_chunks["opponent_name"])
        if "opponent_name" in scan_chunks and scan_chunks["opponent_name"]
        else None
    )
    unique_eids, first_index, counts = np.unique(
        all_eids, return_index=True, return_counts=True
    )
    appearance_order = np.argsort(first_index)
    unique_eids = unique_eids[appearance_order]
    counts = counts[appearance_order]
    first_index = first_index[appearance_order]
    # One canonical (day_id, player_name, opponent_name) tuple per episode --
    # every row of an episode carries the same values, so first_index is the
    # cheapest representative.
    ep_day_ids = all_day_ids[first_index]
    ep_player = all_player[first_index] if all_player is not None else None
    ep_opponent = all_opponent[first_index] if all_opponent is not None else None

    # --top-elo filter: keep only episodes where BOTH sides are top-N on their
    # calendar day. Names are joined off the SQLite catalog (agent_elo_daily
    # + agents), source='remote' (the only source populated during rebuild).
    if a.top_elo and a.top_elo > 0:
        if ep_player is None:
            raise SystemExit(
                "--top-elo requires player_name/opponent_name columns "
                "in the parquet; rebuild the dataset"
            )
        import sqlite3 as _sqlite3_top
        _top_conn = _sqlite3_top.connect(str(a.db))
        _top_conn.row_factory = _sqlite3_top.Row
        top_names_by_day: dict[int, set[str]] = {}
        try:
            for row in _top_conn.execute(
                "SELECT day_id, name FROM ("
                "  SELECT aed.day_id, a.name, aed.elo, "
                "         ROW_NUMBER() OVER (PARTITION BY aed.day_id ORDER BY aed.elo DESC) AS rk "
                "  FROM agent_elo_daily aed "
                "  JOIN agents a ON a.id = aed.agent_id "
                "  WHERE aed.source = 'remote'"
                ") WHERE rk <= ?",
                (int(a.top_elo),),
            ):
                top_names_by_day.setdefault(int(row["day_id"]), set()).add(
                    str(row["name"])
                )
        finally:
            _top_conn.close()
        keep = np.zeros(len(unique_eids), dtype=bool)
        for i, day_id in enumerate(ep_day_ids):
            allowed = top_names_by_day.get(int(day_id))
            if not allowed:
                continue
            if str(ep_player[i]) in allowed and str(ep_opponent[i]) in allowed:
                keep[i] = True
        kept = int(keep.sum())
        total = len(unique_eids)
        print(
            f"[bc-train-mlx] --top-elo {a.top_elo}: kept {kept:,}/{total:,} "
            f"episode(s) played by top-{a.top_elo} agents on both sides",
            flush=True,
        )
        unique_eids = unique_eids[keep]
        counts = counts[keep]
        ep_day_ids = ep_day_ids[keep]
        if not len(unique_eids):
            raise SystemExit(
                "--top-elo filter left no episodes; lower N or check "
                "agent_elo_daily coverage"
            )

    # --max-rows-per-day: cap per calendar day (episode-boundary rounded) so
    # multi-day suites train on the same per-day budget regardless of the
    # replay volume. Ordering is stable (day_id ASC, appearance order within
    # a day) to keep runs comparable.
    if a.max_rows_per_day and a.max_rows_per_day > 0:
        order = np.lexsort((np.arange(len(unique_eids)), ep_day_ids))
        selected_positions: list[int] = []
        rows_by_day: dict[int, int] = {}
        for pos in order:
            day_id = int(ep_day_ids[pos])
            if rows_by_day.get(day_id, 0) >= a.max_rows_per_day:
                continue
            selected_positions.append(int(pos))
            rows_by_day[day_id] = rows_by_day.get(day_id, 0) + int(counts[pos])
        selected_positions.sort()
        selected_positions_arr = np.asarray(selected_positions, dtype=np.int64)
        selected_eids = unique_eids[selected_positions_arr]
        selected_counts = counts[selected_positions_arr]
        total = int(selected_counts.sum())
        print(
            f"[bc-train-mlx] --max-rows-per-day {a.max_rows_per_day}: "
            f"kept {len(selected_eids):,} episode(s) across "
            f"{len(rows_by_day)} day(s), {total:,} rows",
            flush=True,
        )
    elif a.max_rows and a.max_rows > 0:
        cum = np.cumsum(counts)
        cutoff = min(int(np.searchsorted(cum, a.max_rows) + 1), len(unique_eids))
        selected_eids = unique_eids[:cutoff]
        print(
            f"[bc-train-mlx] limited to {a.max_rows} rows (max_rows={a.max_rows}): "
            f"capped to {cutoff} episode(s), ~{int(cum[cutoff - 1]):,} rows",
            flush=True,
        )
    else:
        selected_eids = unique_eids

    # D.4 (episode-level val split): unlike the old position-based tail split
    # over one contiguous .npy array, the corpus is now N independent Parquet
    # files, so the split key is the episode set itself, seeded and
    # deterministic. Episodes never straddle files (one zip == one day == one
    # episode's home), so this keeps train/val fully episode-disjoint.
    rng = np.random.default_rng(a.seed)
    shuffled_eids = selected_eids[rng.permutation(len(selected_eids))]
    n_val_eps = max(1, int(round(len(shuffled_eids) * a.val_frac)))
    if n_val_eps >= len(shuffled_eids):
        n_val_eps = len(shuffled_eids) - 1
    if n_val_eps <= 0 or len(shuffled_eids) - n_val_eps <= 0:
        raise ValueError(
            "episode split produced an empty train or val set "
            f"({len(shuffled_eids)} episode(s) selected); add more training "
            "days or lower --val-frac"
        )
    val_episode_ids = shuffled_eids[:n_val_eps]
    train_episode_ids = shuffled_eids[n_val_eps:]
    val_filter = pads.field("episode_id").isin(val_episode_ids.tolist())
    train_filter = pads.field("episode_id").isin(train_episode_ids.tolist())
    _TBPTT_FILTER_CACHE[id(val_filter)] = np.asarray(
        val_episode_ids, dtype=np.int64
    )
    _TBPTT_FILTER_CACHE[id(train_filter)] = np.asarray(
        train_episode_ids, dtype=np.int64
    )
    print(
        f"[bc-train-mlx] episode split: {len(train_episode_ids):,} train / "
        f"{len(val_episode_ids):,} val episodes (val_frac={a.val_frac}, "
        f"seed={a.seed})",
        flush=True,
    )

    _use_tbptt = bool(a.tbptt_chunk > 0)

    # ---- validation split: streamed via a dedicated KV row_group cache ----
    # No pre-materialization. Scan pass 1 records per-val-row
    # (file_idx, row_group_idx, offset). The val loop reads rows on demand
    # via _val_row_group_cache (same hierarchical class as train, distinct
    # instance so the two zones do not compete). Per-lane extras (y,
    # is_attack, is_ko, opt_group) are accumulated during the val loop
    # from the fetched dicts and stitched back to row-order for the metrics
    # pass right after.
    val_meta_columns = ["episode_id", "side", "step_id"]
    (
        val_meta,
        _val_row_file_idx,
        _val_row_group_idx,
        _val_row_offset,
        _val_file_paths,
    ) = _scan_tbptt_locations(
        pa_dataset, val_filter, enc_shapes, int_keys
    )
    n_val = int(len(val_meta["episode_id"]))
    if n_val == 0:
        raise RuntimeError(
            "validation split produced no rows; adjust --val-frac or the "
            "selected training days"
        )
    val_cache_columns = list(keys) + ["y", "is_attack"]
    if a.dedup:
        val_cache_columns.append("opt_group")
    if aux_active:
        val_cache_columns.extend(_AUX_COLUMNS)
    _val_spill_dir = os.path.join(
        os.path.dirname(a.out) or ".", ".cache_spill", "val"
    )
    _val_row_group_cache = _ParquetRowGroupCache(
        file_paths=_val_file_paths,
        columns=val_cache_columns,
        shapes=enc_shapes,
        int_keys=int_keys,
        ssd_spill_dir=_val_spill_dir,
    )
    _val_cache_backend = (
        _val_row_group_cache,
        _val_row_file_idx,
        _val_row_group_idx,
        _val_row_offset,
    )
    # No pre-materialized arrays. Metrics arrays (y_val / vi_atk / vi_ko /
    # gv_np) are rebuilt from streamed batches inside the val loop and
    # projected back to row-order via the plan's row_permutation.
    val_np = None
    y_val = None
    aux_val_np = None
    gv_np = None
    vi_atk = None
    vi_ko = None

    # ---- training split ------------------------------------------------------
    # TBPTT path is streamed row_group-by-row_group via a two-pass loader:
    #  1. _scan_tbptt_locations reads ONLY (episode_id/side/step_id) plus each
    #     row's physical (file_idx, row_group_idx, offset) location.
    #  2. _ParquetRowGroupCache serves data columns on demand from a bounded
    #     LRU of decoded row_groups (peak ~2-3 row_groups resident at a time).
    # Non-TBPTT path is already streamed by pyarrow's batch iterator.
    train_columns = list(keys) + ["y"]
    if a.dedup:
        train_columns.append("opt_group")
    if aux_active:
        train_columns.extend(_AUX_COLUMNS)
    counts_by_eid = dict(zip(unique_eids.tolist(), counts.tolist()))
    if _use_tbptt:
        (
            train_meta,
            _tbptt_row_file_idx,
            _tbptt_row_group_idx,
            _tbptt_row_offset,
            _tbptt_file_paths,
        ) = _scan_tbptt_locations(
            pa_dataset, train_filter, enc_shapes, int_keys
        )
        n_train = int(len(train_meta["episode_id"]))
        _train_spill_dir = os.path.join(
            os.path.dirname(a.out) or ".", ".cache_spill", "train"
        )
        _tbptt_row_group_cache = _ParquetRowGroupCache(
            file_paths=_tbptt_file_paths,
            columns=train_columns,
            shapes=enc_shapes,
            int_keys=int_keys,
            ssd_spill_dir=_train_spill_dir,
        )
        # No preload phase. The cache grows on-demand as _load_temporal_batch
        # touches row_groups and only evicts under host-memory pressure --
        # see _ParquetRowGroupCache._touch. Warm-up cost is the first pass
        # over the dataset; every subsequent pass is fully disk-free unless
        # the OS actually starts to swap.
        train_np = None
        y_train = None
        aux_train_np = None
    else:
        n_train = int(sum(counts_by_eid[e] for e in train_episode_ids.tolist()))
        train_np = None  # streamed per epoch; never fully materialized
        y_train = None
        train_meta = None
        aux_train_np = None
        _tbptt_row_group_cache = None
        _tbptt_row_file_idx = None
        _tbptt_row_group_idx = None
        _tbptt_row_offset = None
    if n_train == 0:
        raise RuntimeError(
            "training split produced no rows; adjust --val-frac or the "
            "selected training days"
        )

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
        f"{' +accum' if a.accum_steps > 1 else ''}"
    )
    print(
        f"[bc-train-mlx] {tag} params={nparams:,} "
        f"N={n_train + n_val} train={n_train} val={n_val} "
        f"batch={a.batch} accum_steps={a.accum_steps}",
        flush=True,
    )

    # --- batch generator (C.1: FP16-native numeric features) ---
    # `chunk` is a plain dict[str, np.ndarray] with row-aligned keys `keys` +
    # "y" (+ "opt_group" iff --dedup) (+ _AUX_COLUMNS iff aux_active) --
    # exactly what the pyarrow streaming re-chunker (_stream_train_microbatches)
    # and the val-batch slicer below both produce, so the same conversion
    # serves training and validation.
    def _to_mx_batch(
        chunk: dict[str, np.ndarray],
    ) -> tuple[dict[str, mx.array], mx.array, dict[str, mx.array] | None]:
        ob = {
            k: mx.array(
                np.asarray(chunk[k]).astype(np.int32 if k in int_keys else np.float16)
            )
            for k in keys
        }
        if a.dedup:
            gb = mx.array(np.asarray(chunk["opt_group"]), dtype=mx.int32)
            canon = (gb == mx.arange(gb.shape[1])[None, :]).astype(mx.float16)
            ob["action_mask"] = ob["action_mask"] * canon
        if a.zero_wouldko:
            attr = np.asarray(ob["opt_attr"]).copy()
            attr[..., WK_LO:WK_HI] = 0.0
            ob["opt_attr"] = mx.array(attr, dtype=mx.float16)
        yb = mx.array(np.asarray(chunk["y"]).astype(np.int32))
        aux_targets = None
        if aux_active and "aux_valid" in chunk:
            aux_targets = {
                c: mx.array(np.asarray(chunk[c]).astype(np.float32))
                for c in _AUX_COLUMNS
            }
        return ob, yb, aux_targets

    def _val_batches(batch_size: int):
        """Stream the validation split as fixed-size microbatches directly
        from parquet, in the same manner as ``_stream_train_microbatches``.
        Yields ``(mx_batch, raw_chunk)`` so the non-TBPTT val loop can
        accumulate ``y`` / ``is_attack`` / ``opt_attr`` (for is_ko) /
        ``opt_group`` from the same chunk without a second scan.
        """
        cols = list(keys) + ["y", "is_attack"]
        if a.dedup:
            cols.append("opt_group")
        if aux_active:
            cols.extend(_AUX_COLUMNS)
        for chunk in _stream_train_microbatches(
            pa_dataset,
            cols,
            val_filter,
            enc_shapes,
            int_keys,
            batch_size,
            seed_key=[int(a.seed), 0, 0, 0],
        ):
            yield _to_mx_batch(chunk), chunk

    # --- loss + grad function ---
    # aux_targets is always the 3rd positional arg (None when aux is
    # inactive) so both branches share one call signature; the inactive
    # branch is byte-identical to the pre-aux legacy path (logits_value,
    # no aux forward at all).
    if aux_active:
        def _loss_fn(model, ob, yb, aux_targets):
            logits, _value, _mem, aux_dict = model.logits_value_aux(ob)
            ce = _cross_entropy_sum(logits, yb)
            aux_mean = _aux_loss(aux_dict, aux_targets, aux_weights)
            # Rescale mean -> sum form (see _batched_sequential_tbptt_loss's
            # matching comment) so a single division by n_examples at the
            # optimizer-step boundary yields mean(CE) + mean(aux).
            total = ce + aux_mean * yb.shape[0]
            return total, aux_mean
    else:
        def _loss_fn(model, ob, yb, aux_targets):
            return _cross_entropy_sum(model.logits_value(ob)[0], yb)

    grad_fn = mx.value_and_grad(_loss_fn, argnums=0)

    # F.3: TBPTT loss + grad function (accepts memory, returns memory_out)
    def _tbptt_loss_fn(
        model,
        lane_observations,
        lane_labels,
        decision_lengths,
        lane_memory_in,
        lane_aux_targets,
    ):
        return _batched_sequential_tbptt_loss(
            model,
            lane_observations,
            lane_labels,
            decision_lengths,
            lane_memory_in,
            lane_aux_targets=lane_aux_targets,
            aux_weights=aux_weights if aux_active else None,
            return_aux=aux_active,
        )

    _tbptt_grad_fn = mx.value_and_grad(_tbptt_loss_fn, argnums=0)

    def tbptt_loss_and_grad(
        model,
        lane_observations,
        lane_labels,
        decision_lengths,
        lane_memory_in,
        lane_aux_targets=None,
    ):
        """Forward with memory, cross-entropy loss, backward through model params only."""
        value, grads = _tbptt_grad_fn(
            model,
            lane_observations,
            lane_labels,
            decision_lengths,
            lane_memory_in,
            lane_aux_targets,
        )
        if aux_active:
            loss, memory_out, aux_dict_all, aux_targets_all = value
            aux_mean = _aux_loss(aux_dict_all, aux_targets_all, aux_weights)
            return loss, grads, memory_out, aux_mean
        loss, memory_out = value
        return loss, grads, memory_out, None

    # --- exact work plan and optimizer-step schedule (C.4 / F.3) ---
    # `_use_tbptt` was already resolved above (before deciding whether to
    # stream or fully materialize the training split); train_meta/val_meta
    # are the plain-dict "episode_id"/"side"/"step_id" arrays materialized
    # earlier, one entry per row, 0-indexed within their own split.
    _tbptt_groups: list[np.ndarray] = []
    _tbptt_decision_groups: list[list[np.ndarray]] = []
    _tbptt_plan: list[list[_TBPTTChunk]] = []
    _val_tbptt_plan: list[list[_TBPTTChunk]] = []
    if _use_tbptt:
        _tbptt_groups = _build_tbptt_groups(train_meta, n_train)
        _tbptt_decision_groups = _build_tbptt_decision_groups(train_meta, n_train)
        _tbptt_plan = _build_tbptt_plan(
            _tbptt_decision_groups,
            chunk_size=a.tbptt_chunk,
            row_budget=a.batch,
        )
        validation_decision_groups = _build_tbptt_decision_groups(val_meta, n_val)
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
        microbatches_per_epoch = _ceil_div(n_train, a.batch)
        progress_mode = f"streamed shuffled batches (batch<={a.batch:,})"

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
        payload = {
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
                "data_days": [row["day_date"] for row in datasets],
                "data_paths": dataset_paths,
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
            "dataset_build_fingerprint": dataset_build_fingerprint,
            "phase_id": a.phase_id,
            "epoch": epoch,
            "gstep": gstep,
            "val_acc": val_acc,
            "best_val_acc": best,
            "seed": a.seed,
            "dataset_days": [row["day_date"] for row in datasets],
            "dataset_paths": dataset_paths,
            "accum_steps": a.accum_steps,
            "microbatches_per_epoch": microbatches_per_epoch,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "scheduler_phase_step": scheduler_phase_step,
            "scheduler_total_steps": scheduler_total_steps,
            "scheduler_contract": scheduler_contract,
            "scheduler_state": a.scheduler_state,
        }
        return payload

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
        """Clip gradients in MLX graph (no float() calls). Returns
        ``(clipped_grads, grad_norm_pre_clip)`` -- caller decides whether to
        use the norm (currently only tensorboard emits it)."""
        flat = [g.reshape(-1) for _, g in nn.utils.tree_flatten(grads) if g is not None]
        if not flat:
            return grads, 0.0
        gn = mx.sqrt(sum(mx.sum(g**2) for g in flat))
        if max_norm <= 0:
            mx.eval(gn)
            return grads, float(gn)
        scale = mx.where(gn > max_norm, max_norm / mx.maximum(gn, 1e-6), 1.0)
        grads = nn.utils.tree_map(lambda g: (g * scale) if g is not None else g, grads)
        mx.eval(grads, gn)
        return grads, float(gn)

    # --- train step with gradient accumulation (C.2) ---
    def train_step_accum(ob: dict, yb: mx.array, aux_targets: dict | None):
        """Forward + backward for one microbatch using an FP32 loss sum."""
        result, grads = grad_fn(model, ob, yb, aux_targets)
        if aux_active:
            loss, aux_mean = result
            mx.eval(loss, aux_mean)
            aux_val = float(aux_mean)
        else:
            loss = result
            mx.eval(loss)
            aux_val = 0.0
        loss_val = float(loss)
        return loss_val, aux_val, _to_fp32_grads(grads)

    # Exposed to the training loop so per-step tensorboard scalars can log
    # the un-clipped gradient magnitude every optimizer step, without a
    # second graph traversal.
    _last_step_metrics: dict = {}

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
        grads, grad_norm = clip_grads(grads, a.max_grad_norm)
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
        _last_step_metrics["grad_norm"] = grad_norm
        _last_step_metrics["lr"] = float(optimizer.learning_rate)
        _last_step_metrics["scheduler_phase_step"] = int(scheduler_phase_step)
        _last_step_metrics["n_examples"] = int(n_examples)

    # Compile stable shuffled-batch forward/backward; clipping stays outside.
    if compile_active:
        from functools import partial

        _state = [model.state, optimizer.state]

        @partial(mx.compile, inputs=_state, outputs=_state)
        def compiled_step(ob, yb, aux_targets):
            """Compiled forward + backward; clipping runs after accumulation."""
            result, grads = grad_fn(model, ob, yb, aux_targets)
            return result, grads

        print("[bc-train-mlx] compiled train_step with state capture", flush=True)

    # Streaming I/O overlap is intrinsic to the pyarrow batch iterator
    # (pyarrow.dataset.Scanner reads ahead internally). There is no separate
    # slab/thread/queue prefetch step; the hierarchical row-group cache
    # (see _ParquetRowGroupCache) owns cross-batch retention.

    # ---- F.3: TBPTT batch generator ----
    def _load_temporal_batch(
        temporal_batch: list[_TBPTTChunk],
        arrays: dict | None,
        labels: np.ndarray | None,
        *,
        label_base: int = 0,
        aux_arrays: dict[str, np.ndarray] | None = None,
        source: str = "materialized",
        cache_backend: tuple | None = None,
        emit_fetched: bool = False,
    ):
        """Materialize one pre-planned temporal batch with lane-local row order.

        Two backing modes are supported:

        * ``source="materialized"`` (in-RAM caller-owned arrays for the
          non-TBPTT train path via ``_stream_train_microbatches``):
          ``arrays``, ``labels`` and ``aux_arrays`` are sliced with the
          chunk's row indices.
        * ``source="tbptt-cache"`` (streaming row_group cache): rows are
          fetched on demand from ``cache_backend`` -- a 4-tuple
          ``(cache, file_idx, rg_idx, offset)``. When ``cache_backend`` is
          None the train backend from the enclosing scope is used.

        Returned tuple is 5 elements by default:
        ``(lane_observations, lane_labels, lane_decision_lengths,
        lane_rows, lane_aux_targets)``. With ``emit_fetched=True`` (cache
        mode only) a 6th element ``lane_fetched`` is appended -- the raw
        per-lane fetched dicts. The val loop uses it to accumulate y,
        is_attack, is_ko and opt_group without a second cache read.
        """
        lane_observations: list[dict[str, mx.array]] = []
        lane_labels: list[mx.array] = []
        lane_decision_lengths: list[list[int]] = []
        lane_rows: list[np.ndarray] = []
        lane_aux_targets: list[dict[str, mx.array]] | None = (
            []
            if aux_arrays is not None or source == "tbptt-cache" and aux_active
            else None
        )
        lane_fetched: list[dict[str, np.ndarray]] | None = (
            [] if (source == "tbptt-cache" and emit_fetched) else None
        )
        if source == "tbptt-cache":
            if cache_backend is not None:
                cache_obj, cb_file_idx, cb_rg_idx, cb_offset = cache_backend
            else:
                cache_obj = _tbptt_row_group_cache
                cb_file_idx = _tbptt_row_file_idx
                cb_rg_idx = _tbptt_row_group_idx
                cb_offset = _tbptt_row_offset
        for chunk in temporal_batch:
            chunk_arr = np.concatenate(chunk.decisions)
            lane_rows.append(chunk_arr)
            if source == "tbptt-cache":
                fetched = cache_obj.read_rows(
                    chunk_arr,
                    cb_file_idx,
                    cb_rg_idx,
                    cb_offset,
                )
                # dedup label relabel is per-chunk (same as the .npy loader);
                # this is exact because the remap is row-local.
                _apply_dedup_relabel(fetched)
                lane_observations.append(
                    {
                        key: mx.array(
                            fetched[key].astype(
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
                lane_labels.append(mx.array(fetched["y"].astype(np.int32)))
                if aux_active:
                    lane_aux_targets.append(
                        {
                            c: mx.array(fetched[c].astype(np.float32))
                            for c in _AUX_COLUMNS
                        }
                    )
                lane_decision_lengths.append(
                    [len(decision_rows) for decision_rows in chunk.decisions]
                )
                if lane_fetched is not None:
                    lane_fetched.append(fetched)
                continue
            # materialized path
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
                mx.array(labels[label_base + chunk_arr].astype(np.int32))
            )
            if aux_arrays is not None:
                lane_aux_targets.append(
                    {
                        c: mx.array(
                            np.asarray(aux_arrays[c][chunk_arr]).astype(np.float32)
                        )
                        for c in _AUX_COLUMNS
                    }
                )
            lane_decision_lengths.append(
                [len(decision_rows) for decision_rows in chunk.decisions]
            )
        if emit_fetched:
            return (
                lane_observations,
                lane_labels,
                lane_decision_lengths,
                lane_rows,
                lane_aux_targets,
                lane_fetched,
            )
        return (
            lane_observations,
            lane_labels,
            lane_decision_lengths,
            lane_rows,
            lane_aux_targets,
        )

    def _tbptt_batches():
        """Yield the exact packed temporal batches used by the work plan,
        with one-step lookahead: while MLX is running forward+backward on
        step N the pool submits ``_load_temporal_batch`` for step N+1, so
        the row_group decode (pyarrow C++ → releases the GIL) overlaps with
        GPU compute. Cache capacity is 6 row_groups so the prefetched entry
        does not evict what step N is still reading.
        """
        def _load_one(temporal_batch):
            return _load_temporal_batch(
                temporal_batch,
                arrays=None,
                labels=None,
                aux_arrays=None,
                source="tbptt-cache",
            )

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tbptt-prefetch") as pool:
            if not _tbptt_plan:
                return
            current_future = pool.submit(_load_one, _tbptt_plan[0])
            for i, temporal_batch in enumerate(_tbptt_plan):
                # Kick off the next load before waiting on the current one.
                next_future = (
                    pool.submit(_load_one, _tbptt_plan[i + 1])
                    if i + 1 < len(_tbptt_plan)
                    else None
                )
                lane_observations, lane_labels, lane_decision_lengths, _, lane_aux_targets = (
                    current_future.result()
                )
                yield (
                    temporal_batch,
                    lane_observations,
                    lane_labels,
                    lane_decision_lengths,
                    lane_aux_targets,
                )
                current_future = next_future

    # ---- tensorboard writer -------------------------------------------------
    # Emits scalars into a per-run directory under ``runs/``. In a second
    # shell the user runs ``uv run tensorboard --logdir runs`` and opens
    # the local web UI while training is still writing. The writer is
    # created lazily so a missing tensorboard package (e.g. slimmed Kaggle
    # image) does not break training.
    _tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        _tb_run_name = (
            os.path.splitext(os.path.basename(a.out))[0] + "_" + str(int(time.time()))
        )
        _tb_run_dir = os.path.join("runs", _tb_run_name)
        os.makedirs(_tb_run_dir, exist_ok=True)
        _tb_writer = SummaryWriter(log_dir=_tb_run_dir)
        print(
            f"[tb] tensorboard logdir: {_tb_run_dir} "
            "(open with: uv run tensorboard --logdir runs)",
            flush=True,
        )
    except ImportError:
        print("[tb] tensorboard not installed, metrics logging disabled", flush=True)

    # ---- training loop ----
    _running_loss: float = 0.0
    _running_aux_loss: float = 0.0
    _running_n: int = 0
    _compile_pending: bool = compile_active
    _micro_count: int = 0
    _accum_grads = None
    _accum_examples: int = 0
    _accum_loss_sum: float = 0.0
    _accum_aux_loss_sum: float = 0.0
    _tbptt_memories: dict[int, mx.array] = {}
    _tb_last_step_ts = time.perf_counter()

    def _tb_log_step(loss_val: float, aux_val: float, micro_n: int):
        """One per-optimizer-step scalar bundle written to tensorboard.

        Writes the loss (running-mean numerator over accumulation window
        divided by n_examples so the y-axis is comparable across accum_steps),
        aux loss, gradient norm (pre-clip), current LR, memory footprint,
        cache hit-rate to date, and the wall-clock step time. Keyed by
        ``gstep`` so multiple runs align.
        """
        nonlocal _tb_last_step_ts
        if _tb_writer is None:
            return
        now = time.perf_counter()
        step_time_ms = (now - _tb_last_step_ts) * 1000.0
        _tb_last_step_ts = now
        try:
            _tb_writer.add_scalar("train/loss", float(loss_val) / max(micro_n, 1), gstep)
            if aux_active:
                _tb_writer.add_scalar("train/aux_loss", float(aux_val), gstep)
            gm = _last_step_metrics.get("grad_norm")
            if gm is not None and np.isfinite(gm):
                _tb_writer.add_scalar("train/grad_norm", float(gm), gstep)
            _tb_writer.add_scalar(
                "train/lr", float(_last_step_metrics.get("lr", 0.0)), gstep
            )
            _tb_writer.add_scalar(
                "train/examples_per_step",
                int(_last_step_metrics.get("n_examples", micro_n)),
                gstep,
            )
            _tb_writer.add_scalar("train/step_time_ms", float(step_time_ms), gstep)
            _tb_writer.add_scalar(
                "train/scheduler_phase_step",
                int(_last_step_metrics.get("scheduler_phase_step", 0)),
                gstep,
            )
            try:
                _tb_writer.add_scalar(
                    "sys/mlx_peak_memory_gib",
                    float(mx.get_peak_memory()) / (1024**3),
                    gstep,
                )
            except Exception:
                pass
            try:
                import psutil as _psutil_step
                _tb_writer.add_scalar(
                    "sys/host_memory_percent",
                    float(_psutil_step.virtual_memory().percent),
                    gstep,
                )
            except Exception:
                pass
            for _tag, _c in (
                ("train", _tbptt_row_group_cache),
                ("val", _val_row_group_cache if _use_tbptt else None),
            ):
                if _c is None:
                    continue
                _cs = _c.report()
                _tot = _cs["hits"] + _cs["misses"]
                _rate = (_cs["hits"] / _tot * 100.0) if _tot else 0.0
                _tb_writer.add_scalar(f"cache_{_tag}/hit_rate_pct", float(_rate), gstep)
                _tb_writer.add_scalar(f"cache_{_tag}/resident_hot", int(_cs["resident_hot"]), gstep)
                _tb_writer.add_scalar(f"cache_{_tag}/resident_transient", int(_cs["resident_transient"]), gstep)
                _tb_writer.add_scalar(f"cache_{_tag}/promotions", int(_cs["promotions"]), gstep)
                _tb_writer.add_scalar(f"cache_{_tag}/evictions", int(_cs["evictions"]), gstep)
                _tb_writer.add_scalar(f"cache_{_tag}/ssd_hits", int(_cs["ssd_hits"]), gstep)
                _tb_writer.add_scalar(f"cache_{_tag}/ssd_spills", int(_cs["ssd_spills"]), gstep)
                _tb_writer.add_scalar(f"cache_{_tag}/ssd_resident", int(_cs["ssd_resident"]), gstep)
        except Exception:
            # Never let TB logging break training.
            pass

    validation_batches = (
        len(_val_tbptt_plan)
        if _use_tbptt
        else _ceil_div(n_val, a.val_batch_size)
    )
    train_t0 = time.time()

    for ep in range(start_epoch, run_end_epoch):
        mx.reset_peak_memory()
        model.train()
        ep_t0: float = time.time()
        ep_step: int = 0  # optimizer steps in this epoch
        ep_micro: int = 0  # microbatches processed in this epoch
        _running_loss = 0.0
        _running_aux_loss = 0.0
        _running_n = 0
        _micro_count = 0
        _accum_grads = None
        _accum_examples = 0
        _accum_loss_sum = 0.0
        _accum_aux_loss_sum = 0.0
        _tbptt_memories = {}  # reset every episode-side lane at epoch start
        local_epoch = ep - start_epoch + 1
        print(
            f"[bc-train-mlx] === epoch {ep + 1} (run {local_epoch}/{run_epochs}) ===",
            flush=True,
        )

        # One phase-aware task per epoch. The bar is reset for validation and
        # checkpointing, so 100% always means the displayed phase is complete.
        _progress_columns = [
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
        ]
        if aux_active:
            _progress_columns.append(TextColumn("Aux: {task.fields[aux]}"))
        _progress_columns += [
            TextColumn("LR: {task.fields[lr]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]
        _progress_bar = Progress(*_progress_columns)
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
            aux="--",
            lr=f"{float(optimizer.learning_rate):.2e}",
        )

        # Flatten all streamed pyarrow microbatches into a single generator so
        # gradient accumulation can accumulate across I/O-batch boundaries.
        def _all_batches():
            """Yield (ob, yb, aux_targets) streamed straight from the Parquet dataset."""
            for chunk in _stream_train_microbatches(
                pa_dataset,
                train_columns,
                train_filter,
                enc_shapes,
                int_keys,
                a.batch,
                seed_key=[a.seed, ep],
            ):
                yield _to_mx_batch(chunk)

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
                    lane_aux_targets,
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
                ob, yb, aux_targets = _batch_tuple  # standard path: 3-tuple
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
                    loss, grads, mem_out, aux_mean = tbptt_loss_and_grad(
                        model,
                        lane_observations,
                        lane_labels,
                        lane_decision_lengths,
                        lane_memory_in,
                        lane_aux_targets,
                    )
                    grads = _to_fp32_grads(grads)
                    if aux_active:
                        mx.eval(loss, grads, mem_out, aux_mean)
                        aux_val = float(aux_mean)
                    else:
                        mx.eval(loss, grads, mem_out)
                        aux_val = 0.0
                    loss_val = float(loss)
                    detached_memories = mx.stop_gradient(mem_out)
                    mx.eval(detached_memories)
                    for lane_index, chunk in enumerate(temporal_batch):
                        _tbptt_memories[chunk.group_index] = detached_memories[
                            lane_index : lane_index + 1
                        ]
                else:
                    loss_val, aux_val, grads = train_step_accum(ob, yb, aux_targets)
                if not np.isfinite(loss_val):
                    raise FloatingPointError(
                        f"non-finite training loss at epoch={ep + 1}, "
                        f"microbatch={ep_micro + 1}"
                    )
                if _accum_grads is None:
                    _accum_grads = grads
                    _accum_examples = micro_n
                    _accum_loss_sum = loss_val
                    _accum_aux_loss_sum = aux_val * micro_n
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
                    _accum_aux_loss_sum += aux_val * micro_n
                _micro_count += 1

                # Accumulate in FP32 for numerical stability
                if _micro_count % a.accum_steps == 0 or is_final_microbatch:
                    _cache_step_ctx = (
                        _tbptt_row_group_cache.in_opt_step()
                        if _tbptt_row_group_cache is not None
                        else nullcontext()
                    )
                    with _cache_step_ctx:
                        optimizer_step(_accum_grads, _accum_examples)
                    ep_step += 1
                    optimizer_updated = True
                    _running_loss += _accum_loss_sum
                    _running_aux_loss += _accum_aux_loss_sum
                    _running_n += _accum_examples
                    _tb_log_step(_accum_loss_sum, _accum_aux_loss_sum / max(_accum_examples, 1), _accum_examples)
                    _accum_grads = None
                    _accum_examples = 0
                    _accum_loss_sum = 0.0
                    _accum_aux_loss_sum = 0.0
            else:
                # No accumulation: single microbatch = full step
                if _use_tbptt:
                    loss, grads, mem_out, aux_mean = tbptt_loss_and_grad(
                        model,
                        lane_observations,
                        lane_labels,
                        lane_decision_lengths,
                        lane_memory_in,
                        lane_aux_targets,
                    )
                    grads = _to_fp32_grads(grads)
                    if aux_active:
                        mx.eval(loss, grads, mem_out, aux_mean)
                        aux_val = float(aux_mean)
                    else:
                        mx.eval(loss, grads, mem_out)
                        aux_val = 0.0
                    loss_val = float(loss)
                    detached_memories = mx.stop_gradient(mem_out)
                    mx.eval(detached_memories)
                    for lane_index, chunk in enumerate(temporal_batch):
                        _tbptt_memories[chunk.group_index] = detached_memories[
                            lane_index : lane_index + 1
                        ]
                elif compile_active:
                    result, grads = compiled_step(ob, yb, aux_targets)
                    grads = _to_fp32_grads(grads)
                    if aux_active:
                        loss, aux_mean = result
                        mx.eval(loss, grads, aux_mean)
                        aux_val = float(aux_mean)
                    else:
                        loss = result
                        mx.eval(loss, grads)
                        aux_val = 0.0
                    loss_val = float(loss)
                else:
                    result, grads = grad_fn(model, ob, yb, aux_targets)
                    grads = _to_fp32_grads(grads)
                    if aux_active:
                        loss, aux_mean = result
                        mx.eval(loss, grads, aux_mean)
                        aux_val = float(aux_mean)
                    else:
                        loss = result
                        mx.eval(loss, grads)
                        aux_val = 0.0
                    loss_val = float(loss)
                if not np.isfinite(loss_val):
                    raise FloatingPointError(
                        f"non-finite training loss at epoch={ep + 1}, "
                        f"microbatch={ep_micro + 1}"
                    )
                _cache_step_ctx = (
                    _tbptt_row_group_cache.in_opt_step()
                    if _tbptt_row_group_cache is not None
                    else nullcontext()
                )
                with _cache_step_ctx:
                    optimizer_step(grads, micro_n)
                ep_step += 1
                optimizer_updated = True
                _running_loss += loss_val
                _running_aux_loss += aux_val * micro_n
                _running_n += micro_n
                _tb_log_step(loss_val, aux_val, micro_n)

            ep_micro += 1
            if _compile_pending:
                print(f" done ({time.time() - _compile_t:.0f}s)", flush=True)
                _compile_pending = False

            # Include the current, not-yet-stepped accumulation window so the
            # displayed loss never drops to a misleading 0.0000 between
            # optimizer updates.
            display_loss_sum = _running_loss + _accum_loss_sum
            display_aux_loss_sum = _running_aux_loss + _accum_aux_loss_sum
            display_examples = _running_n + _accum_examples
            avg = display_loss_sum / max(display_examples, 1)
            aux_avg = display_aux_loss_sum / max(display_examples, 1)
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
                aux=f"{aux_avg:.4f}" if aux_active else "--",
                lr=f"{float(optimizer.learning_rate):.2e}",
            )

            if (
                a.log_interval > 0
                and optimizer_updated
                and ep_step % a.log_interval == 0
            ):
                aux_log = f" aux={aux_avg:.4f}" if aux_active else ""
                print(
                    f"[bc-train-mlx]   opt_step "
                    f"{ep_step:,}/{optimizer_steps_per_epoch:,} "
                    f"(run {gstep - run_start_gstep:,}/{run_optimizer_steps:,}) "
                    f"micro={ep_micro:,}/{microbatches_per_epoch:,} "
                    f"loss={avg:.4f}{aux_log} lr={float(optimizer.learning_rate):.2e} "
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
            aux="--",
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
            validation_y: list[np.ndarray] = []
            validation_is_attack: list[np.ndarray] = []
            validation_is_ko: list[np.ndarray] = []
            validation_opt_group: list[np.ndarray] = []
            aux_pred_val: dict[str, list[np.ndarray]] = defaultdict(list)
            aux_target_val: dict[str, list[np.ndarray]] = defaultdict(list)
            for val_batch, temporal_batch in enumerate(
                _val_tbptt_plan, start=1
            ):
                (
                    lane_observations,
                    lane_labels,
                    lane_decision_lengths,
                    lane_rows,
                    lane_aux_targets,
                    lane_fetched,
                ) = _load_temporal_batch(
                    temporal_batch,
                    arrays=None,
                    labels=None,
                    aux_arrays=None,
                    source="tbptt-cache",
                    cache_backend=_val_cache_backend,
                    emit_fetched=True,
                )
                lane_memory_in = [
                    (
                        None
                        if chunk.is_new_group
                        else validation_memories[chunk.group_index]
                    )
                    for chunk in temporal_batch
                ]
                tbptt_result = _batched_sequential_tbptt_loss(
                    model,
                    lane_observations,
                    lane_labels,
                    lane_decision_lengths,
                    lane_memory_in,
                    return_logits=True,
                    lane_aux_targets=lane_aux_targets,
                    aux_weights=aux_weights if aux_active else None,
                    return_aux=aux_active,
                )
                if aux_active:
                    _, memory_out, lane_logits, aux_dict_all, aux_targets_all = (
                        tbptt_result
                    )
                    mx.eval(memory_out, lane_logits, aux_dict_all, aux_targets_all)
                else:
                    _, memory_out, lane_logits = tbptt_result
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
                    fetched = lane_fetched[lane_index]
                    validation_y.append(fetched["y"].astype(np.int64))
                    validation_is_attack.append(
                        fetched["is_attack"].astype(bool)
                    )
                    validation_is_ko.append(
                        (fetched["opt_attr"][..., WK_LO] >= 0.5).any(axis=1)
                    )
                    if a.dedup:
                        validation_opt_group.append(
                            fetched["opt_group"].astype(np.int32)
                        )
                if aux_active:
                    for key, value in aux_dict_all.items():
                        aux_pred_val[key].append(np.asarray(value, dtype=np.float32))
                    for key, value in aux_targets_all.items():
                        aux_target_val[key].append(
                            np.asarray(value, dtype=np.float32)
                        )
                _progress_bar.update(
                    _progress_task,
                    completed=val_batch,
                    unit=f"temporal {val_batch:,}/{validation_batches:,}",
                )

            row_order = np.concatenate(validation_rows)
            row_permutation = np.argsort(row_order)
            if not np.array_equal(
                row_order[row_permutation], np.arange(n_val)
            ):
                raise RuntimeError(
                    "temporal validation did not cover every validation row exactly once"
                )
            lg_np = np.concatenate(validation_logits, axis=0)[row_permutation]
            yb_np = np.concatenate(validation_y)[row_permutation]
            vi_atk = np.concatenate(validation_is_attack)[row_permutation]
            vi_ko = np.concatenate(validation_is_ko)[row_permutation]
            gv_np = (
                np.concatenate(validation_opt_group, axis=0)[row_permutation]
                if a.dedup
                else None
            )
            aux_metrics = (
                _aux_metrics(
                    {k: np.concatenate(v) for k, v in aux_pred_val.items()},
                    {k: np.concatenate(v) for k, v in aux_target_val.items()},
                )
                if aux_active
                else None
            )
        else:
            logits_parts: list[np.ndarray] = []
            y_parts: list[np.ndarray] = []
            is_attack_parts: list[np.ndarray] = []
            is_ko_parts: list[np.ndarray] = []
            opt_group_parts: list[np.ndarray] = []
            aux_pred_val = defaultdict(list)
            aux_target_val = defaultdict(list)
            for val_batch, ((ob, _, aux_targets), raw_chunk) in enumerate(
                _val_batches(a.val_batch_size), start=1
            ):
                if aux_active:
                    lg, _, _, aux_dict = model.logits_value_aux(ob)
                    mx.eval(lg, aux_dict)
                    for key, value in aux_dict.items():
                        aux_pred_val[key].append(np.asarray(value, dtype=np.float32))
                    for key, value in aux_targets.items():
                        aux_target_val[key].append(
                            np.asarray(value, dtype=np.float32)
                        )
                else:
                    lg, _, _ = model.logits_value(ob)
                logits_parts.append(np.asarray(lg, dtype=np.float32))
                y_parts.append(raw_chunk["y"].astype(np.int64))
                is_attack_parts.append(raw_chunk["is_attack"].astype(bool))
                is_ko_parts.append(
                    (raw_chunk["opt_attr"][..., WK_LO] >= 0.5).any(axis=1)
                )
                if a.dedup:
                    opt_group_parts.append(raw_chunk["opt_group"].astype(np.int32))
                _progress_bar.update(
                    _progress_task,
                    completed=val_batch,
                    unit=f"batch {val_batch:,}/{validation_batches:,}",
                )
            lg_np = np.concatenate(logits_parts, axis=0)
            yb_np = np.concatenate(y_parts)
            vi_atk = np.concatenate(is_attack_parts)
            vi_ko = np.concatenate(is_ko_parts)
            gv_np = (
                np.concatenate(opt_group_parts, axis=0) if a.dedup else None
            )
            aux_metrics = (
                _aux_metrics(
                    {k: np.concatenate(v) for k, v in aux_pred_val.items()},
                    {k: np.concatenate(v) for k, v in aux_target_val.items()},
                )
                if aux_active
                else None
            )

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
            aux="--",
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
            yv_group = gv_np[np.arange(n_val), yb_np]
            eq = float((gv_np[np.arange(n_val), am_cat] == yv_group).mean())

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
            aux=(
                f"{_running_aux_loss / max(_running_n, 1):.4f}" if aux_active else "--"
            ),
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
        aux_metrics_log = ""
        if aux_active and aux_metrics is not None:
            aux_metrics_log = (
                f" aux_ko_bce={aux_metrics['aux_ko_bce']:.4f} "
                f"aux_prize_mse={aux_metrics['aux_prize_mse']:.4f} "
                f"aux_terminal_bce={aux_metrics['aux_terminal_bce']:.4f} "
                f"aux_return_mse={aux_metrics['aux_return_mse']:.4f}"
            )
        print(
            f"[bc-train-mlx] epoch {ep + 1} complete: "
            f"val_acc={acc:.4f} equiv={eq:.4f} top3={t3:.4f} "
            f"atk={atk:.4f} ko={ko:.4f} loss={vloss / max(tot, 1):.4f}"
            f"{aux_metrics_log} "
            f"t={_format_compact_duration(ep_time)} run_ETA={eta_str} "
            f"gstep={gstep:,}",
            flush=True,
        )
        if _tb_writer is not None:
            _tb_writer.add_scalar("val/acc", float(acc), ep + 1)
            _tb_writer.add_scalar("val/equiv", float(eq), ep + 1)
            _tb_writer.add_scalar("val/top3", float(t3), ep + 1)
            _tb_writer.add_scalar("val/atk", float(atk), ep + 1)
            _tb_writer.add_scalar("val/ko", float(ko), ep + 1)
            _tb_writer.add_scalar("val/loss", float(vloss / max(tot, 1)), ep + 1)
            _tb_writer.add_scalar("val/epoch_time_s", float(ep_time), ep + 1)
            _tb_writer.add_scalar(
                "train/running_loss", float(_running_loss / max(_running_n, 1)),
                ep + 1,
            )
            if aux_active:
                _tb_writer.add_scalar(
                    "train/running_aux_loss",
                    float(_running_aux_loss / max(_running_n, 1)),
                    ep + 1,
                )
            if aux_metrics is not None:
                for _k, _v in aux_metrics.items():
                    _tb_writer.add_scalar(f"aux/{_k}", float(_v), ep + 1)
            _tb_writer.add_scalar(
                "train/lr", float(optimizer.learning_rate), ep + 1
            )
            _tb_writer.add_scalar("train/gstep", int(gstep), ep + 1)
            _tb_writer.flush()

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
    def _print_cache_report(tag: str, cache):
        if cache is None:
            return
        s = cache.report()
        total = s["hits"] + s["misses"]
        hit_rate = (s["hits"] / total * 100.0) if total else 0.0
        print(
            f"[cache-{tag}] hits={s['hits']:,} misses={s['misses']:,} "
            f"hit_rate={hit_rate:.1f}% resident={s['resident_row_groups']} rg "
            f"(hot={s['resident_hot']}/transient={s['resident_transient']}) "
            f"promotions={s['promotions']:,} evictions={s['evictions']:,} "
            f"ssd_hits={s['ssd_hits']:,} ssd_spills={s['ssd_spills']:,} "
            f"ssd_resident={s['ssd_resident']} "
            f"decoded={s['bytes_loaded'] / (1024**2):.1f} MiB",
            flush=True,
        )

    _print_cache_report("train", _tbptt_row_group_cache)
    _print_cache_report("val", _val_row_group_cache if _use_tbptt else None)
    print(
        f"[bc-train-mlx] RESULT: best_val_acc={best:.4f} params={nparams:,} gstep={gstep}",
        flush=True,
    )
    if _tb_writer is not None:
        _tb_writer.add_scalar("summary/best_val_acc", float(best), 0)
        _tb_writer.add_scalar("summary/final_gstep", int(gstep), 0)
        # add_hparams turns this run into a directly-comparable row in
        # tensorboard's HParams tab (across the whole suite / seed sweep).
        # Only scalar-typed values are accepted.
        hparams = {
            "batch": int(a.batch),
            "tbptt_chunk": int(a.tbptt_chunk),
            "lr": float(a.lr),
            "warmup_steps": int(a.warmup_steps),
            "aux_ko_weight": float(a.aux_ko_weight),
            "aux_prize_weight": float(a.aux_prize_weight),
            "aux_terminal_weight": float(a.aux_terminal_weight),
            "aux_return_weight": float(a.aux_return_weight),
            "scratch_registers": int(a.scratch_registers),
            "d_model": int(a.d_model),
            "nhead": int(a.nhead),
            "nlayers": int(a.nlayers),
            "epochs": int(a.epochs),
            "seed": int(a.seed),
            "max_rows_per_day": int(a.max_rows_per_day or 0),
            "top_elo": int(a.top_elo or 0),
            "n_train_rows": int(n_train),
            "n_val_rows": int(n_val),
        }
        try:
            _tb_writer.add_hparams(hparams, {"hparam/best_val_acc": float(best)})
        except Exception:
            pass
        _tb_writer.flush()
        _tb_writer.close()


if __name__ == "__main__":
    main()
