"""Behavioral-cloning trainer — MLX version (Apple Silicon native).

Same architecture as bc_train.py but uses MLX instead of PyTorch.
Faster on M1/M2 via native Metal GPU (no MPS NaN bug).

FP16-native: numeric features stay float16 end-to-end.
Gradient accumulation: --accum-steps K accumulates K microbatches before update.

Usage:
  uv run tcg-train data/bc_data/bc_2026_07_21 \
      --d-model 128 --static --split-heads --epochs 8 --batch 128
"""

from rich.progress import TimeElapsedColumn
from datetime import timedelta

from rich.progress import ProgressColumn
from rich.text import Text
import argparse
import os
from pathlib import Path
import queue
import shutil
import threading
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder
from rl.encoder.enc_constants import OPT_WK
from rl.policy_mlx import build_token_net_mlx
from rl.lr_schedule import lr_at
from rl.train_config import load_config

WK_LO, WK_HI = OPT_WK, OPT_WK + 3


def _format_progress_duration(seconds: float | None) -> str:
    """Format a duration like Rich's elapsed-time column."""
    if seconds is None:
        return "-:--:--"
    return str(timedelta(seconds=max(0, int(seconds))))


def _format_compact_duration(seconds: float | None) -> str:
    """Format a duration for the human-readable ETA field."""
    if seconds is None:
        return "--"
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{seconds:02d}s"


class _GlobalElapsedColumn(ProgressColumn):
    """Elapsed time since this training invocation started."""

    def __init__(self, started_at: float):
        super().__init__()
        self.started_at = started_at

    def render(self, task) -> Text:
        elapsed = time.monotonic() - self.started_at
        return Text(_format_progress_duration(elapsed), style="yellow")


class _GlobalTimeRemainingColumn(ProgressColumn):
    """ETA for the complete run, independent of the current epoch task."""

    def __init__(self, started_at: float, progress_state: dict, total_steps: int):
        super().__init__()
        self.started_at = started_at
        self.progress_state = progress_state
        self.total_steps = max(1, total_steps)

    def render(self, task) -> Text:
        elapsed = max(0.0, time.monotonic() - self.started_at)
        completed = min(self.total_steps, self.progress_state["completed"])
        if completed <= 0:
            return Text("-:--:--", style="blue")
        remaining = elapsed * (self.total_steps - completed) / completed
        return Text(_format_progress_duration(remaining), style="blue")


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
    p.add_argument("--slab-rows", type=int, default=None)
    p.add_argument(
        "--max-rows", type=int, default=None, help="Max training rows (0=all)"
    )
    p.add_argument("--zero-wouldko", action="store_true", default=None)
    p.add_argument("--dedup", action="store_true", default=None)
    # Output
    p.add_argument("--out", default=None)
    p.add_argument("--resume", default=None)
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
        "tbptt_chunk": "tbptt_chunk",
        "compile": "compile",
        "prefetch": "prefetch",
        "log_interval": "log_interval",
        "val_frac": "val_frac",
        "slab_rows": "slab_rows",
        "max_rows": "max_rows",
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
    a.tbptt_chunk = a.tbptt_chunk if a.tbptt_chunk is not None else cfg.tbptt_chunk
    a.compile = a.compile if a.compile is not None else cfg.compile
    a.prefetch = a.prefetch if a.prefetch is not None else cfg.prefetch
    a.log_interval = a.log_interval if a.log_interval is not None else cfg.log_interval
    a.val_frac = a.val_frac if a.val_frac is not None else cfg.val_frac
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
    N = int(d["__labels__"].shape[0])

    # Apply max_rows limit (for smoke testing)
    if a.max_rows and a.max_rows > 0:
        N = min(N, a.max_rows)
        print(
            f"[bc-train-mlx] limited to {a.max_rows} rows (max_rows={a.max_rows})",
            flush=True,
        )

    labels = read_rows(d["__labels__"], 0, N)
    keys = [
        k
        for k in d
        if k not in ("__labels__", "__is_attack__", "__group__", "episode_meta")
    ]
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
    meta_path = os.path.join(a.data, "episode_meta.npy") if mmapped else None
    if meta_path and os.path.exists(meta_path):
        try:
            meta = np.load(meta_path)
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
        best = float(state.get("val_acc", 0.0))
        # Restore gstep for scheduler continuity
        gstep = int(state.get("gstep", 0))
        # Validate arch_config if present (backward compat with old checkpoints)
        saved_cfg = state.get("arch_config")
        if saved_cfg is not None:
            cur_cfg = model.get_config()
            mismatches = []
            for k, v in saved_cfg.items():
                if k in cur_cfg and cur_cfg[k] != v:
                    mismatches.append(f"{k}: saved={v} current={cur_cfg[k]}")
            if mismatches:
                print(
                    f"[bc-train-mlx] WARNING: arch_config mismatch: {', '.join(mismatches)}"
                )
                print("[bc-train-mlx] proceeding anyway — results may be invalid")
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

    # Optimizer
    optimizer = optim.Muon(learning_rate=a.lr)

    # Restore optimizer state if present in checkpoint (C.5)
    if a.resume:
        import pickle

        with open(a.resume, "rb") as f:
            state = pickle.load(f)
        saved_opt_state = state.get("optimizer")
        if saved_opt_state is not None:
            try:
                optimizer.state.update(saved_opt_state)
                print(f"[bc-train-mlx] restored optimizer state from checkpoint")
            except Exception as e:
                print(f"[bc-train-mlx] WARNING: could not restore optimizer state: {e}")

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
    # Keys kept in float32 despite being numeric (needed for masking comparisons):
    _FP32_KEYS = frozenset({"action_mask"})

    def batches(
        arrs: dict, grp: np.ndarray | None, base: int, order: np.ndarray, bs: int
    ):
        for i in range(0, len(order), bs):
            b = order[i : i + bs]
            ob = {
                k: mx.array(
                    np.asarray(arrs[k][b]).astype(
                        np.int32
                        if k in int_keys
                        else (np.float32 if k in _FP32_KEYS else np.float16)
                    )
                )
                for k in keys
            }
            if a.dedup:
                gb = mx.array(np.asarray(grp[b]), dtype=mx.int32)
                canon = (gb == mx.arange(gb.shape[1])[None, :]).astype(mx.float32)
                ob["action_mask"] = ob["action_mask"] * canon
            if a.zero_wouldko:
                attr = np.asarray(ob["opt_attr"]).copy()
                attr[..., WK_LO:WK_HI] = 0.0
                ob["opt_attr"] = mx.array(attr)
            yield ob, mx.array(y[base + b].astype(np.int32))

    # --- loss + grad function ---
    grad_fn = mx.value_and_grad(
        lambda model, ob, yb: nn.losses.cross_entropy(
            model.logits_value(ob)[0], yb
        ).mean(),  # logits only for loss
        argnums=0,
    )

    # F.3: TBPTT loss + grad function (accepts memory, returns memory_out)
    def _tbptt_loss_fn(model, ob, yb, memory_in):
        logits, _, memory_out = model.logits_value(ob, memory_in=memory_in)
        loss = nn.losses.cross_entropy(logits, yb).mean()
        return loss, memory_out

    _tbptt_grad_fn = mx.value_and_grad(_tbptt_loss_fn, argnums=0)

    def tbptt_loss_and_grad(model, ob, yb, memory_in):
        """Forward with memory, cross-entropy loss, backward through model params only."""
        (loss, memory_out), grads = _tbptt_grad_fn(model, ob, yb, memory_in)
        return loss, grads, memory_out

    if a.compile:
        from functools import partial

        _state = [model.state, optimizer.state]

        @partial(mx.compile, inputs=_state, outputs=_state)
        def compiled_step(ob, yb):
            """Compiled forward + backward + update (no clipping — done outside)."""
            loss, grads = grad_fn(model, ob, yb)
            optimizer.update(model, grads)
            return loss, grads

        print(f"[bc-train-mlx] compiled train_step with state capture", flush=True)

    # --- slab boundaries and total optimizer steps (C.4) ---
    slab_bounds = [(s, min(s + a.slab_rows, v0)) for s in range(0, v0, a.slab_rows)]
    steps_per_epoch = max(
        1, sum((e0 - s0 + a.batch - 1) // a.batch for s0, e0 in slab_bounds)
    )
    total_opt_steps = run_epochs * max(1, steps_per_epoch // a.accum_steps)
    warmup_steps = min(a.warmup_steps, max(1, total_opt_steps // 5))
    run_end_epoch = start_epoch + run_epochs
    print(
        f"[bc-train-mlx] global epoch range: {start_epoch + 1}-{run_end_epoch} "
        f"(local epochs={run_epochs})",
        flush=True,
    )

    def _checkpoint_payload(epoch: int, val_acc: float) -> dict:
        return {
            "model": model.parameters(),
            "optimizer": optimizer.state,
            "arch_config": model.get_config(),
            "epoch": epoch,
            "gstep": gstep,
            "val_acc": val_acc,
            "seed": a.seed,
            "dataset_path": a.data,
            "accum_steps": a.accum_steps,
        }

    def _save_checkpoint(path: str, epoch: int, val_acc: float) -> None:
        import pickle

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(_checkpoint_payload(epoch, val_acc), f)

    def _periodic_checkpoint_path(epoch: int) -> str:
        out_path = Path(a.out)
        return str(
            out_path.with_name(
                f"{out_path.stem}_epoch_{epoch + 1:04d}{out_path.suffix}"
            )
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
    def train_step_accum(
        ob: dict, yb: mx.array, micro_step: int, accum_steps: int
    ) -> float:
        """Forward + backward for one microbatch. Returns loss (Python float)."""
        loss, grads = grad_fn(model, ob, yb)
        mx.eval(loss)
        loss_val = float(loss)
        return loss_val, grads

    def optimizer_step(grads, accum_steps, n_examples):
        """Normalize accumulated grads, clip, update optimizer, advance gstep."""
        nonlocal gstep
        gstep += 1
        # Normalize by total examples (FP32 reduction)
        grads = nn.utils.tree_map(
            lambda g: (g / n_examples) if g is not None else g, grads
        )
        # Clip (C.3: graph-safe, no float())
        grads = clip_grads(grads, a.max_grad_norm)
        # LR schedule on optimizer step (C.4)
        if a.lr_schedule != "none":
            optimizer.learning_rate = lr_at(
                gstep,
                total_opt_steps,
                a.lr,
                a.lr_schedule,
                warmup_steps,
                a.lr_min_ratio,
            )
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        mx.eval(optimizer.state)

    # Compiled path (no clipping — clipping uses eager which has float() calls)
    if a.compile:
        from functools import partial

        _state = [model.state, optimizer.state]

        @partial(mx.compile, inputs=_state, outputs=_state)
        def compiled_step(ob, yb):
            """Compiled forward + backward (no clipping)."""
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
    def _tbptt_batches(chunk_size: int):
        """Yield (ob, yb, memory_in) groups in episode order for TBPTT.

        Loads episode_meta.npy (must have 'episode_id', 'side', 'new_episode').
        Groups rows by (episode_id, side), iterates in order, yields chunks.
        Memory is carried between chunks within the same (episode, side) group.
        """
        meta_path = os.path.join(a.data, "episode_meta.npy")
        if not mmapped or not os.path.exists(meta_path):
            print(
                f"[bc-train-mlx] F.3: no episode_meta.npy found, falling back to shuffled"
            )
            return

        meta = np.load(meta_path)
        ep_ids = np.asarray(meta["episode_id"][:v0])
        sides = np.asarray(meta["side"][:v0])

        # Group row indices by (episode_id, side)
        from collections import defaultdict

        groups: dict[tuple, list[int]] = defaultdict(list)
        for i in range(v0):
            key = (int(ep_ids[i]), int(sides[i]))
            groups[key].append(i)

        # Sort groups by episode_id then side for deterministic order
        sorted_keys = sorted(groups.keys())

        for key in sorted_keys:
            row_indices = groups[key]
            # Process this (episode, side) group in chunks
            # Yield (ob, yb, is_new_group) so trainer can reset memory at group boundary
            for ci in range(0, len(row_indices), chunk_size):
                chunk_rows = row_indices[ci : ci + chunk_size]
                chunk_arr = np.array(chunk_rows, dtype=np.int64)
                ob = {
                    k: mx.array(
                        np.asarray(obs_np[k][chunk_arr]).astype(
                            np.int32 if k in int_keys else np.float16
                        )
                    )
                    for k in keys
                }
                yb = mx.array(y[chunk_arr].astype(np.int32))
                yield ob, yb, (ci == 0)  # is_new_group=True for first chunk in group

    # ---- training loop ----
    _running_loss: float = 0.0
    _running_n: int = 0
    _compile_pending: bool = a.compile
    _micro_count: int = 0
    _accum_grads = None
    _accum_examples: int = 0
    _accum_loss_sum: float = 0.0
    _tbptt_memory = None  # F.3: persistent memory for TBPTT training

    # Rich progress bar
    try:
        from rich.progress import (
            Progress,
            SpinnerColumn,
            TextColumn,
            BarColumn,
            TaskProgressColumn,
            TimeRemainingColumn,
        )

        _rich_available = True
    except ImportError:
        _rich_available = False

    ep_t0: float = time.time()
    train_t0 = time.time()

    for ep in range(start_epoch, run_end_epoch):
        ep_step: int = 0  # optimizer steps in this epoch
        ep_micro: int = 0  # microbatches processed in this epoch
        _running_loss = 0.0
        _running_n = 0
        _micro_count = 0
        _accum_grads = None
        _accum_examples = 0
        _accum_loss_sum = 0.0
        _tbptt_memory = None  # F.3: reset memory at epoch start
        print(f"[bc-train-mlx] === epoch {ep + 1}/{a.epochs} ===", flush=True)

        # Rich progress bar for this epoch
        _progress_bar = None
        _progress_task = None
        if _rich_available:
            _progress_bar = Progress(
                SpinnerColumn(),
                TextColumn(
                    "[bold blue]Epoch {task.fields[epoch]} (run {task.fields[local_epoch]}/{task.fields[total_epochs]})"
                ),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("Loss: {task.fields[loss]}"),
                TextColumn("LR: {task.fields[lr]}"),
                TextColumn("ETA: {task.fields[eta]}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            _progress_bar.start()

        if _progress_bar is not None:
            _progress_task = _progress_bar.add_task(
                "training",
                epoch=ep + 1,
                local_epoch=ep - start_epoch + 1,
                total_epochs=run_epochs,
                total=total_opt_steps,
                loss="--",
                lr="--",
                eta="--",
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

        # F.3: TBPTT training path (opt-in, when --tbptt-chunk > 0 and episode_meta exists)
        _use_tbptt = (
            a.tbptt_chunk > 0
            and mmapped
            and os.path.exists(os.path.join(a.data, "episode_meta.npy"))
        )

        if _use_tbptt:
            print(
                f"[bc-train-mlx] F.3: TBPTT enabled (chunk={a.tbptt_chunk})", flush=True
            )

        _batch_iter = _tbptt_batches(a.tbptt_chunk) if _use_tbptt else _all_batches()

        for _batch_tuple in _batch_iter:
            if _use_tbptt:
                ob, yb, is_new_group = _batch_tuple  # TBPTT path: 3-tuple
                if is_new_group:
                    _tbptt_memory = None  # reset memory at episode/side boundary
            else:
                ob, yb = _batch_tuple  # standard path: 2-tuple

            micro_n = len(yb)

            # Forward + backward this microbatch
            if a.compile and _compile_pending:
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
                        model, ob, yb, _tbptt_memory
                    )
                    mx.eval(loss)
                    loss_val = float(loss)
                    # Carry only the last row's memory (last timestep) to next chunk
                    _tbptt_memory = mx.stop_gradient(mem_out[-1:])
                    # Additional stop_gradient at accumulation boundary
                    if _micro_count > 0 and _micro_count % a.accum_steps == 0:
                        _tbptt_memory = mx.stop_gradient(_tbptt_memory)
                else:
                    loss_val, grads = train_step_accum(
                        ob, yb, _micro_count, a.accum_steps
                    )
                if _accum_grads is None:
                    _accum_grads = grads
                    _accum_examples = micro_n
                    _accum_loss_sum = loss_val * micro_n
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
                    _accum_loss_sum += loss_val * micro_n
                _micro_count += 1

                # Accumulate in FP32 for numerical stability
                if _micro_count % a.accum_steps == 0:
                    optimizer_step(_accum_grads, a.accum_steps, _accum_examples)
                    ep_step += 1
                    _running_loss += _accum_loss_sum
                    _running_n += _accum_examples
                    _accum_grads = None
                    _accum_examples = 0
                    _accum_loss_sum = 0.0
            else:
                # No accumulation: single microbatch = full step
                if _use_tbptt:
                    loss, grads, mem_out = tbptt_loss_and_grad(
                        model, ob, yb, _tbptt_memory
                    )
                    mx.eval(loss)
                    loss_val = float(loss)
                    # Carry only the last row's memory (last timestep) to next chunk
                    _tbptt_memory = mx.stop_gradient(mem_out[-1:])
                elif a.compile and a.max_grad_norm <= 0:
                    loss, grads = compiled_step(ob, yb)
                    mx.eval(loss)
                    loss_val = float(loss)
                else:
                    loss, grads = grad_fn(model, ob, yb)
                    mx.eval(loss)
                    loss_val = float(loss)
                grads = clip_grads(grads, a.max_grad_norm)
                optimizer_step(grads, 1, micro_n)
                ep_step += 1
                _running_loss += loss_val * micro_n
                _running_n += micro_n

            ep_micro += 1
            if _compile_pending:
                print(f" done ({time.time() - _compile_t:.0f}s)", flush=True)
                _compile_pending = False

            avg = _running_loss / max(_running_n, 1)
            elapsed_s = time.time() - ep_t0
            steps_left = max(0, total_opt_steps - ep_step)
            eta_step = (elapsed_s / max(ep_step, 1)) * steps_left
            el_m, el_s = divmod(int(elapsed_s), 60)
            el_h, el_m = divmod(el_m, 60)
            el_str = f"{el_h}h{el_m:02d}m" if el_h else f"{el_m}m{el_s:02d}s"
            eta_m_s, eta_s_s = divmod(int(eta_step), 60)
            eta_h, eta_m_s = divmod(eta_m_s, 60)
            eta_str_s = (
                f"{eta_h}h{eta_m_s:02d}m" if eta_h else f"{eta_m_s}m{eta_s_s:02d}s"
            )
            # Rich progress bar update
            if _progress_bar is not None:
                _progress_bar.update(
                    _progress_task,
                    completed=ep_step,
                    loss=f"{avg:.4f}",
                    lr=f"{optimizer.learning_rate:.2e}",
                    elapsed=el_str,
                    eta=eta_str_s,
                )

            if a.log_interval > 0:
                if ep_step % a.log_interval == 0 and ep_step > 0:
                    print(
                        f"[bc-train-mlx]   opt_step {ep_step}/{total_opt_steps} "
                        f"micro={ep_micro} loss={avg:.4f} lr={optimizer.learning_rate:.2e} "
                        f"elapsed={el_str} ETA={eta_str_s}",
                        flush=True,
                    )

        # ---- validation ----
        model.eval()
        preds: list[np.ndarray] = []
        am_all: list[np.ndarray] = []
        vloss: float = 0.0
        tot: int = 0
        for ob, yb in batches(val_np, gv_np, v0, np.arange(nval), a.batch):
            lg, _, _ = model.logits_value(ob)
            lg_np = np.asarray(lg)
            yb_np = np.asarray(yb)
            # Proper cross-entropy: -(logit[label] - logsumexp(logits))
            logsumexp = np.logaddexp.reduce(lg_np, axis=1)
            ce = -(lg_np[np.arange(len(yb_np)), yb_np] - logsumexp)
            vloss += float(ce.mean()) * len(yb_np)
            tot += len(yb_np)
            top3 = np.argsort(-lg_np, axis=1)[:, :3]
            correct = np.argmax(lg_np, axis=1) == yb_np
            in_top3 = np.array([yb_np[i] in top3[i] for i in range(len(yb_np))])
            preds.append(
                np.stack([correct.astype(float), in_top3.astype(float)], axis=1)
            )
            am_all.append(np.argmax(lg_np, axis=1))

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

        # Flush any remaining accumulated gradients before validation
        if a.accum_steps > 1 and _accum_grads is not None and _accum_examples > 0:
            optimizer_step(_accum_grads, a.accum_steps, _accum_examples)
            ep_step += 1
            _running_loss += _accum_loss_sum
            _running_n += _accum_examples
            _accum_grads = None
            _accum_examples = 0
            _accum_loss_sum = 0.0

        # Complete checkpoint: save model, optimizer, arch_config, scheduler, seed (C.5)
        if acc > best:
            _save_checkpoint(a.out, ep, acc)
        best = max(best, acc)

        # Also retain periodic snapshots so an interrupted run can resume
        # close to its latest epoch even when validation accuracy did not
        # improve. The final epoch is always retained as well.
        local_epoch = ep - start_epoch + 1
        if local_epoch % a.checkpoint_every_epochs == 0 or ep == run_end_epoch - 1:
            periodic_path = _periodic_checkpoint_path(ep)
            _save_checkpoint(periodic_path, ep, acc)
            print(
                f"[bc-train-mlx] checkpoint saved: {periodic_path} "
                f"(epoch {ep + 1}, val_acc={acc:.4f})",
                flush=True,
            )

        ep_time: float = time.time() - ep_t0
        elapsed: float = time.time() - train_t0
        completed: int = ep - start_epoch + 1
        remaining: int = run_end_epoch - ep - 1
        eta_s: float = (elapsed / max(completed, 1)) * remaining
        eta_m, eta_s = divmod(int(eta_s), 60)
        eta_str: str = f"{eta_m}m{eta_s:02d}s" if eta_m else f"{eta_s}s"
        print(
            f"[bc-train-mlx] ep{ep} val_acc={acc:.4f} equiv={eq:.4f} top3={t3:.4f} "
            f"atk={atk:.4f} ko={ko:.4f} loss={vloss / max(tot, 1):.4f} "
            f"t={ep_time:.0f}s ETA={eta_str} gstep={gstep}",
            flush=True,
        )

        # Stop Rich progress bar for this epoch
        if _progress_bar is not None:
            _progress_bar.stop()

    # Save final best
    if a.out and os.path.exists(a.out):
        final_path = "model/bc_model/bc_best_mlx_final.pkl"
        shutil.copy2(a.out, final_path)
        print(f"[bc-train-mlx] best checkpoint copied to {final_path}", flush=True)
    print(
        f"[bc-train-mlx] RESULT: best_val_acc={best:.4f} params={nparams:,} gstep={gstep}",
        flush=True,
    )


if __name__ == "__main__":
    main()
