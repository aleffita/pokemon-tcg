"""Behavioral-cloning trainer — MLX version (Apple Silicon native).

Same architecture as bc_train.py but uses MLX instead of PyTorch.
Faster on M1/M2 via native Metal GPU (no MPS NaN bug).

Usage:
  PYTHONPATH=. python3 scripts/bc/bc_train_mlx.py data/bc_data/bc_2026_07_21 \
      --d-model 128 --static --split-heads --epochs 8 --batch 128
"""
import argparse
import os
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

WK_LO, WK_HI = OPT_WK, OPT_WK + 3


def read_rows(arr: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Rows [start:stop) — buffered sequential read for memmap, slice for in-RAM."""
    if not isinstance(arr, np.memmap):
        return np.asarray(arr[start:stop])
    rowel = int(np.prod(arr.shape[1:], dtype=np.int64))
    flat = np.fromfile(arr.filename, dtype=arr.dtype, count=(stop - start) * rowel,
                       offset=arr.offset + start * rowel * arr.dtype.itemsize)
    return flat.reshape((stop - start,) + arr.shape[1:])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data", help=".npz or DIRECTORY of per-key .npy files")
    p.add_argument("--d-model", type=int, default=128,
                   help="Transformer hidden dim (128=~1M, 256=~3.5M params)")
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--nlayers", type=int, default=3)
    p.add_argument("--ff", type=int, default=256, help="FFN width (default 4*d_model)")
    p.add_argument("--static", action="store_true")
    p.add_argument("--split-heads", action="store_true")
    p.add_argument("--structured", action="store_true")
    p.add_argument("--zero-wouldko", action="store_true")
    p.add_argument("--dedup", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--compile", action="store_true", help="mx.compile the loss function")
    p.add_argument("--prefetch", action="store_true",
                   help="Prefetch next slab on CPU while GPU trains (stream overlap)")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr-schedule", choices=["cosine", "linear", "none"], default="cosine")
    p.add_argument("--warmup-steps", type=int, default=1500)
    p.add_argument("--lr-min-ratio", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--slab-rows", type=int, default=262144)
    p.add_argument("--log-interval", type=int, default=100,
                   help="Print training stats every N steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="model/checkpoint/bc_best_mlx.pkl")
    p.add_argument("--resume", default=None)
    a = p.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs("model/bc_model", exist_ok=True)

    # MLX auto-detecta GPU (Metal) — sem device management!
    print(f"[bc-train-mlx] device={mx.default_device()}", flush=True)

    # --- data loading (igual ao PyTorch) ---
    mmapped = os.path.isdir(a.data)
    if mmapped:
        d = {f[:-4]: np.load(os.path.join(a.data, f), mmap_mode="r")
             for f in sorted(os.listdir(a.data)) if f.endswith(".npy")}
    else:
        z = np.load(a.data)
        d = {k: z[k] for k in z.files}
    N = int(d["__labels__"].shape[0])
    labels = read_rows(d["__labels__"], 0, N)
    keys = [k for k in d if k not in ("__labels__", "__is_attack__", "__group__")]
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
    is_attack = read_rows(d["__is_attack__"], 0, N).astype(bool) if "__is_attack__" in d else np.zeros(N, dtype=bool)

    # Val split (game-level holdout)
    nval = max(1, int(N * a.val_frac))
    v0 = N - nval
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
    cfg = {"arch": "transformer2", "d_model": a.d_model,
           "nhead": a.nhead, "nlayers": a.nlayers, "ff": a.ff,
           "static": a.static, "structured": a.structured,
           "split_heads": a.split_heads}
    model = build_token_net_mlx(ct, cfg)

    # Resume from checkpoint
    start_epoch = 0
    best = 0.0
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
                print(f"[bc-train-mlx] WARNING: arch_config mismatch: {', '.join(mismatches)}")
                print(f"[bc-train-mlx] proceeding anyway — results may be invalid")
            else:
                print(f"[bc-train-mlx] arch_config validated OK")
        else:
            print(f"[bc-train-mlx] WARNING: no arch_config in checkpoint (old format) — "
                  f"proceeding without validation")
        print(f"[bc-train-mlx] resumed from {a.resume} (epoch {start_epoch}, "
              f"val_acc={best:.4f}, gstep={gstep})")

    # Optimizer
    optimizer = optim.Adam(learning_rate=a.lr)

    nparams = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    tag = (f"d{a.d_model}L{a.nlayers}h{a.nhead}"
           f"{' +static' if a.static else ''}"
           f"{' +split' if a.split_heads else ''}"
           f"{' +struct' if a.structured else ''}"
           f"{' +compile' if a.compile else ''}"
           f"{' +prefetch' if a.prefetch else ''}")
    print(f"[bc-train-mlx] {tag} params={nparams:,} N={N} train={len(ti)} val={len(vi)} "
          f"batch={a.batch}", flush=True)

    # --- batch generator (igual ao PyTorch, mas converte pra mx.array) ---
    def batches(arrs: dict, grp: np.ndarray | None, base: int, order: np.ndarray, bs: int):
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            ob = {k: mx.array(np.asarray(arrs[k][b]).astype(
                    np.int32 if k in int_keys else np.float32))
                  for k in keys}
            if a.dedup:
                gb = mx.array(np.asarray(grp[b]), dtype=mx.int32)
                canon = (gb == mx.arange(gb.shape[1])[None, :]).astype(mx.float32)
                ob["action_mask"] = ob["action_mask"] * canon
            if a.zero_wouldko:
                ob["opt_attr"] = mx.array(np.asarray(ob["opt_attr"]))
                # Zero the would_ko columns
                attr = np.asarray(ob["opt_attr"]).copy()
                attr[..., WK_LO:WK_HI] = 0.0
                ob["opt_attr"] = mx.array(attr)
            yield ob, mx.array(y[base + b].astype(np.int32))

    # --- loss + grad function ---
    grad_fn = mx.value_and_grad(
        lambda model, ob, yb: nn.losses.cross_entropy(
            model.logits_value(ob)[0], yb).mean(),
        argnums=0
    )

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

    # --- slab boundaries ---
    slab_bounds = [(s, min(s + a.slab_rows, v0)) for s in range(0, v0, a.slab_rows)]
    total_steps = max(1, sum((e0 - s0 + a.batch - 1) // a.batch for s0, e0 in slab_bounds))
    warmup_steps = min(a.warmup_steps, max(1, total_steps // 5))

    gstep = 0
    train_t0 = time.time()

    # Non-compiled train_step (fallback)
    def train_step_eager(ob: dict, yb: mx.array) -> float:
        nonlocal gstep
        gstep += 1
        if a.lr_schedule != "none":
            optimizer.learning_rate = lr_at(gstep, total_steps, a.lr,
                                            a.lr_schedule, warmup_steps, a.lr_min_ratio)
        loss, grads = grad_fn(model, ob, yb)
        if a.max_grad_norm > 0:
            flat = [g.reshape(-1) for _, g in nn.utils.tree_flatten(grads) if g is not None]
            if flat:
                gn = float(mx.sqrt(mx.sum(mx.concatenate(flat) ** 2)))
                if gn > a.max_grad_norm:
                    s = a.max_grad_norm / max(gn, 1e-6)
                    grads = nn.utils.tree_map(lambda g: g * s if g is not None else g, grads)
        optimizer.update(model, grads)
        return float(loss)

    # Compiled path (no clipping — clipping uses eager which has float() calls)
    def train_step_compiled(ob: dict, yb: mx.array) -> float:
        nonlocal gstep
        gstep += 1
        if a.lr_schedule != "none":
            optimizer.learning_rate = lr_at(gstep, total_steps, a.lr,
                                            a.lr_schedule, warmup_steps, a.lr_min_ratio)
        loss, _ = compiled_step(ob, yb)
        return float(loss)

    # Compiled path: no clipping (compile can't eval arrays). Eager path: with clipping.
    if a.compile and a.max_grad_norm <= 0:
        train_step = train_step_compiled
    elif a.compile:
        print(f"[bc-train-mlx] NOTE: --compile with clipping uses eager (clipping needs float())",
              flush=True)
        train_step = train_step_eager
    else:
        train_step = train_step_eager

    # ---- prefetch helper (stream overlap: CPU loads next slab while GPU trains) ----
    def _slab_generator(slab_indices):
        """Yield (slab_i, si, sd, sg, perm) for each slab."""
        for slab_i, si in enumerate(slab_indices):
            s0, e0 = slab_bounds[si]
            sd = {k: read_rows(obs_np[k], s0, e0) for k in keys}
            sg = read_rows(group_np, s0, e0) if (a.dedup and group_np is not None) else None
            perm = np.random.default_rng([a.seed, ep, s0]).permutation(e0 - s0)
            yield slab_i, si, s0, e0, sd, sg, perm

    def _prefetch_slabs(slab_indices, q: queue.Queue):
        """Background thread: load slabs ahead of the GPU."""
        for item in _slab_generator(slab_indices):
            q.put(item)
        q.put(None)

    # ---- training loop ----
    _running_loss: float = 0.0
    _running_n: int = 0
    _compile_pending: bool = a.compile

    for ep in range(start_epoch, a.epochs):
        ep_t0: float = time.time()
        ep_step: int = 0  # per-epoch step counter (for display + ETA)
        _running_loss = 0.0
        _running_n = 0
        print(f"[bc-train-mlx] === epoch {ep + 1}/{a.epochs} ===", flush=True)

        if mmapped:
            perm_slabs = np.random.default_rng([a.seed, ep]).permutation(len(slab_bounds))

            # Prefetch mode: background thread loads next slab while GPU trains
            if a.prefetch:
                q: queue.Queue = queue.Queue(maxsize=1)
                t = threading.Thread(target=_prefetch_slabs, args=(perm_slabs, q), daemon=True)
                t.start()
                _slab_iter = iter(lambda: q.get(), None)
            else:
                _slab_iter = _slab_generator(perm_slabs)

            for slab_i, si, s0, e0, sd, sg, perm in _slab_iter:
                slab_t: float = time.time()
                load_t = 0.0  # already loaded in prefetch mode
                if not a.prefetch:
                    print(f"[bc-train-mlx]   slab {slab_i + 1}/{len(slab_bounds)} "
                          f"(rows {s0:,}-{e0:,}) loading...", end="", flush=True)
                    load_t = time.time() - slab_t
                    print(f" {load_t:.1f}s", flush=True)
                else:
                    print(f"[bc-train-mlx]   slab {slab_i + 1}/{len(slab_bounds)} "
                          f"(rows {s0:,}-{e0:,}) prefetched", flush=True)
                for ob, yb in batches(sd, sg, s0, perm, a.batch):
                    if _compile_pending:
                        print("[bc-train-mlx]   compiling (first call, may take several minutes)...",
                              end="", flush=True)
                        _compile_t = time.time()
                    loss_val: float = train_step(ob, yb)
                    ep_step += 1
                    if _compile_pending:
                        print(f" done ({time.time() - _compile_t:.0f}s)", flush=True)
                        _compile_pending = False
                    _running_loss += loss_val * len(yb)
                    _running_n += len(yb)
                    if ep_step % a.log_interval == 0:
                        avg: float = _running_loss / max(_running_n, 1)
                        elapsed_s: float = time.time() - ep_t0
                        steps_left: int = total_steps - ep_step
                        eta_step: float = (elapsed_s / max(ep_step, 1)) * steps_left
                        el_m, el_s = divmod(int(elapsed_s), 60)
                        el_h, el_m = divmod(el_m, 60)
                        el_str = f"{el_h}h{el_m:02d}m" if el_h else f"{el_m}m{el_s:02d}s"
                        eta_m_s, eta_s_s = divmod(int(eta_step), 60)
                        eta_h, eta_m_s = divmod(eta_m_s, 60)
                        eta_str_s: str = f"{eta_h}h{eta_m_s:02d}m" if eta_h else f"{eta_m_s}m{eta_s_s:02d}s"
                        print(f"[bc-train-mlx]   step {ep_step}/{total_steps} "
                              f"loss={avg:.4f} lr={optimizer.learning_rate:.2e} "
                              f"elapsed={el_str} ETA={eta_str_s}", flush=True)
                del sd, sg
                slab_loss: float = _running_loss / max(_running_n, 1)
                print(f"[bc-train-mlx]   slab {slab_i + 1}/{len(slab_bounds)} done "
                      f"train_loss={slab_loss:.4f} ({time.time() - ep_t0:.0f}s total, "
                      f"{load_t:.1f}s load)", flush=True)
        else:
            g = np.random.default_rng([a.seed, ep])
            order = g.permutation(len(ti))
            for ob, yb in batches(obs_np, group_np, 0, order, a.batch):
                loss_val = train_step(ob, yb)
                ep_step += 1
                _running_loss += loss_val * len(yb)
                _running_n += len(yb)
                if ep_step % a.log_interval == 0:
                    avg = _running_loss / max(_running_n, 1)
                    elapsed_s = time.time() - ep_t0
                    steps_left = total_steps - ep_step
                    eta_step = (elapsed_s / max(ep_step, 1)) * steps_left
                    el_m, el_s = divmod(int(elapsed_s), 60)
                    el_h, el_m = divmod(el_m, 60)
                    el_str = f"{el_h}h{el_m:02d}m" if el_h else f"{el_m}m{el_s:02d}s"
                    eta_m_s, eta_s_s = divmod(int(eta_step), 60)
                    eta_h, eta_m_s = divmod(eta_m_s, 60)
                    eta_str_s = f"{eta_h}h{eta_m_s:02d}m" if eta_h else f"{eta_m_s}m{eta_s_s:02d}s"
                    print(f"[bc-train-mlx]   step {ep_step}/{total_steps} "
                          f"loss={avg:.4f} lr={optimizer.learning_rate:.2e} "
                          f"elapsed={el_str} ETA={eta_str_s}", flush=True)

        # ---- validation ----
        model.eval()
        preds: list[np.ndarray] = []
        am_all: list[np.ndarray] = []
        vloss: float = 0.0
        tot: int = 0
        for ob, yb in batches(val_np, gv_np, v0, np.arange(nval), a.batch):
            lg, _ = model.logits_value(ob)
            lg_np = np.asarray(lg)
            yb_np = np.asarray(yb)
            # Cross-entropy loss manually
            log_probs = np.log(np.clip(lg_np[np.arange(len(yb_np)), yb_np], 1e-8, 1.0))
            vloss += float(-log_probs.mean()) * len(yb_np)
            tot += len(yb_np)
            top3 = np.argsort(-lg_np, axis=1)[:, :3]
            correct = (np.argmax(lg_np, axis=1) == yb_np)
            in_top3 = np.array([yb_np[i] in top3[i] for i in range(len(yb_np))])
            preds.append(np.stack([correct.astype(float), in_top3.astype(float)], axis=1))
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

        if acc > best:
            import pickle
            with open(a.out, "wb") as f:
                pickle.dump({
                    "model": model.parameters(),
                    "arch_config": model.get_config(),
                    "epoch": ep,
                    "gstep": gstep,
                    "val_acc": acc,
                    "seed": a.seed,
                    "dataset_path": a.data,
                }, f)
        best = max(best, acc)

        ep_time: float = time.time() - ep_t0
        elapsed: float = time.time() - train_t0
        completed: int = ep - start_epoch + 1
        remaining: int = a.epochs - ep - 1
        eta_s: float = (elapsed / max(completed, 1)) * remaining
        eta_m, eta_s = divmod(int(eta_s), 60)
        eta_str: str = f"{eta_m}m{eta_s:02d}s" if eta_m else f"{eta_s}s"
        print(f"[bc-train-mlx] ep{ep} val_acc={acc:.4f} equiv={eq:.4f} top3={t3:.4f} "
              f"atk={atk:.4f} ko={ko:.4f} loss={vloss / max(tot, 1):.4f} "
              f"t={ep_time:.0f}s ETA={eta_str}", flush=True)

    # Save final best
    if a.out and os.path.exists(a.out):
        final_path = "model/bc_model/bc_best_mlx_final.pkl"
        shutil.copy2(a.out, final_path)
        print(f"[bc-train-mlx] best checkpoint copied to {final_path}", flush=True)
    print(f"[bc-train-mlx] RESULT: best_val_acc={best:.4f} params={nparams:,}", flush=True)


if __name__ == "__main__":
    main()