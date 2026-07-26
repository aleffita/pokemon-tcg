"""Behavioral-cloning trainer — adapted for M1 8GB / single-device.

Trains a TokenTransformer on encoded BC data (from build_bc_dataset.py / build_bc_from_zips.py).
Reports val top-1 accuracy, top-3, attack-subset, and effect-equivalence-aware accuracy.

Usage:
  python scripts/bc/bc_train.py data/bc_data/bc_2026_07_16 --d-model 128 --epochs 10
  python scripts/bc/bc_train.py data/bc_data/bc_2026_07_16 --static --split-heads
  python scripts/bc/bc_train.py data/bc_data/bc_2026_07_16 --dedup --out bc_d128.pt
  python scripts/bc/bc_train.py data/bc_data/bc_2026_07_16 --resume bc_d128.pt --epochs 20
"""
import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder
from rl.encoder.enc_constants import OPT_WK
from rl.policy import build_token_net
from rl.lr_schedule import lr_at

WK_LO, WK_HI = OPT_WK, OPT_WK + 3   # opt_attr cols for would_ko trio


def read_rows(arr, start, stop):
    """Rows [start:stop) — buffered sequential read for memmap, slice for in-RAM."""
    if not isinstance(arr, np.memmap):
        return np.asarray(arr[start:stop])
    rowel = int(np.prod(arr.shape[1:], dtype=np.int64))
    flat = np.fromfile(arr.filename, dtype=arr.dtype, count=(stop - start) * rowel,
                       offset=arr.offset + start * rowel * arr.dtype.itemsize)
    return flat.reshape((stop - start,) + arr.shape[1:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", help=".npz file or DIRECTORY of per-key .npy files (memmap mode)")
    p.add_argument("--d-model", type=int, default=128,
                   help="Transformer hidden dimension (128=~1M, 256=~5M, 512=~20M params)")
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--nlayers", type=int, default=3)
    p.add_argument("--ff", type=int, default=256, help="FFN width (default 4*d_model)")
    p.add_argument("--static", action="store_true", help="Use static card features")
    p.add_argument("--split-heads", action="store_true",
                   help="Dedicated value/submit tokens (match production RL recipe)")
    p.add_argument("--structured", action="store_true", help="Verb-conditioned action head")
    p.add_argument("--zero-wouldko", action="store_true", help="Zero the would_ko trio")
    p.add_argument("--dedup", action="store_true",
                   help="Collapse interchangeable options via __group__")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=128, help="Batch size (128 safe for M1 8GB)")
    p.add_argument("--lr", type=float, default=5e-4, help="Peak LR")
    p.add_argument("--lr-schedule", choices=["cosine", "linear", "none"], default="cosine")
    p.add_argument("--warmup-steps", type=int, default=1500)
    p.add_argument("--lr-min-ratio", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="Grad clipping (0=off)")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--slab-rows", type=int, default=65536,
                    help="Memmap slab size (65k rows ~2.5GB; lower for M1 8GB)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="model/checkpoint/bc_best.pt",
                   help="Save best checkpoint to this path")
    p.add_argument("--resume", default=None, help="Resume training from a saved checkpoint")
    a = p.parse_args()

    # Ensure output directories exist
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs("model/bc_model", exist_ok=True)

    # Device: CUDA > CPU (MPS skipped — PyTorch MPS has NaN bugs with Transformers)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[bc-train] device={device}", flush=True)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    ct = get_card_table()
    enc = TokenEncoder(ct)
    int_keys = set(enc.int_keys)

    # ---- data loading ----
    mmapped = os.path.isdir(a.data)
    print(f"[bc-train] loading data from {a.data} ...", end="", flush=True)
    if mmapped:
        d = {f[:-4]: np.load(os.path.join(a.data, f), mmap_mode="r")
             for f in sorted(os.listdir(a.data)) if f.endswith(".npy")}
    else:
        z = np.load(a.data)
        d = {k: z[k] for k in z.files}
    N = int(d["__labels__"].shape[0])
    n_keys = len([k for k in d if k not in ("__labels__", "__is_attack__", "__group__")])
    print(f" {N:,} rows, {n_keys} feature keys ({'memmap' if mmapped else 'in-RAM'})", flush=True)
    labels = read_rows(d["__labels__"], 0, N)
    has_group = "__group__" in d
    if a.dedup and not has_group:
        raise SystemExit("--dedup needs a dataset built with __group__")
    keys = [k for k in d if k not in ("__labels__", "__is_attack__", "__group__")]
    obs_np = {k: d[k] for k in keys}
    group_np = d.get("__group__")

    # ---- dedup arm: relabel expert picks to canonical ----
    if a.dedup:
        for _s in range(0, N, 1 << 20):
            _e = min(N, _s + (1 << 20))
            _g = read_rows(group_np, _s, _e)
            labels[_s:_e] = _g[np.arange(_e - _s), labels[_s:_e]]
    y = torch.as_tensor(labels, dtype=torch.long)
    is_attack = (torch.as_tensor(read_rows(d["__is_attack__"], 0, N), dtype=torch.long).bool()
                 if "__is_attack__" in d else torch.zeros(N, dtype=torch.bool))

    # ---- game-level holdout: tail = whole held-out games (no leakage) ----
    nval = max(1, int(N * a.val_frac))
    v0 = N - nval
    idx = np.arange(N)
    vi, ti = idx[v0:], idx[:v0]

    # ---- val cache: one sequential read, reused every epoch ----
    val_np = {k: read_rows(obs_np[k], v0, N) for k in keys}
    gv_np = read_rows(group_np, v0, N) if group_np is not None else None
    oa_v = val_np["opt_attr"]
    vi_atk = is_attack[torch.as_tensor(vi)]
    vi_ko = torch.as_tensor((oa_v[..., WK_LO] >= 0.5).any(axis=1))
    wk_present = float(np.abs(oa_v[..., WK_LO:WK_HI]).sum())
    print(f"[bc-train] val cache loaded: {nval} rows, wk_present={wk_present:.1f}", flush=True)

    # ---- model ----
    cfg = {"arch": "transformer2", "d_model": a.d_model,
           "nhead": a.nhead, "nlayers": a.nlayers, "ff": a.ff,
           "static": a.static, "structured": a.structured,
           "split_heads": a.split_heads}
    net = build_token_net(ct, cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    nparams = sum(pp.numel() for pp in net.parameters())

    # ---- checkpoint resume ----
    start_epoch = 0
    best = 0.0
    if a.resume:
        ckpt = torch.load(a.resume, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["net"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best = ckpt.get("bc_val_acc", 0.0)
        print(f"[bc-train] resumed from {a.resume} (epoch {start_epoch}, val_acc={best:.4f})", flush=True)

    tag = (f"d{a.d_model}L{a.nlayers}h{a.nhead}"
           f"{' +static' if a.static else ''}"
           f"{' +split' if a.split_heads else ''}"
           f"{' +struct' if a.structured else ''}"
           f"{' +NOWK' if a.zero_wouldko else ' +WK'}"
           f"{' +DEDUP' if a.dedup else ' +RAW'}")
    print(f"[bc-train] {tag} params={nparams:,} N={N} train={len(ti)} val={len(vi)}", flush=True)

    # ---- batch generator ----
    def batches(arrs, grp, base, order, bs):
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            ob = {k: torch.as_tensor(np.asarray(arrs[k][b]),
                                     dtype=(torch.long if k in int_keys else torch.float32),
                                     device=device) for k in keys}
            if a.dedup:
                gb = torch.as_tensor(np.asarray(grp[b]), dtype=torch.long, device=device)
                canon = (gb == torch.arange(gb.shape[1], device=device)[None, :]).float()
                ob["action_mask"] = ob["action_mask"] * canon
            if a.zero_wouldko:
                ob["opt_attr"][..., WK_LO:WK_HI] = 0.0
            yield ob, y[torch.as_tensor(base + b)].to(device)

    # ---- slab boundaries for memmap ----
    slab_bounds = [(s, min(s + a.slab_rows, v0)) for s in range(0, v0, a.slab_rows)]

    # ---- total steps for LR schedule ----
    if mmapped:
        steps_per_ep = max(1, sum((e0 - s0 + a.batch - 1) // a.batch for s0, e0 in slab_bounds))
    else:
        steps_per_ep = (len(ti) + a.batch - 1) // a.batch
    total_steps = max(1, a.epochs * steps_per_ep)
    warmup_steps = min(a.warmup_steps, max(1, total_steps // 5))
    slab_info = f"{len(slab_bounds)} slabs ({a.slab_rows:,} rows/slab), " if mmapped else ""
    print(f"[bc-train] {slab_info}{steps_per_ep:,} steps/epoch, "
          f"{total_steps:,} total_steps, warmup={warmup_steps}, "
          f"lr={a.lr} {a.lr_schedule}", flush=True)

    lossf = nn.CrossEntropyLoss()
    gstep = 0
    _running_loss = 0.0
    _running_n = 0
    gv = torch.as_tensor(gv_np, dtype=torch.long) if gv_np is not None else None

    def _gpu_mem():
        if device.type == "cuda":
            a_gb = torch.cuda.memory_allocated(device) / 1024**3
            r_gb = torch.cuda.memory_reserved(device) / 1024**3
            return f" gpu={a_gb:.1f}/{r_gb:.1f}GB"
        return ""

    def train_step(ob, yb):
        nonlocal gstep, _running_loss, _running_n
        gstep += 1
        if a.lr_schedule != "none":
            opt.param_groups[0]["lr"] = lr_at(gstep, total_steps, a.lr,
                                              a.lr_schedule, warmup_steps, a.lr_min_ratio)
        logits, _ = net.logits_value(ob)
        loss = lossf(logits.float(), yb)
        loss_val = loss.detach().item()  # capture BEFORE backward (MPS float() bug)
        opt.zero_grad()
        loss.backward()
        if a.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(net.parameters(), a.max_grad_norm)
        opt.step()
        _running_loss += loss_val * len(yb)
        _running_n += len(yb)
        if gstep % 100 == 0:
            avg = _running_loss / max(_running_n, 1)
            print(f"[bc-train]   step {gstep}/{total_steps} "
                  f"loss={avg:.4f} lr={opt.param_groups[0]['lr']:.2e}{_gpu_mem()}", flush=True)

    # ---- training loop ----
    train_t0 = time.time()
    for ep in range(start_epoch, a.epochs):
        net.train()
        ep_t0 = time.time()
        print(f"[bc-train] === epoch {ep + 1}/{a.epochs} ===", flush=True)
        if mmapped:
            perm_slabs = np.random.default_rng([a.seed, ep]).permutation(len(slab_bounds))
            for slab_i, si in enumerate(perm_slabs):
                s0, e0 = slab_bounds[si]
                slab_t = time.time()
                print(f"[bc-train]   slab {slab_i + 1}/{len(slab_bounds)} "
                      f"(rows {s0:,}-{e0:,}) loading...", end="", flush=True)
                sd = {k: read_rows(obs_np[k], s0, e0) for k in keys}
                sg = read_rows(group_np, s0, e0) if (a.dedup and group_np is not None) else None
                load_t = time.time() - slab_t
                print(f" {load_t:.1f}s", flush=True)
                _running_loss = 0.0; _running_n = 0  # reset per-slab
                perm = np.random.default_rng([a.seed, ep, s0]).permutation(e0 - s0)
                for ob, yb in batches(sd, sg, s0, perm, a.batch):
                    train_step(ob, yb)
                slab_loss = _running_loss / max(_running_n, 1)
                print(f"[bc-train]   slab {slab_i + 1}/{len(slab_bounds)} done "
                      f"train_loss={slab_loss:.4f} ({time.time() - slab_t:.0f}s total, "
                      f"{load_t:.0f}s load){_gpu_mem()}", flush=True)
                del sd, sg
        else:
            g = torch.Generator().manual_seed(a.seed * 100_000 + ep)
            order = ti[torch.randperm(len(ti), generator=g).numpy()]
            for ob, yb in batches(obs_np, group_np, 0, order, a.batch):
                train_step(ob, yb)

        # ---- validation ----
        net.eval()
        preds, am_all = [], []
        vloss, tot = 0.0, 0
        with torch.no_grad():
            for ob, yb in batches(val_np, gv_np, v0, np.arange(nval), a.batch):
                lg, _ = net.logits_value(ob)
                lg = lg.float()
                vloss += lossf(lg, yb).item() * len(yb)
                tot += len(yb)
                top3 = (lg.topk(3, 1).indices == yb[:, None]).any(1)
                preds.append(torch.stack([(lg.argmax(1) == yb), top3], 1))
                am_all.append(lg.argmax(1))
        pr = torch.cat(preds)
        c1, c3 = pr[:, 0], pr[:, 1]
        acc = float(c1.float().mean())
        t3 = float(c3.float().mean())
        atk = float(c1[vi_atk].float().mean()) if int(vi_atk.sum()) else 0.0
        ko = float(c1[vi_ko].float().mean()) if int(vi_ko.sum()) else 0.0

        # effect-equivalence-aware accuracy
        eq = acc
        if gv is not None:
            am_cat = torch.cat(am_all)
            yv_group = gv[torch.arange(len(vi)), y[torch.as_tensor(vi)]]
            eq = float((gv[torch.arange(len(vi)), am_cat] == yv_group).float().mean())

        if a.out and acc > best:
            torch.save({"net": net.state_dict(), "net_config": cfg,
                        "opt": opt.state_dict(), "gstep": gstep,
                        "bc_val_acc": acc, "epoch": ep}, a.out)
        best = max(best, acc)
        ep_time = time.time() - ep_t0
        # ETA: elapsed / completed_epochs * remaining_epochs
        elapsed = time.time() - train_t0
        completed = ep - start_epoch + 1
        remaining = a.epochs - ep - 1
        eta_s = (elapsed / max(completed, 1)) * remaining
        eta_m, eta_s = divmod(int(eta_s), 60)
        eta_str = f"{eta_m}m{eta_s:02d}s" if eta_m else f"{eta_s}s"
        print(f"[bc-train] ep{ep} val_acc={acc:.4f} equiv={eq:.4f} top3={t3:.4f} "
              f"atk={atk:.4f} ko={ko:.4f} loss={vloss / max(tot, 1):.4f} "
              f"t={ep_time:.0f}s ETA={eta_str}", flush=True)

    # Save final best to bc_model/ if we have a checkpoint
    if a.out and os.path.exists(a.out):
        final_path = "model/bc_model/bc_best_final.pt"
        shutil.copy2(a.out, final_path)
        print(f"[bc-train] best checkpoint copied to {final_path}", flush=True)
    print(f"[bc-train] RESULT: best_val_acc={best:.4f} params={nparams:,}", flush=True)


if __name__ == "__main__":
    main()
