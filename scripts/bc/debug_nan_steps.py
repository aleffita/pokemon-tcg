"""Run500 training steps and report exactly when NaN first appears.

Usage:
  PYTHONPATH=. python3 -u scripts/bc/debug_nan_steps.py data/bc_data/bc_2026_07_16
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder
from rl.lr_schedule import lr_at


def read_rows(arr, start, stop):
    if not isinstance(arr, np.memmap):
        return np.asarray(arr[start:stop])
    rowel = int(np.prod(arr.shape[1:], dtype=np.int64))
    flat = np.fromfile(arr.filename, dtype=arr.dtype, count=(stop - start) * rowel,
                       offset=arr.offset + start * rowel * arr.dtype.itemsize)
    return flat.reshape((stop - start,) + arr.shape[1:])


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/bc_data/bc_2026_07_16"
    max_steps = 2000
    batch_size = 64

    d = {f[:-4]: np.load(os.path.join(data_dir, f), mmap_mode="r")
         for f in sorted(os.listdir(data_dir)) if f.endswith(".npy")}
    keys = [k for k in d if k not in ("__labels__", "__is_attack__", "__group__")]
    N = int(d["__labels__"].shape[0])
    labels = read_rows(d["__labels__"], 0, N)
    y = torch.as_tensor(labels, dtype=torch.long)
    int_keys = set(TokenEncoder(get_card_table()).int_keys)

    ct = get_card_table()
    from rl.policy import build_token_net
    cfg = {"d_model": 128, "nhead": 4, "nlayers": 3, "static": True, "split_heads": True}
    # Force MPS to match real training
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS", flush=True)
    else:
        device = torch.device("cpu")
        print("Using CPU", flush=True)
    net = build_token_net(ct, cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    lossf = nn.CrossEntropyLoss()

    # Use all rows (slab-like sequential processing)
    n_train = N
    order = np.arange(n_train)
    np.random.shuffle(order)  # shuffle once, then process sequentially like slabs

    print(f"Running {max_steps} steps, batch={batch_size}, train_rows={n_train}", flush=True)
    print(f"Processing sequentially (like slab streaming)", flush=True)

    cursor = 0
    for step in range(1, max_steps + 1):
        # Sequential batch (like slab streaming)
        if cursor + batch_size > n_train:
            np.random.shuffle(order)
            cursor = 0
        idx = order[cursor:cursor + batch_size]
        cursor += batch_size
        ob = {k: torch.as_tensor(np.asarray(d[k][idx]),
                                  dtype=(torch.long if k in int_keys else torch.float32),
                                  device=device)
              for k in keys}
        yb = y[idx].to(device)

        # Forward
        net.train()
        logits, value = net.logits_value(ob)

        # Check logits
        if torch.isnan(logits).any():
            print(f"  step {step}: NaN in LOGITS (before loss)", flush=True)
            # Which rows have NaN?
            nan_rows = torch.isnan(logits).any(dim=1).nonzero().squeeze()
            print(f"  NaN in rows: {nan_rows.tolist()[:10]}", flush=True)
            # Check if it's all-masked rows
            am = ob["action_mask"]
            for r in nan_rows[:3]:
                legal = (am[r] > 0.5).sum().item()
                print(f"  row {r.item()}: {legal} legal options", flush=True)
            break

        loss = lossf(logits.float(), yb)

        if torch.isnan(loss).item():
            print(f"  step {step}: NaN in LOSS (logits were finite)", flush=True)
            break

        # Backward + step
        opt.zero_grad()
        loss.backward()

        # Check gradients
        nan_grad = False
        for name, p in net.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                print(f"  step {step}: NaN grad in '{name}'", flush=True)
                nan_grad = True
        if nan_grad:
            break

        # Clip + step
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        # Check weights
        nan_w = False
        for name, p in net.named_parameters():
            if torch.isnan(p).any():
                print(f"  step {step}: NaN weight in '{name}' after update", flush=True)
                nan_w = True
        if nan_w:
            break

        if step % 50 == 0:
            grad_norm = sum(p.grad.norm().item() ** 2 for p in net.parameters()
                           if p.grad is not None) ** 0.5
            max_logit = logits[logits > -1e8].max().item() if (logits > -1e8).any() else 0
            print(f"  step {step}: loss={loss.item():.4f} grad_norm={grad_norm:.4f} "
                  f"max_logit={max_logit:.4f} lr={5e-4:.6f}", flush=True)
            if device.type == "mps":
                torch.mps.empty_cache()

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
