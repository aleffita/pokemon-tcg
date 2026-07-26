"""Diagnostic script to find the source of NaN in BC training.

Checks:
  1. NaN/Inf in dataset features
  2. Model forward produces finite outputs
  3. Loss computation is finite
  4. Gradient update doesn't explode

Usage:
  uv run python scripts/bc/debug_nan.py data/bc_data/bc_2026_07_16
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder
from rl.policy import build_token_net


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/bc_data/bc_2026_07_16"

    print("=== 1. CHECKING DATASET FOR NaN/Inf ===")
    d = {f[:-4]: np.load(os.path.join(data_dir, f), mmap_mode="r")
         for f in sorted(os.listdir(data_dir)) if f.endswith(".npy")}
    keys = [k for k in d if k not in ("__labels__", "__is_attack__", "__group__")]
    N = int(d["__labels__"].shape[0])
    print(f"  {N} rows, {len(keys)} keys")

    for k in keys:
        arr = d[k]
        # Check a sample of rows (first 10000) for NaN/Inf
        sample = np.asarray(arr[:min(10000, N)])
        nan_count = np.isnan(sample).sum()
        inf_count = np.isinf(sample).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"  PROBLEM: {k}: {nan_count} NaN, {inf_count} Inf, shape={arr.shape}, dtype={arr.dtype}")
        else:
            min_val = float(sample.min())
            max_val = float(sample.max())
            print(f"  OK: {k}: shape={arr.shape}, range=[{min_val:.4f}, {max_val:.4f}]")

    print()
    print("=== 2. CHECKING MODEL FORWARD ===")
    ct = get_card_table()
    enc = TokenEncoder(ct)
    int_keys = set(enc.int_keys)

    cfg = {"d_model": 128, "nhead": 4, "nlayers": 3, "static": True, "split_heads": True}
    net = build_token_net(ct, cfg)
    nparams = sum(p.numel() for p in net.parameters())
    print(f"  Model: {nparams:,} params")

    # Load a small batch from the dataset
    labels = np.asarray(d["__labels__"][:64])
    y = torch.as_tensor(labels, dtype=torch.long)
    ob = {k: torch.as_tensor(np.asarray(d[k][:64]),
                              dtype=(torch.long if k in int_keys else torch.float32))
          for k in keys}

    print(f"  Batch: {len(labels)} rows")

    # Check for NaN in input tensors
    for k, v in ob.items():
        if torch.isnan(v).any():
            print(f"  PROBLEM: NaN in input tensor '{k}'")
        if torch.isinf(v).any():
            print(f"  PROBLEM: Inf in input tensor '{k}'")

    net.eval()
    with torch.no_grad():
        logits, value = net.logits_value(ob)
        print(f"  logits: shape={logits.shape}, nan={torch.isnan(logits).any().item()}, "
              f"inf={torch.isinf(logits).any().item()}")
        print(f"  value:  shape={value.shape}, nan={torch.isnan(value).any().item()}, "
              f"inf={torch.isinf(value).any().item()}")
        if not torch.isnan(logits).any():
            print(f"  logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")

    print()
    print("=== 3. CHECKING LOSS ===")
    lossf = nn.CrossEntropyLoss()
    net.train()
    logits, value = net.logits_value(ob)
    loss = lossf(logits.float(), y)
    print(f"  loss = {loss.item()}, nan={torch.isnan(loss).item()}, inf={torch.isinf(loss).item()}")

    print()
    print("=== 4. CHECKING GRADIENT UPDATE ===")
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    opt.zero_grad()
    loss.backward()

    # Check gradients
    nan_grads = 0
    inf_grads = 0
    big_grads = 0
    for name, p in net.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                nan_grads += 1
                print(f"  PROBLEM: NaN grad in {name}")
            if torch.isinf(p.grad).any():
                inf_grads += 1
                print(f"  PROBLEM: Inf grad in {name}")
            if p.grad.abs().max() > 100:
                big_grads += 1
                print(f"  WARNING: large grad in {name}: max={p.grad.abs().max().item():.2f}")

    if nan_grads == 0 and inf_grads == 0:
        print(f"  All gradients finite (big_grads={big_grads})")

    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()

    # Check weights after update
    nan_weights = 0
    for name, p in net.named_parameters():
        if torch.isnan(p).any():
            nan_weights += 1
            print(f"  PROBLEM: NaN weight after update in {name}")

    if nan_weights == 0:
        print(f"  All weights finite after update")

    print()
    print("=== 5. FORWARD AFTER UPDATE ===")
    net.eval()
    with torch.no_grad():
        logits2, value2 = net.logits_value(ob)
        print(f"  logits: nan={torch.isnan(logits2).any().item()}, inf={torch.isinf(logits2).any().item()}")
        if not torch.isnan(logits2).any():
            print(f"  logits range: [{logits2.min().item():.4f}, {logits2.max().item():.4f}]")

    print()
    print("=== SUMMARY ===")
    if nan_grads > 0 or nan_weights > 0:
        print("  ROOT CAUSE: NaN in gradients or weights -> numerical instability in model")
    elif torch.isnan(logits).any():
        print("  ROOT CAUSE: NaN in model output -> check model architecture")
    elif torch.isnan(loss).item():
        print("  ROOT CAUSE: NaN loss -> check labels and action_mask")
    else:
        print("  Everything looks fine for 1 step! NaN may appear after many steps.")
        print("  Try: lower LR, check data for edge cases in later batches.")


if __name__ == "__main__":
    main()
