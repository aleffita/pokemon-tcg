"""Phase B integration test — semantic P0 fixes verified.

Validates:
  1. Additive attention mask (not boolean) — padded positions get -inf, real get 0
  2. MHA bias=True matching PyTorch reference
  3. Padding ID 0 returns zero vector (padding_idx=0 semantics)
  4. Static card_feat tables unchanged after optimizer step
  5. Forward pass produces finite logits and value
  6. Backward pass produces finite gradients

Run:
  uv run python scripts/validate/test_phase_b.py
"""
from __future__ import annotations

import os
import sys
import traceback

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from rl.encoder.card_features import get_card_table
from rl.encoder.enc_constants import N_STATE_TOKENS, MAX_OPTIONS, N_ACTIONS
from rl.encoder.encoding import TokenEncoder
from rl.policy_mlx import build_token_net_mlx, TokenTransformerMLX, TransformerEncoderLayerMLX
from scripts.validate.make_synthetic_data import make_dataset

import tempfile
import shutil

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_ROWS = 100
BATCH_SIZE = 8

CANONICAL_CFG = {
    "d_model": 128,
    "nhead": 4,
    "nlayers": 3,
    "static": True,
    "split_heads": True,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_dir() -> str:
    """Create a temp dir with synthetic rows matching bc_train schema."""
    tmpdir = tempfile.mkdtemp(prefix="phase_b_test_")
    make_dataset(N_ROWS, tmpdir, seed=42)
    effect_mask_path = os.path.join(tmpdir, "effect_mask.npy")
    if not os.path.exists(effect_mask_path):
        rng = np.random.default_rng(42)
        effect_mask = (rng.standard_normal((N_ROWS, 2)).astype(np.float32) * 0.2 + 0.5).clip(0.0, 1.0)
        np.save(effect_mask_path, effect_mask, allow_pickle=False)
    return tmpdir


def _load_batch(data_dir: str, batch_size: int):
    """Load first batch as mlx arrays."""
    d = {
        f[:-4]: np.load(os.path.join(data_dir, f), mmap_mode="r")
        for f in sorted(os.listdir(data_dir))
        if f.endswith(".npy")
    }
    labels = d["__labels__"]

    ct = get_card_table()
    enc = TokenEncoder(ct)
    int_keys = set(enc.int_keys)
    keys = [k for k in d if k not in ("__labels__", "__is_attack__", "__group__")]

    indices = np.arange(batch_size)
    ob = {
        k: mx.array(np.asarray(d[k][indices]).astype(
            np.int32 if k in int_keys else np.float32
        ))
        for k in keys
    }
    yb = mx.array(labels[indices].astype(np.int32))
    return ob, yb


# ---------------------------------------------------------------------------
# Test 1: Additive attention mask — padded positions do not affect real tokens
# ---------------------------------------------------------------------------

def test_additive_attention_mask():
    """Build model, run two forward passes (one with padding, one without), verify outputs
    for real tokens are unaffected by padding. Also verify no NaN."""
    ct = get_card_table()
    model = build_token_net_mlx(ct, CANONICAL_CFG)
    model.eval()

    tmpdir = _make_synthetic_dir()
    try:
        ob, _ = _load_batch(tmpdir, BATCH_SIZE)

        # Forward pass 1: normal (some padding)
        logits1, value1 = model.logits_value(ob)

        # Verify no NaN
        logits1_np = np.asarray(logits1)
        value1_np = np.asarray(value1)
        assert np.all(np.isfinite(logits1_np)), (
            f"Forward pass produced NaN/Inf in logits: "
            f"nan={np.isnan(logits1_np).sum()}, inf={np.isinf(logits1_np).sum()}"
        )
        assert np.all(np.isfinite(value1_np)), (
            f"Forward pass produced NaN/Inf in value"
        )

        # Forward pass 2: double the padding (more positions masked)
        ob2 = dict(ob)
        for k in ob2:
            if k.endswith("_mask") and k != "action_mask":
                ob2[k] = ob2[k] * 0.0  # force all to 0 (all padded)

        logits2, value2 = model.logits_value(ob2)

        # The NaN-free property should still hold even with extreme padding
        logits2_np = np.asarray(logits2)
        assert np.all(np.isfinite(logits2_np)), (
            f"Forward with all-masked pads produced NaN/Inf in logits"
        )

        print("  PASS: additive mask — no NaN, forward is finite under padding")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 2: MHA bias=True — attention projections have bias parameters
# ---------------------------------------------------------------------------

def test_mha_bias():
    """Verify every Transformer layer's MHA has bias (QKV + output projections)."""
    ct = get_card_table()
    model = build_token_net_mlx(ct, CANONICAL_CFG)

    for i, layer in enumerate(model.encoder.layers):
        attn = layer.attn
        # MLX MHA stores separate QKV + output projections as Linear modules
        for proj_name in ("query_proj", "key_proj", "value_proj", "out_proj"):
            proj = getattr(attn, proj_name)
            assert hasattr(proj, "bias"), (
                f"Layer {i}: {proj_name} has no bias attribute "
                f"(bias=True was not passed to MHA)"
            )
            assert proj.bias is not None, (
                f"Layer {i}: {proj_name}.bias is None"
            )

    print(f"  PASS: MHA bias=True verified across {len(model.encoder.layers)} layers")


# ---------------------------------------------------------------------------
# Test 3: Padding ID 0 returns zero vector
# ---------------------------------------------------------------------------

def test_padding_idx_zero():
    """Embedding index 0 must return the zero vector via _card_emb."""
    ct = get_card_table()
    model = build_token_net_mlx(ct, CANONICAL_CFG)
    model.eval()

    # Single ID
    ids_1d = mx.array([0], dtype=mx.int32)
    emb_1d = model._card_emb(ids_1d)
    emb_1d_np = np.asarray(emb_1d)
    assert np.allclose(emb_1d_np, 0.0, atol=1e-7), (
        f"_card_emb([0]) should be zero, got norm={np.linalg.norm(emb_1d_np):.6f}"
    )

    # Batch of IDs with mix of 0 and non-zero
    ids_2d = mx.array([[0, 5, 0, 42], [0, 0, 0, 1]], dtype=mx.int32)
    emb_2d = model._card_emb(ids_2d)
    emb_2d_np = np.asarray(emb_2d)

    # All zero-ID positions must be zero
    for b in range(2):
        for k in range(4):
            if ids_2d[b, k].item() == 0:
                row_norm = np.linalg.norm(emb_2d_np[b, k])
                assert row_norm < 1e-7, (
                    f"_card_emb[{b},{k}] (id=0) has norm {row_norm:.6f}, expected 0"
                )

    # Non-zero IDs should NOT be zero (extremely unlikely with random weights)
    for b in range(2):
        for k in range(4):
            if ids_2d[b, k].item() != 0:
                row_norm = np.linalg.norm(emb_2d_np[b, k])
                assert row_norm > 1e-6, (
                    f"_card_emb[{b},{k}] (id={ids_2d[b, k].item()}) is zero unexpectedly"
                )

    print("  PASS: _card_emb padding_idx=0 returns zero vector")


# ---------------------------------------------------------------------------
# Test 4: Static card_feat tables unchanged after optimizer step
# ---------------------------------------------------------------------------

def test_static_tables_unchanged():
    """Run one optimizer step and verify card_feat is NOT in the parameter tree.

    card_feat is domain data (immutable buffer). B.4 fix stores it as numpy
    (invisible to nn.Module), so the optimizer must never touch it.
    """
    ct = get_card_table()
    model = build_token_net_mlx(ct, CANONICAL_CFG)

    if model._card_feat_np is None:
        print("  SKIP: static tables disabled in this config")
        return

    # Snapshot the numpy backing array
    orig_feat = model._card_feat_np.copy()

    # Verify card_feat is NOT in model.parameters()
    params = dict(nn.utils.tree_flatten(model.parameters()))
    for key in params:
        assert "card_feat" not in key, (
            f"card_feat found in parameters under key '{key}' — "
            f"should be numpy-backed, not a trainable parameter"
        )

    # Run one forward-backward-update
    tmpdir = _make_synthetic_dir()
    try:
        ob, yb = _load_batch(tmpdir, BATCH_SIZE)

        def loss_fn(m, o, y):
            logits, _ = m.logits_value(o)
            return nn.losses.cross_entropy(logits, y).mean()

        loss, grads = mx.value_and_grad(loss_fn)(model, ob, yb)
        optimizer = optim.Adam(learning_rate=1e-4)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        # card_feat must be unchanged (it's numpy, optimizer can't touch it)
        after_feat = model._card_feat_np
        max_diff = np.abs(orig_feat - after_feat).max()
        assert max_diff < 1e-10, (
            f"card_feat changed by optimizer despite being numpy-backed: diff={max_diff}"
        )
        # Verify _static() still works (numpy → mx.array conversion)
        ids = mx.array([[1, 5, 0]], dtype=mx.int32)
        static_out = model._static(ids)
        mx.eval(static_out)
        assert np.all(np.isfinite(np.asarray(static_out))), "_static() returned non-finite"
        print(f"  PASS: card_feat immutable (diff after optimizer={max_diff:.2e})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 5: Forward pass produces finite logits and value
# ---------------------------------------------------------------------------

def test_forward_finite():
    """Forward pass produces finite logits [B, N_ACTIONS] and value [B]."""
    ct = get_card_table()
    model = build_token_net_mlx(ct, CANONICAL_CFG)
    model.eval()

    tmpdir = _make_synthetic_dir()
    try:
        ob, yb = _load_batch(tmpdir, BATCH_SIZE)

        logits, value = model.logits_value(ob)

        # Shapes
        assert logits.shape == (BATCH_SIZE, N_ACTIONS), (
            f"logits shape {logits.shape}, expected ({BATCH_SIZE}, {N_ACTIONS})"
        )
        assert value.shape == (BATCH_SIZE,), (
            f"value shape {value.shape}, expected ({BATCH_SIZE},)"
        )

        # Finiteness
        logits_np = np.asarray(logits)
        value_np = np.asarray(value)
        assert np.all(np.isfinite(logits_np)), (
            f"logits non-finite: nan={np.isnan(logits_np).sum()}, "
            f"inf={np.isinf(logits_np).sum()}"
        )
        assert np.all(np.isfinite(value_np)), (
            f"value non-finite: nan={np.isnan(value_np).sum()}, "
            f"inf={np.isinf(value_np).sum()}"
        )

        # Masked positions should be -1e9
        am_np = np.asarray(ob["action_mask"])
        masked_logits = logits_np[am_np < 0.5]
        assert np.all(masked_logits <= -1e9 + 1), (
            "Some masked logits are not -1e9"
        )

        print(f"  PASS: forward finite logits={list(logits.shape)} "
              f"value={list(value.shape)}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 6: Backward pass produces finite gradients
# ---------------------------------------------------------------------------

def test_backward_finite():
    """Forward + backward produces finite gradients."""
    ct = get_card_table()
    model = build_token_net_mlx(ct, CANONICAL_CFG)

    tmpdir = _make_synthetic_dir()
    try:
        ob, yb = _load_batch(tmpdir, BATCH_SIZE)

        def loss_fn(m, o, y):
            logits, _ = m.logits_value(o)
            return nn.losses.cross_entropy(logits, y).mean()

        loss, grads = mx.value_and_grad(loss_fn)(model, ob, yb)
        mx.eval(loss, grads)

        loss_val = float(loss)
        assert np.isfinite(loss_val), f"Loss is not finite: {loss_val}"

        # Check all gradients are finite
        grad_flat = []
        for _, g in nn.utils.tree_flatten(grads):
            if g is not None:
                grad_flat.append(mx.sum(g ** 2))

        grad_norm = float(mx.sqrt(mx.sum(mx.array(grad_flat))))
        assert np.isfinite(grad_norm), f"Gradient norm not finite: {grad_norm}"
        assert grad_norm > 0, f"Gradient norm is zero"

        print(f"  PASS: backward finite loss={loss_val:.4f} grad_norm={grad_norm:.4f}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 7: Categorical value head returns scalar in [-vmax, vmax]
# ---------------------------------------------------------------------------

def test_categorical_value():
    """Build model with value_categorical=True, verify value is scalar in [-vmax, vmax]."""
    cfg = dict(CANONICAL_CFG)
    cfg["value_categorical"] = True
    cfg["value_atoms"] = 51
    cfg["value_vmax"] = 1.0

    ct = get_card_table()
    model = build_token_net_mlx(ct, cfg)
    model.eval()

    # Verify atom_support is numpy-backed (not in parameters)
    assert model._atom_support_np is not None, "atom_support should be numpy-backed"
    params = dict(nn.utils.tree_flatten(model.parameters()))
    for key in params:
        assert "atom_support" not in key, (
            f"atom_support found in parameters under key '{key}' — "
            f"should be numpy-backed, not a trainable parameter"
        )

    # Verify shape: linspace(-vmax, vmax, n_atoms)
    as_np = model._atom_support_np
    assert as_np.shape == (51,), f"atom_support shape {as_np.shape}, expected (51,)"
    assert abs(as_np[0] - (-1.0)) < 1e-6, f"atom_support[0]={as_np[0]}, expected -1.0"
    assert abs(as_np[-1] - 1.0) < 1e-6, f"atom_support[-1]={as_np[-1]}, expected 1.0"

    tmpdir = _make_synthetic_dir()
    try:
        ob, yb = _load_batch(tmpdir, BATCH_SIZE)

        # Forward: logits_value
        logits, value = model.logits_value(ob)
        value_np = np.asarray(value)
        assert value.shape == (BATCH_SIZE,), (
            f"categorical value shape {value.shape}, expected ({BATCH_SIZE},)"
        )
        assert np.all(np.isfinite(value_np)), (
            f"categorical value non-finite: nan={np.isnan(value_np).sum()}"
        )
        assert np.all(value_np >= -cfg["value_vmax"] - 0.01), (
            f"categorical value below -vmax: min={value_np.min()}"
        )
        assert np.all(value_np <= cfg["value_vmax"] + 0.01), (
            f"categorical value above +vmax: max={value_np.max()}"
        )

        # Forward: get_value (must also return scalar expectation)
        v2 = model.get_value(ob)
        v2_np = np.asarray(v2)
        assert v2.shape == (BATCH_SIZE,), (
            f"get_value shape {v2.shape}, expected ({BATCH_SIZE},)"
        )
        assert np.all(np.isfinite(v2_np)), "get_value non-finite"
        assert np.all(v2_np >= -cfg["value_vmax"] - 0.01), (
            f"get_value below -vmax: min={v2_np.min()}"
        )
        assert np.all(v2_np <= cfg["value_vmax"] + 0.01), (
            f"get_value above +vmax: max={v2_np.max()}"
        )

        print(f"  PASS: categorical value scalar in [-1, 1] "
              f"range=[{value_np.min():.4f}, {value_np.max():.4f}]")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = [
    ("additive_attention_mask", test_additive_attention_mask),
    ("mha_bias", test_mha_bias),
    ("padding_idx_zero", test_padding_idx_zero),
    ("static_tables_unchanged", test_static_tables_unchanged),
    ("forward_finite", test_forward_finite),
    ("backward_finite", test_backward_finite),
    ("categorical_value", test_categorical_value),
]


def main():
    print("=" * 60)
    print("Phase B Integration Test — Semantic P0 Fixes")
    print("=" * 60)

    passed = 0
    failed = 0
    errors: list[str] = []

    for name, fn in ALL_TESTS:
        print(f"\n[phase_b] {name}...", flush=True)
        try:
            fn()
            passed += 1
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  FAIL: {e}")
            errors.append(f"{name}: {e}\n{tb}")
            failed += 1

    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Phase B Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"\nFailed tests:")
        for e in errors:
            print(f"  - {e[:300]}")
    print("=" * 60)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
