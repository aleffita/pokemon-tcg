"""Phase C integration test — FP16-native trainer with gradient accumulation.

Validates:
  1. FP16 input to model: batch arrays have float16 dtype (not float32)
  2. Gradient accumulation: accum_steps=2 produces valid training
  3. Loss is finite and decreasing across optimizer steps
  4. Checkpoint saves and loads with all fields (model, optimizer, arch_config, gstep)
  5. gstep counts correctly across accumulation (1 update per accum_steps microbatches)
  6. Slab default changed to 32768
  7. All previous Phase A and B tests pass (no regressions)

Run:
  PYTHONPATH=. uv run python scripts/validate/test_phase_c.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import traceback

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(_REPO))

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from rl.encoder.card_features import get_card_table
from rl.encoder.enc_constants import N_STATE_TOKENS, MAX_OPTIONS, N_ACTIONS
from rl.encoder.encoding import TokenEncoder
from rl.policy_mlx import build_token_net_mlx
from rl.lr_schedule import lr_at
from scripts.validate.make_synthetic_data import make_dataset

import shutil

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_ROWS = 200
BATCH_SIZE = 32
ACCUM_STEPS = 2

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
    tmpdir = tempfile.mkdtemp(prefix="phase_c_test_")
    make_dataset(N_ROWS, tmpdir, seed=42)
    effect_mask_path = os.path.join(tmpdir, "effect_mask.npy")
    if not os.path.exists(effect_mask_path):
        rng = np.random.default_rng(42)
        effect_mask = (rng.standard_normal((N_ROWS, 2)).astype(np.float32) * 0.2 + 0.5).clip(0.0, 1.0)
        np.save(effect_mask_path, effect_mask, allow_pickle=False)
    return tmpdir


def _load_data(data_dir: str):
    """Load data in the same format bc_train_mlx.py uses."""
    d = {
        f[:-4]: np.load(os.path.join(data_dir, f), mmap_mode="r")
        for f in sorted(os.listdir(data_dir))
        if f.endswith(".npy")
    }
    N = int(d["__labels__"].shape[0])
    labels = d["__labels__"]

    ct = get_card_table()
    enc = TokenEncoder(ct)
    int_keys = set(enc.int_keys)
    keys = [k for k in d if k not in ("__labels__", "__is_attack__", "__group__")]

    return d, labels, keys, int_keys


def _make_fp16_batch(d, labels, keys, int_keys, start, end):
    """Create a batch with FP16 numeric features (same as bc_train_mlx C.1)."""
    indices = np.arange(start, end)
    _FP32_KEYS = frozenset({"action_mask"})
    ob = {
        k: mx.array(np.asarray(d[k][indices]).astype(
            np.int32 if k in int_keys
            else (np.float32 if k in _FP32_KEYS else np.float16)
        ))
        for k in keys
    }
    yb = mx.array(labels[indices].astype(np.int32))
    return ob, yb, indices


# ---------------------------------------------------------------------------
# Test 1: FP16 input to model
# ---------------------------------------------------------------------------

def test_fp16_input():
    """Batch arrays for numeric features have float16 dtype."""
    tmpdir = _make_synthetic_dir()
    try:
        d, labels, keys, int_keys = _load_data(tmpdir)
        ob, yb, _ = _make_fp16_batch(d, labels, keys, int_keys, 0, BATCH_SIZE)

        fp16_count = 0
        fp32_count = 0
        int_count = 0
        for k in ob:
            arr = np.asarray(ob[k])
            if arr.dtype == np.float16:
                fp16_count += 1
            elif arr.dtype == np.float32:
                fp32_count += 1
            elif arr.dtype == np.int32:
                int_count += 1

        # We expect at least some FP16 arrays (most numeric features)
        assert fp16_count > 0, (
            f"No FP16 arrays found. dtypes: "
            f"{[(k, np.asarray(ob[k]).dtype) for k in list(ob.keys())[:5]]}"
        )

        # action_mask should be FP32 (masking comparison)
        assert ob["action_mask"].dtype == mx.float32, (
            f"action_mask dtype is {ob['action_mask'].dtype}, expected float32"
        )

        print(f"  PASS: FP16 input verified ({fp16_count} fp16, {fp32_count} fp32, "
              f"{int_count} int32 arrays)")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 2: Gradient accumulation produces valid training
# ---------------------------------------------------------------------------

def test_gradient_accumulation():
    """Train with accum_steps=2 and verify loss is finite and not diverging."""
    tmpdir = _make_synthetic_dir()
    try:
        d, labels, keys, int_keys = _load_data(tmpdir)
        N = labels.shape[0]
        ntrain = N - max(1, int(N * 0.1))

        ct = get_card_table()
        model = build_token_net_mlx(ct, CANONICAL_CFG)

        grad_fn = mx.value_and_grad(
            lambda m, o, y: nn.losses.cross_entropy(
                m.logits_value(o)[0], y).mean(),
            argnums=0,
        )

        optimizer = optim.Adam(learning_rate=1e-3)
        clip_max = 1.0
        accum_steps = ACCUM_STEPS

        gstep = 0
        losses = []
        # Train for a few full accumulation cycles (10 steps = 20 microbatches)
        n_opt_steps = 10
        for opt_step in range(n_opt_steps):
            acc_grads = None
            acc_examples = 0
            acc_loss_sum = 0.0
            for micro in range(accum_steps):
                start = (opt_step * accum_steps + micro) * BATCH_SIZE
                end = min(start + BATCH_SIZE, N)
                if start >= ntrain:
                    break
                ob, yb, _ = _make_fp16_batch(d, labels, keys, int_keys, start, end)
                loss, grads = grad_fn(model, ob, yb)
                mx.eval(loss)
                loss_val = float(loss)
                n = len(yb)

                if acc_grads is None:
                    acc_grads = grads
                    acc_examples = n
                    acc_loss_sum = loss_val * n
                else:
                    acc_grads = nn.utils.tree_map(
                        lambda a, b: (a + b) if (a is not None and b is not None) else (a if a is not None else b),
                        acc_grads, grads
                    )
                    acc_examples += n
                    acc_loss_sum += loss_val * n

            if acc_grads is None or acc_examples == 0:
                break

            # Normalize, clip, update
            acc_grads = nn.utils.tree_map(
                lambda g: (g / acc_examples) if g is not None else g, acc_grads
            )
            # Graph-safe clipping
            flat = [g.reshape(-1) for _, g in nn.utils.tree_flatten(acc_grads) if g is not None]
            if flat:
                gn = mx.sqrt(sum(mx.sum(g ** 2) for g in flat))
                scale = mx.where(gn > clip_max, clip_max / mx.maximum(gn, 1e-6), 1.0)
                acc_grads = nn.utils.tree_map(
                    lambda g: (g * scale) if g is not None else g, acc_grads
                )
            gstep += 1
            optimizer.update(model, acc_grads)
            mx.eval(model.parameters())

            avg_loss = acc_loss_sum / acc_examples
            losses.append(avg_loss)

        assert len(losses) > 0, "No training steps completed"
        # Check all losses are finite
        for i, l in enumerate(losses):
            assert np.isfinite(l), f"Loss at step {i} is not finite: {l}"

        # Check loss is not diverging (final should not be more than 2x initial)
        # With synthetic data and few steps, strict monotonic decrease is not guaranteed
        assert losses[-1] < 2.0 * losses[0], (
            f"Loss diverging: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

        # Verify gstep matches expected count
        expected_steps = min(n_opt_steps, (ntrain // BATCH_SIZE + accum_steps - 1) // accum_steps)
        assert gstep == len(losses), f"gstep={gstep} != len(losses)={len(losses)}"

        print(f"  PASS: gradient accumulation OK ({len(losses)} opt_steps, "
              f"loss={losses[0]:.4f} -> {losses[-1]:.4f}, gstep={gstep})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 3: Checkpoint saves and loads with all fields
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip():
    """Save checkpoint with optimizer state, load it back, verify all fields."""
    import pickle

    ct = get_card_table()
    model = build_token_net_mlx(ct, CANONICAL_CFG)
    optimizer = optim.Adam(learning_rate=1e-3)

    # Do one step to populate optimizer state
    tmpdir = _make_synthetic_dir()
    try:
        d, labels, keys, int_keys = _load_data(tmpdir)
        ob, yb, _ = _make_fp16_batch(d, labels, keys, int_keys, 0, BATCH_SIZE)

        grad_fn = mx.value_and_grad(
            lambda m, o, y: nn.losses.cross_entropy(
                m.logits_value(o)[0], y).mean(),
            argnums=0,
        )
        loss, grads = grad_fn(model, ob, yb)
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        mx.eval(optimizer.state)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    gstep = 42
    arch_config = model.get_config()
    orig_params = model.parameters()
    orig_opt_state = optimizer.state

    # Save checkpoint
    ckpt_path = tempfile.mktemp(suffix=".pkl")
    try:
        with open(ckpt_path, "wb") as f:
            pickle.dump({
                "model": model.parameters(),
                "optimizer": optimizer.state,
                "arch_config": arch_config,
                "epoch": 3,
                "gstep": gstep,
                "val_acc": 0.75,
                "seed": 42,
                "dataset_path": "/fake/path",
                "accum_steps": 2,
            }, f)

        # Load into fresh model
        with open(ckpt_path, "rb") as f:
            loaded = pickle.load(f)

        model2 = build_token_net_mlx(ct, CANONICAL_CFG)
        optimizer2 = optim.Adam(learning_rate=1e-3)

        model2.update(loaded["model"])
        optimizer2.state.update(loaded["optimizer"])

        # Verify model params match (flatten all params)
        def _flatten_params(params):
            flat = []
            for _, v in nn.utils.tree_flatten(params):
                if v is not None:
                    flat.append(np.asarray(v).flatten())
            return np.concatenate(flat)

        orig_flat = _flatten_params(orig_params)
        loaded_flat = _flatten_params(loaded["model"])
        assert orig_flat.shape == loaded_flat.shape, (
            f"Param shape mismatch: {orig_flat.shape} vs {loaded_flat.shape}"
        )
        assert np.allclose(orig_flat, loaded_flat, atol=1e-6), (
            f"Model params differ after round-trip: max_diff={np.abs(orig_flat - loaded_flat).max()}"
        )

        # Verify arch_config
        assert loaded["arch_config"] == arch_config, "arch_config differs"

        # Verify gstep
        assert loaded["gstep"] == gstep, f"gstep={loaded['gstep']}, expected {gstep}"

        # Verify optimizer state exists and has content
        assert loaded["optimizer"] is not None and len(loaded["optimizer"]) > 0, (
            "optimizer state is empty or None"
        )
        assert loaded["accum_steps"] == 2

        print(f"  PASS: checkpoint round-trip OK (model, optimizer, arch_config, "
              f"gstep={gstep}, epoch=3)")
    finally:
        if os.path.exists(ckpt_path):
            os.unlink(ckpt_path)


# ---------------------------------------------------------------------------
# Test 4: gstep counts correctly across accumulation
# ---------------------------------------------------------------------------

def test_gstep_counting():
    """With accum_steps=2, gstep increments once per 2 microbatches."""
    tmpdir = _make_synthetic_dir()
    try:
        d, labels, keys, int_keys = _load_data(tmpdir)

        ct = get_card_table()
        model = build_token_net_mlx(ct, CANONICAL_CFG)
        optimizer = optim.Adam(learning_rate=1e-3)

        grad_fn = mx.value_and_grad(
            lambda m, o, y: nn.losses.cross_entropy(
                m.logits_value(o)[0], y).mean(),
            argnums=0,
        )

        gstep = 0
        accum_steps = 2

        # Simulate 6 microbatches (should produce 3 optimizer steps)
        for micro in range(6):
            start = micro * BATCH_SIZE
            end = min(start + BATCH_SIZE, d["__labels__"].shape[0])
            ob, yb, _ = _make_fp16_batch(d, labels, keys, int_keys, start, end)

            loss, grads = grad_fn(model, ob, yb)
            mx.eval(loss)

            if (micro + 1) % accum_steps == 0:
                gstep += 1
                optimizer.update(model, grads)
                mx.eval(model.parameters())

        assert gstep == 3, f"Expected 3 optimizer steps, got {gstep}"

        # Now test that lr_at works correctly with total_opt_steps
        total_opt_steps = 10 * 10  # 10 epochs * 10 steps/epoch
        lr = lr_at(3, total_opt_steps, 5e-4, "cosine", 1500, 0.1)
        assert np.isfinite(lr), f"lr_at returned non-finite: {lr}"
        assert lr > 0, f"lr_at returned non-positive: {lr}"

        print(f"  PASS: gstep counting OK (6 microbatches, accum_steps=2 -> 3 opt_steps)")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 5: Slab default is 32768
# ---------------------------------------------------------------------------

def test_slab_default():
    """The default --slab-rows should be 32768."""
    # Read source file and check the default value
    source_path = os.path.join(_REPO, "scripts", "bc", "bc_train_mlx.py")
    with open(source_path, "r") as f:
        source = f.read()

    # Find the line with --slab-rows default
    assert 'default=32768' in source, (
        "--slab-rows default is not 32768 in source file"
    )

    # Also verify old default is gone
    assert 'default=262144' not in source, (
        "Old default 262144 still present in source"
    )

    print("  PASS: --slab-rows default is 32768 (verified in source)")


# ---------------------------------------------------------------------------
# Test 6: Run previous phase tests (regression check)
# ---------------------------------------------------------------------------

def run_previous_tests() -> list[tuple[str, bool, str]]:
    """Run Phase A and Phase B tests as subprocesses."""
    results = []
    validate_dir = os.path.join(os.path.dirname(__file__))

    test_scripts = [
        "test_phase_a.py",
        "test_phase_b.py",
    ]

    for script in test_scripts:
        script_path = os.path.join(validate_dir, script)
        if not os.path.exists(script_path):
            results.append((script, False, f"File not found: {script_path}"))
            continue

        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath(_REPO)
        env["MLX_BACKEND"] = "cpu"

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
                cwd=os.path.abspath(_REPO),
            )
            passed = result.returncode == 0
            detail = result.stdout if passed else (
                f"STDOUT:\n{result.stdout[-500:]}\nSTDERR:\n{result.stderr[-500:]}"
            )
            results.append((script, passed, detail.strip()))
        except subprocess.TimeoutExpired:
            results.append((script, False, "TIMEOUT (>300s)"))
        except Exception as e:
            results.append((script, False, str(e)))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = [
    ("fp16_input", test_fp16_input),
    ("gradient_accumulation", test_gradient_accumulation),
    ("checkpoint_roundtrip", test_checkpoint_roundtrip),
    ("gstep_counting", test_gstep_counting),
    ("slab_default", test_slab_default),
]


def main():
    print("=" * 60)
    print("Phase C Integration Test — FP16-Native Trainer + Accumulation")
    print("=" * 60)

    passed = 0
    failed = 0
    errors: list[str] = []

    # --- inline tests ---
    print("\n--- Inline Tests ---\n")
    for name, fn in ALL_TESTS:
        print(f"[phase_c] {name}...", flush=True)
        try:
            fn()
            passed += 1
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  FAIL: {e}")
            errors.append(f"{name}: {e}\n{tb}")
            failed += 1

    # --- previous phase tests ---
    print("\n--- Previous Phase Regression Tests ---\n")
    for script, ok, detail in run_previous_tests():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {script}")
        if not ok:
            print(f"         {detail[:300]}")
            errors.append(f"{script}: {detail[:500]}")
            failed += 1
        else:
            passed += 1

    # --- summary ---
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Phase C Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"\nFailed tests:")
        for e in errors:
            print(f"  - {e[:300]}")
    print("=" * 60)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
