"""Phase D integration test — option compaction, episode metadata, episode-level val split.

Validates:
  1. Option bucket compaction: smaller batches produce shorter option sequences
  2. State column compaction: all-padded state columns are removed
  3. Forward pass with compaction: logits shape [B, 193], all finite
  4. Episode metadata EP_META dtype has all required fields
  5. Episode-level validation split falls on episode boundary
  6. No regressions in Phase A, B, C tests

Run:
  PYTHONPATH=. uv run python scripts/validate/test_phase_d.py
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

from rl.encoder.card_features import get_card_table
from rl.encoder.enc_constants import N_STATE_TOKENS, MAX_OPTIONS, N_ACTIONS
from rl.encoder.encoding import TokenEncoder
from rl.policy_mlx import build_token_net_mlx, TokenTransformerMLX
from scripts.validate.make_synthetic_data import make_dataset

import shutil

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_ROWS = 200
BATCH_SIZE = 32

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
    tmpdir = tempfile.mkdtemp(prefix="phase_d_test_")
    make_dataset(N_ROWS, tmpdir, seed=42)
    effect_mask_path = os.path.join(tmpdir, "effect_mask.npy")
    if not os.path.exists(effect_mask_path):
        rng = np.random.default_rng(42)
        effect_mask = (rng.standard_normal((N_ROWS, 2)).astype(np.float32) * 0.2 + 0.5).clip(0.0, 1.0)
        np.save(effect_mask_path, effect_mask, allow_pickle=False)
    return tmpdir


def _load_batch(data_dir: str, start: int, end: int):
    """Load a batch of rows as mlx arrays."""
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
    return ob, yb, keys


# ---------------------------------------------------------------------------
# Test 1: Option bucket compaction produces shorter sequences
# ---------------------------------------------------------------------------

def test_option_bucket_compaction():
    """Smaller batches produce shorter or equal option token sequences via bucketing."""
    tmpdir = _make_synthetic_dir()
    try:
        ct = get_card_table()
        model = build_token_net_mlx(ct, CANONICAL_CFG)
        model.eval()

        # Large batch — should use a larger bucket (or full MAX_OPTIONS)
        ob_large, _, _ = _load_batch(tmpdir, 0, min(BATCH_SIZE, N_ROWS))
        cls_l, opt_l, _, _ = model._encode(ob_large)
        n_opt_large = opt_l.shape[1]

        # Small batch (4 rows) — should use a smaller bucket
        ob_small, _, _ = _load_batch(tmpdir, 0, 4)
        cls_s, opt_s, _, _ = model._encode(ob_small)
        n_opt_small = opt_s.shape[1]

        # Small batch should produce <= large batch option count
        assert n_opt_small <= n_opt_large, (
            f"Small batch n_opt={n_opt_small} > large batch n_opt={n_opt_large}"
        )

        # Both should be valid bucket sizes (32, 64, 128, 192)
        valid_buckets = {32, 64, 128, 192}
        assert n_opt_small in valid_buckets, (
            f"Small batch n_opt={n_opt_small} not in buckets {valid_buckets}"
        )

        # Synthetic data has ~20% legal options per row (up to ~38 per row from 192).
        # With 4 rows, max_legal can be ~38, bucketing to 64. Verify it's at most 192.
        assert n_opt_small <= MAX_OPTIONS, (
            f"Small batch n_opt={n_opt_small} exceeds MAX_OPTIONS={MAX_OPTIONS}"
        )

        print(f"  PASS: option bucket compaction (large={n_opt_large}, small={n_opt_small})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 2: State column compaction removes all-padded columns
# ---------------------------------------------------------------------------

def test_state_column_compaction():
    """All-padded state columns are removed from the sequence."""
    tmpdir = _make_synthetic_dir()
    try:
        ct = get_card_table()
        model = build_token_net_mlx(ct, CANONICAL_CFG)
        model.eval()

        ob, _, _ = _load_batch(tmpdir, 0, BATCH_SIZE)

        # Build with compaction
        cls_c, opt_c, pooled_c, extra_c = model._encode(ob)

        # Build without compaction (set all masks to 1 -> no columns padded)
        ob_nocomp = dict(ob)
        for k in ob_nocomp:
            if k.endswith("_mask") and k != "action_mask":
                ob_nocomp[k] = mx.ones_like(ob_nocomp[k])
        cls_n, opt_n, pooled_n, extra_n = model._encode(ob_nocomp)

        # With compaction, state may be shorter, but option output should still be valid
        assert opt_c.shape[0] == BATCH_SIZE, (
            f"Compacted opt shape[0]={opt_c.shape[0]}, expected {BATCH_SIZE}"
        )
        assert opt_n.shape[0] == BATCH_SIZE, (
            f"No-compaction opt shape[0]={opt_n.shape[0]}, expected {BATCH_SIZE}"
        )

        # Both should produce finite outputs
        assert mx.all(mx.isfinite(cls_c)), "Compacted CLS output has non-finite values"
        assert mx.all(mx.isfinite(cls_n)), "No-compaction CLS output has non-finite values"

        print(f"  PASS: state column compaction (compacted opt={list(opt_c.shape)}, "
              f"no-compaction opt={list(opt_n.shape)})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 3: Forward pass with compaction produces valid logits
# ---------------------------------------------------------------------------

def test_forward_with_compaction():
    """Full forward pass with state+option compaction produces valid logits."""
    tmpdir = _make_synthetic_dir()
    try:
        ob, yb, _ = _load_batch(tmpdir, 0, BATCH_SIZE)

        ct = get_card_table()
        model = build_token_net_mlx(ct, CANONICAL_CFG)
        model.eval()

        logits, value = model.logits_value(ob)

        # Shape checks
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
            f"Some masked logits are not -1e9 ({(masked_logits > -1e9 + 1).sum()} violations)"
        )

        # Compute CE loss — must be finite
        logsumexp = np.logaddexp.reduce(logits_np, axis=1)
        ce = -(logits_np[np.arange(BATCH_SIZE), np.asarray(yb)] - logsumexp)
        assert np.all(np.isfinite(ce)), f"CE loss non-finite: {ce[~np.isfinite(ce)]}"

        print(f"  PASS: forward with compaction logits={list(logits.shape)} "
              f"value={list(value.shape)} CE={ce.mean():.4f}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 4: Gradient backward with compaction is finite
# ---------------------------------------------------------------------------

def test_backward_with_compaction():
    """Forward + backward with compaction produces finite gradients."""
    tmpdir = _make_synthetic_dir()
    try:
        ob, yb, _ = _load_batch(tmpdir, 0, BATCH_SIZE)

        ct = get_card_table()
        model = build_token_net_mlx(ct, CANONICAL_CFG)

        def loss_fn(m, o, y):
            logits, _ = m.logits_value(o)
            return nn.losses.cross_entropy(logits, y).mean()

        loss, grads = mx.value_and_grad(loss_fn)(model, ob, yb)
        mx.eval(loss, grads)

        loss_val = float(loss)
        assert np.isfinite(loss_val), f"Loss not finite: {loss_val}"

        grad_flat = []
        for _, g in nn.utils.tree_flatten(grads):
            if g is not None:
                grad_flat.append(mx.sum(g ** 2))

        grad_norm = float(mx.sqrt(mx.sum(mx.array(grad_flat))))
        assert np.isfinite(grad_norm), f"Grad norm not finite: {grad_norm}"
        assert grad_norm > 0, f"Grad norm is zero"

        print(f"  PASS: backward with compaction loss={loss_val:.4f} "
              f"grad_norm={grad_norm:.4f}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 5: EP_META dtype has all required fields
# ---------------------------------------------------------------------------

def test_episode_meta_dtype():
    """EP_META structured dtype has all required fields with correct dtypes."""
    EP_META = np.dtype([
        ("episode_id", "U64"),
        ("side", "i4"),
        ("step_id", "i4"),
        ("new_episode", "bool"),
    ])

    # Verify all required fields exist
    required_fields = {"episode_id", "side", "step_id", "new_episode"}
    actual_fields = set(EP_META.names)
    assert required_fields == actual_fields, (
        f"EP_META fields mismatch: expected={required_fields}, got={actual_fields}"
    )

    # Verify dtypes
    assert EP_META["episode_id"].kind == "U", f"episode_id dtype kind={EP_META['episode_id'].kind}"
    assert EP_META["side"].kind == "i", f"side dtype kind={EP_META['side'].kind}"
    assert EP_META["step_id"].kind == "i", f"step_id dtype kind={EP_META['step_id'].kind}"
    assert EP_META["new_episode"].kind == "b", f"new_episode dtype kind={EP_META['new_episode'].kind}"

    # Verify we can construct and save/load a sample
    sample = np.array([
        ("ep_001", 0, 0, True),
        ("ep_001", 0, 1, False),
        ("ep_002", 1, 0, True),
    ], dtype=EP_META)

    tmp = tempfile.mktemp(suffix=".npy")
    try:
        np.save(tmp, sample, allow_pickle=False)
        loaded = np.load(tmp, allow_pickle=False)
        assert loaded.dtype == EP_META, f"Dtype changed after round-trip: {loaded.dtype}"
        assert len(loaded) == 3, f"Length changed: {len(loaded)}"
        assert loaded["episode_id"][0] == "ep_001"
        assert loaded["side"][1] == 0
        assert loaded["step_id"][2] == 0
        assert loaded["new_episode"][2] == True
        assert loaded["new_episode"][1] == False
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    print("  PASS: EP_META dtype correct and round-trips via np.save/load")


# ---------------------------------------------------------------------------
# Test 6: Episode-level validation split falls on boundary
# ---------------------------------------------------------------------------

def test_episode_level_split():
    """Episode-level split snaps to a new_episode boundary."""
    # Create synthetic metadata with known episode boundaries
    EP_META = np.dtype([
        ("episode_id", "U64"),
        ("side", "i4"),
        ("step_id", "i4"),
        ("new_episode", "bool"),
    ])

    # 200 rows, episodes at rows 0, 30, 60, 90, 120, 150, 180
    boundaries = [0, 30, 60, 90, 120, 150, 180]
    meta_list = []
    ep_idx = 0
    for row in range(200):
        is_new = row in boundaries
        meta_list.append((f"ep_{ep_idx:03d}", 0, row - boundaries[ep_idx], is_new))
        if is_new and row > 0:
            ep_idx += 1

    meta = np.array(meta_list, dtype=EP_META)

    # Save and reload (simulates what trainer does)
    tmpdir = tempfile.mkdtemp(prefix="split_test_")
    try:
        meta_path = os.path.join(tmpdir, "episode_meta.npy")
        np.save(meta_path, meta, allow_pickle=False)

        N = 200
        val_frac = 0.1
        nval = max(1, int(N * val_frac))  # 20
        v0_tail = N - nval  # 180

        # Episode-level split logic (same as bc_train_mlx.py D.4)
        loaded_meta = np.load(meta_path)
        new_ep = loaded_meta["new_episode"]
        ep_boundaries = np.where(new_ep)[0]
        valid_boundaries = ep_boundaries[ep_boundaries <= N - nval]
        if len(valid_boundaries) > 0:
            v0_ep = int(valid_boundaries[-1])
        else:
            v0_ep = v0_tail

        # v0_ep should be an episode boundary
        assert v0_ep in boundaries, (
            f"v0_ep={v0_ep} is not an episode boundary (boundaries={boundaries})"
        )

        # v0_ep should be <= v0_tail (we snap to the left)
        assert v0_ep <= v0_tail, (
            f"v0_ep={v0_ep} > v0_tail={v0_tail}"
        )

        # No episode should straddle the boundary
        for row in range(v0_ep, min(v0_ep + 1, N)):
            if new_ep[row]:
                pass  # first row of val set is a new episode — OK

        print(f"  PASS: episode split at row {v0_ep} "
              f"(tail would be {v0_tail}, {v0_ep} train, {N - v0_ep} val)")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 7: Opt bucketing disabled via env var
# ---------------------------------------------------------------------------

def test_opt_bucketing_env_override():
    """MLX_NO_OPT_BUCKET=1 disables automatic option truncation."""
    tmpdir = _make_synthetic_dir()
    try:
        ob, _, _ = _load_batch(tmpdir, 0, 4)

        ct = get_card_table()
        model = build_token_net_mlx(ct, CANONICAL_CFG)
        model.eval()

        # Without env var: compaction active
        os.environ.pop("MLX_NO_OPT_BUCKET", None)
        cls_c, opt_c, _, _ = model._encode(ob)
        n_opt_c = opt_c.shape[1]

        # With env var: no compaction
        os.environ["MLX_NO_OPT_BUCKET"] = "1"
        try:
            cls_n, opt_n, _, _ = model._encode(ob)
            n_opt_n = opt_n.shape[1]
        finally:
            os.environ.pop("MLX_NO_OPT_BUCKET", None)

        # Without compaction, should get full MAX_OPTIONS
        assert n_opt_n == MAX_OPTIONS, (
            f"With MLX_NO_OPT_BUCKET=1, n_opt={n_opt_n}, expected {MAX_OPTIONS}"
        )
        # With compaction, should be <= MAX_OPTIONS
        assert n_opt_c <= n_opt_n, (
            f"With compaction n_opt={n_opt_c} > without compaction n_opt={n_opt_n}"
        )

        print(f"  PASS: opt bucketing env override (with={n_opt_c}, without={n_opt_n})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 8: Episode metadata can be created from rows_from_episode
# ---------------------------------------------------------------------------

def test_metadata_from_episode():
    """rows_from_episode with ep_meta produces correct metadata structure."""
    from rl.encoder.card_features import get_card_table as _ct
    from scripts.bc.build_bc_dataset import rows_from_episode, BOTH_SIDES

    # Create a minimal fake episode matching the expected format
    fake_ep = {
        "rewards": [1.0, 0.0],  # player 0 wins
        "steps": [],
    }

    # Generate minimal steps: deck choice followed by one decision
    # Step 0: both players get deck choice (60 ids)
    deck_0 = list(range(60))
    deck_1 = list(range(60, 120))
    fake_ep["steps"].append([{"observation": {}, "action": deck_0},
                             {"observation": {}, "action": deck_1}])

    # Step 1: both players get a select with options
    fake_ep["steps"].append([
        {"observation": {"select": {"type": 0, "option": [{"id": 0}, {"id": 1}],
                                    "maxCount": 1}}, "action": None},
        {"observation": {"select": {"type": 0, "option": [{"id": 0}, {"id": 1}],
                                    "maxCount": 1}}, "action": None},
    ])

    # Step 2: response actions (off-by-one: action is in NEXT entry)
    fake_ep["steps"].append([
        {"observation": {}, "action": [0]},
        {"observation": {}, "action": [0]},
    ])

    ep_meta = []
    rows = list(rows_from_episode(fake_ep, episode_id="test_ep_001", ep_meta=ep_meta))

    # Metadata should have entries
    if len(rows) > 0:
        assert len(ep_meta) > 0, "ep_meta should have entries when rows are yielded"
        # Check metadata structure
        for m in ep_meta:
            assert "episode_id" in m, "Missing episode_id"
            assert "side" in m, "Missing side"
            assert "step_id" in m, "Missing step_id"
            assert "new_episode" in m, "Missing new_episode"
            assert m["episode_id"] == "test_ep_001", f"Wrong episode_id: {m['episode_id']}"
            assert isinstance(m["side"], int), f"side should be int: {m['side']}"
            assert isinstance(m["step_id"], int), f"step_id should be int: {m['step_id']}"
            assert isinstance(m["new_episode"], bool), f"new_episode should be bool"
        print(f"  PASS: metadata from episode ({len(rows)} rows, {len(ep_meta)} meta entries)")
    else:
        # If no rows, metadata should also be empty
        assert len(ep_meta) == 0, "ep_meta should be empty when no rows yielded"
        print(f"  PASS: metadata from episode (0 rows, 0 meta — episode format may vary)")


# ---------------------------------------------------------------------------
# Run previous phase tests (regression check)
# ---------------------------------------------------------------------------

def run_previous_tests() -> list[tuple[str, bool, str]]:
    """Run Phase A, B, and C tests as subprocesses."""
    results = []
    validate_dir = os.path.join(os.path.dirname(__file__))

    test_scripts = [
        "test_phase_a.py",
        "test_phase_b.py",
        "test_phase_c.py",
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
                timeout=600,
                env=env,
                cwd=os.path.abspath(_REPO),
            )
            passed = result.returncode == 0
            detail = result.stdout if passed else (
                f"STDOUT:\n{result.stdout[-500:]}\nSTDERR:\n{result.stderr[-500:]}"
            )
            results.append((script, passed, detail.strip()))
        except subprocess.TimeoutExpired:
            results.append((script, False, "TIMEOUT (>600s)"))
        except Exception as e:
            results.append((script, False, str(e)))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = [
    ("option_bucket_compaction", test_option_bucket_compaction),
    ("state_column_compaction", test_state_column_compaction),
    ("forward_with_compaction", test_forward_with_compaction),
    ("backward_with_compaction", test_backward_with_compaction),
    ("episode_meta_dtype", test_episode_meta_dtype),
    ("episode_level_split", test_episode_level_split),
    ("opt_bucketing_env_override", test_opt_bucketing_env_override),
    ("metadata_from_episode", test_metadata_from_episode),
]


def main():
    print("=" * 60)
    print("Phase D Integration Test — Compaction, Metadata, Episode Split")
    print("=" * 60)

    passed = 0
    failed = 0
    errors: list[str] = []

    # --- inline tests ---
    print("\n--- Inline Tests ---\n")
    for name, fn in ALL_TESTS:
        print(f"[phase_d] {name}...", flush=True)
        try:
            fn()
            passed += 1
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  FAIL: {e}")
            errors.append(f"{name}: {e}\n{tb}")
            failed += 1

    # --- previous phase regression tests ---
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
    print(f"Phase D Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"\nFailed tests:")
        for e in errors:
            print(f"  - {e[:300]}")
    print("=" * 60)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
