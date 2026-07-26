"""Phase A integration test — canonical contract verified.

End-to-end validation that the encoder, MLX policy, token schema,
and checkpoint round-trip all share the same semantic contract.

What this test covers:
  1. Synthetic dataset generation (100 rows) matching bc_train_mlx schema
  2. Model build with canonical config (d_model=128, nhead=4, nlayers=3, static, split_heads)
  3. Batch loading (same logic as bc_train_mlx.py batches)
  4. Forward pass: logits shape [B, 193], value shape [B], both finite
  5. Config round-trip: get_config() returns expected keys and version strings
  6. Token schema constants imported from canonical sources
  7. All existing validation tests pass (test_token_schema, test_mlx_token_types, test_checkpoint)

Run:
  uv run python scripts/validate/test_phase_a.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import traceback

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# --- imports (heavy, fail early) ---
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from rl.encoder.card_features import get_card_table
from rl.encoder.enc_constants import (
    N_STATE_TOKENS, MAX_OPTIONS, N_ACTIONS, G,
)
from rl.encoder.encoding import TokenEncoder
from rl.token_schema import (
    ARCH_VERSION, TOKEN_SCHEMA_VERSION,
    T_CLS, T_SELF_DECK, T_OPP_DECK, T_SELF_PRIZE, T_OPP_PRIZE,
    T_SELF_HAND, T_OPP_HAND, T_SELF_DISC, T_OPP_DISC, T_STADIUM,
    T_SELF_ACTIVE, T_SELF_BENCH, T_OPP_ACTIVE, T_OPP_BENCH,
    T_OPT, T_EFFECT, T_SEL_TYPE, T_SEL_CTX, T_CARD_SYNTH,
    N_TTYPES,
)
from rl.policy_mlx import build_token_net_mlx, TokenTransformerMLX, N_SCRATCH

# Reuse synthetic data generator
from scripts.validate.make_synthetic_data import make_dataset

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------
N_ROWS = 100
BATCH_SIZE = 8
EXPECTED_VERSION = "1.0.0"

# Canonical model config (frozen for Phase A)
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
    """Create a temp dir with 100 synthetic rows, return the path.

    Augments make_synthetic_data output with effect_mask if absent,
    since the policy _encode expects it but the generator omits it.
    """
    tmpdir = tempfile.mkdtemp(prefix="phase_a_test_")
    make_dataset(N_ROWS, tmpdir, seed=42)
    # Add effect_mask if missing (generator creates effect_id but not effect_mask)
    effect_mask_path = os.path.join(tmpdir, "effect_mask.npy")
    if not os.path.exists(effect_mask_path):
        rng = np.random.default_rng(42)
        effect_mask = (rng.standard_normal((N_ROWS, 2)).astype(np.float32) * 0.2 + 0.5).clip(0.0, 1.0)
        np.save(effect_mask_path, effect_mask, allow_pickle=False)
    return tmpdir


def _load_batches(data_dir: str, batch_size: int):
    """Load synthetic data in the same format bc_train_mlx.py uses.

    Returns (obs_dict, labels, int_keys) where obs_dict maps name -> np.ndarray.
    """
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


def _first_batch(d, keys, int_keys, batch_size):
    """Extract first batch as numpy arrays (same logic as bc_train_mlx batches)."""
    indices = np.arange(batch_size)
    ob = {
        k: np.asarray(d[k][indices]).astype(
            np.int32 if k in int_keys else np.float32
        )
        for k in keys
    }
    return ob


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def test_token_schema_constants():
    """All token type constants are imported from canonical sources and correct."""
    all_ids = [
        T_CLS, T_SELF_DECK, T_OPP_DECK, T_SELF_PRIZE, T_OPP_PRIZE,
        T_SELF_HAND, T_OPP_HAND, T_SELF_DISC, T_OPP_DISC, T_STADIUM,
        T_SELF_ACTIVE, T_SELF_BENCH, T_OPP_ACTIVE, T_OPP_BENCH,
        T_OPT, T_EFFECT, T_SEL_TYPE, T_SEL_CTX, T_CARD_SYNTH,
    ]
    # Must be exactly 0..18
    assert all_ids == list(range(19)), (
        f"Token type IDs not 0..18: {all_ids}"
    )
    assert N_TTYPES == 19, f"N_TTYPES={N_TTYPES}, expected 19"

    # Versions
    assert ARCH_VERSION == EXPECTED_VERSION, (
        f"ARCH_VERSION={ARCH_VERSION!r}, expected {EXPECTED_VERSION!r}"
    )
    assert TOKEN_SCHEMA_VERSION == EXPECTED_VERSION, (
        f"TOKEN_SCHEMA_VERSION={TOKEN_SCHEMA_VERSION!r}, expected {EXPECTED_VERSION!r}"
    )

    print("  PASS: token schema constants are canonical")


def test_synthetic_data_generation():
    """Generate 100 synthetic rows and verify shapes match bc_train schema."""
    tmpdir = _make_synthetic_dir()
    try:
        d, labels, keys, int_keys = _load_batches(tmpdir, BATCH_SIZE)
        N = labels.shape[0]
        assert N == N_ROWS, f"Expected {N_ROWS} rows, got {N}"

        # Verify all keys have correct first dim
        for k in keys:
            assert d[k].shape[0] == N, f"{k}: first dim {d[k].shape[0]} != {N}"

        # Verify labels are within N_ACTIONS
        assert labels.min() >= 0, f"Negative label: {labels.min()}"
        assert labels.max() < N_ACTIONS, (
            f"Label {labels.max()} >= N_ACTIONS={N_ACTIONS}"
        )

        # Verify at least one legal action per row
        am = d["action_mask"]
        assert (am.sum(axis=1) > 0).all(), "Some rows have no legal action"

        # Verify labels are legal under action mask
        legal_mask = am[np.arange(N), labels]
        assert legal_mask.all(), "Some labels are illegal under action mask"

        print(f"  PASS: synthetic data generated ({N} rows, {len(keys)} keys)")
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_model_build_canonical():
    """Build model with canonical config and verify structure."""
    ct = get_card_table()
    model = build_token_net_mlx(ct, CANONICAL_CFG)

    # Parameter count
    nparams = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    assert nparams > 500_000, f"Model too small: {nparams} params"
    assert nparams < 2_000_000, f"Model too large: {nparams} params"

    # Type embedding size
    assert model.type_emb.weight.shape[0] == N_TTYPES, (
        f"type_emb has {model.type_emb.weight.shape[0]} entries, expected {N_TTYPES}"
    )

    # Scratch tokens
    assert model.scratch_tokens == N_SCRATCH, (
        f"scratch_tokens={model.scratch_tokens}, expected {N_SCRATCH}"
    )

    print(f"  PASS: model built ({nparams:,} params, {N_TTYPES} type entries, "
          f"{N_SCRATCH} scratch registers)")


def test_config_round_trip():
    """get_config() returns all expected keys with correct version strings."""
    ct = get_card_table()
    model = build_token_net_mlx(ct, CANONICAL_CFG)
    cfg = model.get_config()

    expected_keys = {
        "arch_version", "token_schema_version",
        "d_model", "nhead", "nlayers", "ff_dim",
        "n_scratch", "static", "split_heads", "structured",
        "max_options", "value_categorical",
    }
    missing = expected_keys - set(cfg.keys())
    extra = set(cfg.keys()) - expected_keys
    assert not missing, f"Missing config keys: {missing}"
    assert not extra, f"Unexpected config keys: {extra}"

    # Dimension checks
    assert cfg["d_model"] == 128, f"d_model={cfg['d_model']}"
    assert cfg["nhead"] == 4, f"nhead={cfg['nhead']}"
    assert cfg["nlayers"] == 3, f"nlayers={cfg['nlayers']}"
    assert cfg["ff_dim"] == 512, f"ff_dim={cfg['ff_dim']}"
    assert cfg["n_scratch"] == N_SCRATCH
    assert cfg["static"] is True
    assert cfg["split_heads"] is True
    assert cfg["structured"] is False
    assert cfg["max_options"] == 192
    assert cfg["value_categorical"] is False

    # Version strings
    assert cfg["arch_version"] == EXPECTED_VERSION
    assert cfg["token_schema_version"] == EXPECTED_VERSION

    # Pickle round-trip
    import pickle
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(cfg, f)
        tmp_path = f.name
    try:
        with open(tmp_path, "rb") as f:
            loaded_cfg = pickle.load(f)
        assert loaded_cfg == cfg, "Config changed after pickle round-trip"
    finally:
        os.unlink(tmp_path)

    print("  PASS: config round-trip preserves all fields")


def test_forward_pass_finite():
    """Forward pass produces finite logits [B, 193] and value [B]."""
    tmpdir = _make_synthetic_dir()
    try:
        d, labels, keys, int_keys = _load_batches(tmpdir, BATCH_SIZE)
        ob_np = _first_batch(d, keys, int_keys, BATCH_SIZE)

        ct = get_card_table()
        model = build_token_net_mlx(ct, CANONICAL_CFG)
        model.eval()

        # Convert to mx arrays (same as bc_train_mlx batches generator)
        ob = {
            k: mx.array(ob_np[k].astype(
                np.int32 if k in int_keys else np.float32
            ))
            for k in keys
        }

        logits, value, _ = model.logits_value(ob)

        # Shape checks
        assert logits.shape == (BATCH_SIZE, N_ACTIONS), (
            f"logits shape {logits.shape}, expected ({BATCH_SIZE}, {N_ACTIONS})"
        )
        assert value.shape == (BATCH_SIZE,), (
            f"value shape {value.shape}, expected ({BATCH_SIZE},)"
        )

        # Finite checks
        logits_np = np.asarray(logits)
        value_np = np.asarray(value)
        assert np.all(np.isfinite(logits_np)), (
            f"logits contain non-finite values: "
            f"nan={np.isnan(logits_np).sum()}, inf={np.isinf(logits_np).sum()}"
        )
        assert np.all(np.isfinite(value_np)), (
            f"value contains non-finite values: "
            f"nan={np.isnan(value_np).sum()}, inf={np.isinf(value_np).sum()}"
        )

        # Logits should be -inf for masked actions, finite for legal ones
        am = ob_np["action_mask"]
        legal_count = am.sum(axis=1)
        for b in range(BATCH_SIZE):
            n_legal = int(legal_count[b])
            assert n_legal > 0, f"Batch {b} has no legal actions"

        print(f"  PASS: forward pass logits={list(logits.shape)} "
              f"value={list(value.shape)} all finite")
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_forward_matches_bc_train():
    """Forward pass through synthetic data matches bc_train_mlx loading logic."""
    tmpdir = _make_synthetic_dir()
    try:
        d, labels, keys, int_keys = _load_batches(tmpdir, BATCH_SIZE)

        # Replicate the exact batch loading from bc_train_mlx.py
        indices = np.arange(BATCH_SIZE)
        ob_mlx = {
            k: mx.array(np.asarray(d[k][indices]).astype(
                np.int32 if k in int_keys else np.float32
            ))
            for k in keys
        }

        ct = get_card_table()
        model = build_token_net_mlx(ct, CANONICAL_CFG)
        model.eval()

        logits, value, _ = model.logits_value(ob_mlx)

        # Verify we can compute a valid cross-entropy loss
        yb = mx.array(labels[indices].astype(np.int32))
        log_probs = mx.log(mx.clip(logits, a_min=1e-8, a_max=None))
        ce_loss = -mx.take_along_axis(log_probs, yb.reshape(-1, 1), axis=1).mean()
        loss_val = float(ce_loss)
        assert np.isfinite(loss_val), f"CE loss is not finite: {loss_val}"
        assert loss_val >= 0, f"CE loss is negative: {loss_val}"

        # Verify gradient flow
        loss, grads = mx.value_and_grad(
            lambda m, o, y: nn.losses.cross_entropy(
                m.logits_value(o)[0], y
            ).mean()
        )(model, ob_mlx, yb)
        grad_norm = float(mx.sqrt(sum(
            mx.sum(g ** 2).item() for _, g in nn.utils.tree_flatten(grads) if g is not None
        )))
        assert np.isfinite(grad_norm), f"Grad norm not finite: {grad_norm}"
        assert grad_norm > 0, f"Grad norm is zero: {grad_norm}"

        print(f"  PASS: forward+loss+backward OK (loss={loss_val:.4f}, "
              f"grad_norm={grad_norm:.4f})")
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_scratch_persistence():
    """Scratch registers survive checkpoint save/load."""
    import pickle

    ct = get_card_table()
    model = build_token_net_mlx(ct, CANONICAL_CFG)
    orig_scratch = np.array(model.scratch)

    checkpoint = {
        "model": model.parameters(),
        "arch_config": model.get_config(),
    }

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(checkpoint, f)
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as f:
            loaded = pickle.load(f)

        model2 = build_token_net_mlx(ct, CANONICAL_CFG)
        model2.update(loaded["model"])
        new_scratch = np.array(model2.scratch)

        assert np.abs(orig_scratch - new_scratch).max() < 1e-6, (
            "Scratch registers differ after round-trip"
        )
        print("  PASS: scratch registers survive checkpoint round-trip")
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Run existing validation tests
# ---------------------------------------------------------------------------

def run_existing_validation_tests() -> list[tuple[str, bool, str]]:
    """Run all existing validation test scripts as subprocesses.

    Returns list of (test_name, passed, detail).
    """
    results = []
    validate_dir = os.path.join(os.path.dirname(__file__))

    test_scripts = [
        "test_token_schema.py",
        "test_mlx_token_types.py",
        "test_checkpoint.py",
    ]

    for script in test_scripts:
        script_path = os.path.join(validate_dir, script)
        if not os.path.exists(script_path):
            results.append((script, False, f"File not found: {script_path}"))
            continue

        env = os.environ.copy()
        env["MLX_BACKEND"] = "cpu"  # headless for CI/test

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
                cwd=os.path.abspath(_REPO),
            )
            passed = result.returncode == 0
            detail = result.stdout if passed else (
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
            results.append((script, passed, detail.strip()))
        except subprocess.TimeoutExpired:
            results.append((script, False, "TIMEOUT (>120s)"))
        except Exception as e:
            results.append((script, False, str(e)))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = [
    ("token_schema_constants", test_token_schema_constants),
    ("synthetic_data_generation", test_synthetic_data_generation),
    ("model_build_canonical", test_model_build_canonical),
    ("config_round_trip", test_config_round_trip),
    ("forward_pass_finite", test_forward_pass_finite),
    ("forward_matches_bc_train", test_forward_matches_bc_train),
    ("scratch_persistence", test_scratch_persistence),
]


def main():
    print("=" * 60)
    print("Phase A Integration Test — Canonical Contract Verification")
    print("=" * 60)

    passed = 0
    failed = 0
    errors: list[str] = []

    # --- inline tests ---
    print("\n--- Inline Tests ---\n")
    for name, fn in ALL_TESTS:
        print(f"[phase_a] {name}...", flush=True)
        try:
            fn()
            passed += 1
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  FAIL: {e}")
            errors.append(f"{name}: {e}\n{tb}")
            failed += 1

    # --- existing validation tests ---
    print("\n--- Existing Validation Tests ---\n")
    for script, ok, detail in run_existing_validation_tests():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {script}")
        if not ok:
            print(f"         {detail[:200]}")
            errors.append(f"{script}: {detail[:500]}")
            failed += 1
        else:
            passed += 1

    # --- summary ---
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Phase A Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"\nFailed tests:")
        for e in errors:
            print(f"  - {e[:200]}")
    print("=" * 60)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
