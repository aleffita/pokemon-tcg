"""Checkpoint round-trip validation test for architecture config versioning.

Verifies:
1. get_config() returns all expected fields
2. Checkpoint save/load preserves all fields
3. Config mismatch detection works
4. Backward compat: old checkpoint without arch_config loads with warning

Run:
  uv run python scripts/validate/test_checkpoint.py
"""
from __future__ import annotations

import os
import pickle
import tempfile

# Force headless MLX (no Metal GPU needed for test)
os.environ["MLX_BACKEND"] = "cpu"

import numpy as np
import mlx.core as mx

from rl.encoder.card_features import get_card_table
from rl.policy_mlx import build_token_net_mlx, TokenTransformerMLX
from rl.token_schema import ARCH_VERSION, TOKEN_SCHEMA_VERSION


def _build_model() -> TokenTransformerMLX:
    """Build a small test model with static features."""
    ct = get_card_table()
    cfg = {
        "arch": "transformer2",
        "d_model": 128,
        "nhead": 4,
        "nlayers": 3,
        "ff": 512,
        "static": True,
        "split_heads": True,
        "structured": False,
    }
    return build_token_net_mlx(ct, cfg)


def _get_config() -> dict:
    """Expected default config for the test model."""
    return {
        "arch_version": ARCH_VERSION,
        "token_schema_version": TOKEN_SCHEMA_VERSION,
        "d_model": 128,
        "nhead": 4,
        "nlayers": 3,
        "ff_dim": 512,
        "scratch_registers": 16,
        "static": True,
        "split_heads": True,
        "structured": False,
        "max_options": 192,
        "value_categorical": False,
        "value_atoms": 51,
        "value_vmax": 1.0,
        "has_learned_init": True,
        "dtype": "mlx.core.float32",
    }


def test_get_config_fields():
    """Test that get_config() returns all expected fields with correct values."""
    model = _build_model()
    cfg = model.get_config()
    expected = _get_config()

    missing = set(expected.keys()) - set(cfg.keys())
    extra = set(cfg.keys()) - set(expected.keys())
    assert not missing, f"Missing keys in config: {missing}"
    assert not extra, f"Unexpected keys in config: {extra}"

    for k in expected:
        assert cfg[k] == expected[k], (
            f"Config field '{k}' mismatch: got {cfg[k]!r}, expected {expected[k]!r}"
        )

    # Version strings must be non-empty strings
    assert isinstance(cfg["arch_version"], str) and cfg["arch_version"]
    assert isinstance(cfg["token_schema_version"], str) and cfg["token_schema_version"]

    print("  PASS: get_config() fields match expected values")


def test_round_trip():
    """Test that save -> load preserves all checkpoint fields."""
    model = _build_model()
    orig_cfg = model.get_config()
    orig_params = model.parameters()
    # Snapshot scratch as numpy before round-trip
    orig_scratch_np = np.array(model.scratch)

    checkpoint = {
        "model": orig_params,
        "arch_config": orig_cfg,
        "epoch": 5,
        "gstep": 1234,
        "val_acc": 0.7543,
        "seed": 42,
        "dataset_path": "/some/path/bc_data",
    }

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name
        pickle.dump(checkpoint, tmp)

    try:
        with open(tmp_path, "rb") as f:
            loaded = pickle.load(f)

        # Verify all top-level keys
        assert set(loaded.keys()) == set(checkpoint.keys()), (
            f"Key mismatch: {set(loaded.keys()) ^ set(checkpoint.keys())}"
        )

        # Verify scalar fields
        assert loaded["epoch"] == 5
        assert loaded["gstep"] == 1234
        assert abs(loaded["val_acc"] - 0.7543) < 1e-6
        assert loaded["seed"] == 42
        assert loaded["dataset_path"] == "/some/path/bc_data"

        # Verify arch_config round-trips exactly
        loaded_cfg = loaded["arch_config"]
        assert loaded_cfg == orig_cfg, (
            f"arch_config changed after round-trip: {loaded_cfg}"
        )

        # Verify model params survive round-trip (structure, not exact values)
        model2 = _build_model()
        model2.update(loaded["model"])

        # Check that scratch weights match after round-trip
        new_scratch_np = np.array(model2.scratch)
        assert np.abs(orig_scratch_np - new_scratch_np).max() < 1e-5, (
            "Scratch weights differ after round-trip"
        )

        print("  PASS: checkpoint round-trip preserves all fields")
    finally:
        os.unlink(tmp_path)


def test_config_mismatch_detection():
    """Test that arch_config mismatch is detected and reported."""
    model = _build_model()
    orig_cfg = model.get_config()

    # Tamper with one field
    tampered_cfg = dict(orig_cfg)
    tampered_cfg["d_model"] = 256  # wrong!

    checkpoint = {
        "model": model.parameters(),
        "arch_config": tampered_cfg,
        "epoch": 0,
        "gstep": 0,
        "val_acc": 0.0,
    }

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name
        pickle.dump(checkpoint, tmp)

    try:
        with open(tmp_path, "rb") as f:
            loaded = pickle.load(f)

        saved_cfg = loaded["arch_config"]
        cur_cfg = model.get_config()
        mismatches = []
        for k, v in saved_cfg.items():
            if k in cur_cfg and cur_cfg[k] != v:
                mismatches.append(f"{k}: saved={v} current={cur_cfg[k]}")

        assert len(mismatches) > 0, "Should have detected mismatch but didn't"
        assert "d_model" in mismatches[0], (
            f"Expected d_model mismatch, got: {mismatches}"
        )

        print("  PASS: config mismatch detection works")
    finally:
        os.unlink(tmp_path)


def test_backward_compat_old_checkpoint():
    """Test that old checkpoint without arch_config loads without error."""
    model = _build_model()

    # Simulate old checkpoint format (no arch_config, no gstep)
    old_checkpoint = {
        "model": model.parameters(),
        "epoch": 2,
        "val_acc": 0.6882,
    }

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name
        pickle.dump(old_checkpoint, tmp)

    try:
        with open(tmp_path, "rb") as f:
            loaded = pickle.load(f)

        # Simulate the resume logic
        saved_cfg = loaded.get("arch_config")
        gstep_restored = int(loaded.get("gstep", 0))

        assert saved_cfg is None, "Old checkpoint should not have arch_config"
        assert gstep_restored == 0, "Old checkpoint should default gstep to 0"

        # Should not crash when loading model params
        model_params = loaded["model"]
        model2 = _build_model()
        model2.update(model_params)

        print("  PASS: old checkpoint backward compat works")
    finally:
        os.unlink(tmp_path)


def test_value_categorical_config():
    """Test get_config() correctly reports value_categorical."""
    ct = get_card_table()
    cfg_cat = {
        "arch": "transformer2",
        "d_model": 128,
        "nhead": 4,
        "nlayers": 2,
        "ff": 256,
        "static": False,
        "split_heads": False,
        "value_categorical": True,
        "value_atoms": 51,
    }
    model_cat = build_token_net_mlx(ct, cfg_cat)
    cfg = model_cat.get_config()

    assert cfg["value_categorical"] is True, (
        f"Expected value_categorical=True, got {cfg['value_categorical']}"
    )

    cfg_nocat = {
        "arch": "transformer2",
        "d_model": 128,
        "nhead": 4,
        "nlayers": 2,
        "ff": 256,
        "static": False,
        "split_heads": False,
    }
    model_nocat = build_token_net_mlx(ct, cfg_nocat)
    cfg2 = model_nocat.get_config()

    assert cfg2["value_categorical"] is False, (
        f"Expected value_categorical=False, got {cfg2['value_categorical']}"
    )

    print("  PASS: value_categorical config correct for both models")


def main():
    tests = [
        ("get_config fields", test_get_config_fields),
        ("checkpoint round-trip", test_round_trip),
        ("config mismatch detection", test_config_mismatch_detection),
        ("backward compat (old checkpoint)", test_backward_compat_old_checkpoint),
        ("value_categorical config", test_value_categorical_config),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"[test_checkpoint] {name}...", flush=True)
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n[test_checkpoint] {passed}/{passed + failed} tests passed", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
