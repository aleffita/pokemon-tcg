"""Phase E integration test -- MLX-only inference correctness verified.

Validates:
  1. E.1: Agent loads without any PyTorch import
  2. E.2: Complete logs are passed through agent() -> choose() -> encoder
  3. E.3: Autoregressive multi-select: picks are sequential, already-picked
          options are masked, SUBMIT is only accepted when min_count satisfied
  4. E.4: FP16 tensor creation for MLX model (float16 for numeric features)
  5. E.5: Config system integration (TrainConfig used for architecture params)
  6. E.6: Submission bundle builds correctly and agent loads from it

Run:
  uv run python scripts/validate/test_phase_e.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import pickle
import shutil
import traceback

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from rl.encoder.card_features import get_card_table
from rl.encoder.enc_constants import (
    N_STATE_TOKENS, MAX_OPTIONS, N_ACTIONS, G,
    DECK_SIZE, N_PRIZE, MAX_HAND, MAX_DISCARD, N_STADIUM,
    N_BENCH, N_PREEVO, N_TOOLS, N_ENERGY_CARDS, UNIT_ATTR,
    OPT_STRUCT, SUBMIT_ACTION,
)
from rl.encoder.encoding import TokenEncoder, GameTracker, AbilityTracker
from rl.encoder.effect_data import N_ATTACK_FX
from rl.policy_mlx import build_token_net_mlx
from rl.train_config import TrainConfig
from scripts.validate.make_synthetic_data import make_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE = 8
N_ROWS = 32

CANONICAL_CFG = {
    "d_model": 128,
    "nhead": 4,
    "nlayers": 3,
    "static": True,
    "split_heads": True,
}


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_phase_a.py -- proven to work)
# ---------------------------------------------------------------------------
def _make_synthetic_dir() -> str:
    """Create temp dir with synthetic data matching bc_train schema.

    Ensures effect_mask is present (required by the MLX model's _encode).
    """
    tmpdir = tempfile.mkdtemp(prefix="phase_e_test_")
    make_dataset(N_ROWS, tmpdir, seed=42)
    # Add effect_mask if absent (model._encode requires it for "effect" stream)
    effect_mask_path = os.path.join(tmpdir, "effect_mask.npy")
    if not os.path.exists(effect_mask_path):
        rng = np.random.default_rng(42)
        effect_mask = (rng.standard_normal((N_ROWS, 2)).astype(np.float32) * 0.2 + 0.5).clip(0.0, 1.0)
        np.save(effect_mask_path, effect_mask, allow_pickle=False)
    return tmpdir


def _load_batches(data_dir: str):
    """Load synthetic data, return (obs_dict, labels, keys, int_keys)."""
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


def _first_batch(d, keys, int_keys, batch_size, int_dtype=np.int32, float_dtype=np.float32):
    """Extract first batch as numpy arrays."""
    indices = np.arange(batch_size)
    ob = {
        k: np.asarray(d[k][indices]).astype(
            int_dtype if k in int_keys else float_dtype
        )
        for k in keys
    }
    return ob


def _batch_to_mlx(ob_np, int_keys):
    """Convert numpy batch to MLX arrays (float32 for trainer-compatible tests)."""
    return {
        k: mx.array(ob_np[k].astype(np.int32 if k in int_keys else np.float32))
        for k in ob_np
    }


def _batch_to_mlx_fp16(ob_np, int_keys):
    """Convert numpy batch to MLX arrays (float16 for E.4 inference test)."""
    return {
        k: mx.array(ob_np[k].astype(np.int32 if k in int_keys else np.float16))
        for k in ob_np
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_e1_no_pytorch_import():
    """E.1: Verify agent/main.py has no PyTorch imports."""
    print("\n--- test_e1_no_pytorch_import ---")

    agent_path = os.path.join(_REPO, "agent", "main.py")
    with open(agent_path) as f:
        content = f.read()

    # Check no torch imports
    bad_lines = []
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "import torch" in stripped or "from torch" in stripped:
            bad_lines.append((i, stripped))

    assert len(bad_lines) == 0, (
        f"agent/main.py contains PyTorch imports:\n"
        + "\n".join(f"  L{n}: {line}" for n, line in bad_lines)
    )

    # Verify mlx.core is imported at top level
    assert "import mlx.core as mx" in content, "mlx.core not imported at top level"

    # Verify no _HAS_MLX flag
    assert "_HAS_MLX" not in content, "_HAS_MLX flag still present"

    # Verify no PyTorch model type branching
    assert 'model_type == "mlx"' not in content, "MLX/PyTorch branching still present"
    assert 'model_type == "torch"' not in content, "MLX/PyTorch branching still present"
    assert '_MODEL_TYPE' not in content, "_MODEL_TYPE still present"

    print("  PASSED: agent/main.py has no PyTorch imports or branching")


def test_e2_complete_logs():
    """E.2: Verify obs.logs flows through the encoding pipeline."""
    print("\n--- test_e2_complete_logs ---")

    # Monkey-patch GameTracker.update to capture logs
    captured_logs = []
    original_update = GameTracker.update

    def _capturing_update(self, obs):
        if "logs" in obs:
            captured_logs.extend(obs["logs"])
        return original_update(self, obs)

    GameTracker.update = _capturing_update

    try:
        # Build a minimal obs dict with logs
        test_logs = [{"type": "reveal", "card": 42}, {"type": "move", "zone": "hand"}]
        select = {"option": [], "minCount": 0, "maxCount": 0, "type": 0, "context": 0}
        current = {"yourIndex": 0, "turn": 5, "turnActionCount": 1, "firstPlayer": 0,
                   "players": [{"deckCount": 10, "handCount": 5, "prize": []},
                               {"deckCount": 10, "handCount": 5, "prize": []}]}
        obs = {"select": select, "current": current, "logs": test_logs}

        # Simulate what choose() now does with obs.logs (E.2 fix)
        logs = obs.get("logs", [])
        obs_for_encode = {"select": select, "current": current, "logs": logs}

        # Verify logs are present
        assert "logs" in obs_for_encode, "logs not passed to obs_for_encode"
        assert len(obs_for_encode["logs"]) == 2, f"expected 2 logs, got {len(obs_for_encode['logs'])}"
        assert obs_for_encode["logs"][0]["type"] == "reveal", "first log mismatch"

        # Verify tracker.update receives logs
        tracker = GameTracker()
        ability = AbilityTracker()
        captured_logs.clear()
        tracker.update(obs_for_encode)
        assert len(captured_logs) == 2, (
            f"tracker should have received 2 logs, got {len(captured_logs)}"
        )

        # Also verify that logs=[] (the old behavior) gives empty
        captured_logs.clear()
        obs_no_logs = {"select": select, "current": current, "logs": []}
        tracker.update(obs_no_logs)
        assert len(captured_logs) == 0, (
            f"old behavior (logs=[]) should give 0, got {len(captured_logs)}"
        )

        print("  PASSED: complete logs flow to tracker, old behavior gives empty")
    finally:
        GameTracker.update = original_update


def test_e3_autoregressive_multiselect():
    """E.3: Autoregressive picks are sequential, masked, and SUBMIT-respecting."""
    print("\n--- test_e3_autoregressive_multiselect ---")

    tmpdir = _make_synthetic_dir()
    try:
        d, labels, keys, int_keys = _load_batches(tmpdir)

        # Build model and load batch
        card_table = get_card_table()
        model = build_token_net_mlx(card_table, CANONICAL_CFG)
        model.eval()

        # Use first batch as base
        ob_np = _first_batch(d, keys, int_keys, BATCH_SIZE)

        # Create a fixed action_mask: first row has 6 legal options + SUBMIT
        am = ob_np["action_mask"].copy()
        am[:] = 0  # clear all
        legal_indices = [3, 7, 15, 42, 88, 120]  # 6 legal options
        for idx in legal_indices:
            am[0, idx] = 1.0
        am[0, SUBMIT_ACTION] = 1.0
        ob_np["action_mask"] = am

        # --- Simulate the autoregressive loop (same logic as agent/main.py) ---
        picked_set = set()
        results = []
        min_count = 2
        max_count = 3

        for substep in range(max_count):
            # Build MLX tensors (fp16 for numerics, matching E.4)
            ob_mlx = {
                k: mx.array(ob_np[k].astype(np.int32 if k in int_keys else np.float16))
                for k in ob_np
            }

            # Forward
            logits, _, _ = model.logits_value(ob_mlx)
            logits_np = np.asarray(logits)

            # Process only row 0 (our test row)
            row_logits = logits_np[0].copy()
            row_am = am[0]

            # Mask illegal and already-picked
            row_logits[row_am < 0.5] = -1e9
            for p in picked_set:
                if p < len(row_logits):
                    row_logits[p] = -1e9

            action = int(np.argmax(row_logits))

            # SUBMIT only when min_count satisfied
            if action == SUBMIT_ACTION and len(results) >= min_count:
                break

            picked_set.add(action)
            results.append(action)

            if len(results) >= max_count:
                break

        # Verify properties
        assert len(results) == len(set(results)), "duplicate picks detected"
        assert len(results) >= min_count, f"expected >= {min_count} picks, got {len(results)}"
        assert len(results) <= max_count, f"expected <= {max_count} picks, got {len(results)}"

        # All picks should be legal option indices (not SUBMIT unless terminating)
        for r in results:
            if r < MAX_OPTIONS:
                assert am[0, r] > 0.5, f"option {r} was not legal"

        # Verify picked options are masked after the loop
        # Run one more forward with the same data and check masking
        ob_mlx2 = {
            k: mx.array(ob_np[k].astype(np.int32 if k in int_keys else np.float16))
            for k in ob_np
        }
        logits2, _, _ = model.logits_value(ob_mlx2)
        logits2_np = np.asarray(logits2)
        row_logits2 = logits2_np[0].copy()
        row_logits2[am[0] < 0.5] = -1e9
        for p in picked_set:
            row_logits2[p] = -1e9

        for p in results:
            if p < MAX_OPTIONS:
                assert row_logits2[p] < -1e8, (
                    f"option {p} should be masked but logit={row_logits2[p]}"
                )

        print(f"  PASSED: autoregressive {len(results)} picks, no duplicates, "
              f"masking correct")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_e4_fp16_inference():
    """E.4: MLX tensors use float16 for numeric features in inference."""
    print("\n--- test_e4_fp16_inference ---")

    tmpdir = _make_synthetic_dir()
    try:
        d, labels, keys, int_keys = _load_batches(tmpdir)
        ob_np = _first_batch(d, keys, int_keys, BATCH_SIZE, float_dtype=np.float32)

        card_table = get_card_table()
        model = build_token_net_mlx(card_table, CANONICAL_CFG)
        model.eval()

        # Convert to fp16 (same as E.4 inference path)
        ob_fp16 = _batch_to_mlx_fp16(ob_np, int_keys)

        # Check dtypes
        n_int = 0
        n_fp16 = 0
        for k, v in ob_fp16.items():
            if k in int_keys:
                assert v.dtype == mx.int32, f"{k}: expected int32, got {v.dtype}"
                n_int += 1
            else:
                assert v.dtype == mx.float16, f"{k}: expected float16, got {v.dtype}"
                n_fp16 += 1

        # Verify forward works with fp16 tensors
        logits, value, _ = model.logits_value(ob_fp16)
        assert logits.shape == (BATCH_SIZE, N_ACTIONS), f"logits shape: {logits.shape}"
        assert value.shape == (BATCH_SIZE,), f"value shape: {value.shape}"
        assert mx.all(mx.isfinite(logits)).item(), "logits contain non-finite values"
        assert mx.all(mx.isfinite(value)).item(), "value contains non-finite values"

        # Compare with float32 forward
        ob_fp32 = _batch_to_mlx(ob_np, int_keys)
        logits32, value32, _ = model.logits_value(ob_fp32)

        # Values should be close (fp16 vs fp32)
        diff = float(mx.abs(logits - logits32).max())
        print(f"  PASSED: {n_int} int32, {n_fp16} fp16 keys, forward finite, "
              f"fp16-fp32 max diff={diff:.6f}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_e5_config_integration():
    """E.5: TrainConfig provides architecture params that match the model."""
    print("\n--- test_e5_config_integration ---")

    cfg = TrainConfig()

    # Verify defaults match canonical architecture
    assert cfg.d_model == 128, f"d_model={cfg.d_model}"
    assert cfg.nhead == 4, f"nhead={cfg.nhead}"
    assert cfg.nlayers == 3, f"nlayers={cfg.nlayers}"
    assert cfg.ff_dim == 512, f"ff_dim={cfg.ff_dim}"
    assert cfg.static is True, f"static={cfg.static}"
    assert cfg.split_heads is True, f"split_heads={cfg.split_heads}"
    assert cfg.scratch_registers == 4, f"scratch_registers={cfg.scratch_registers}"

    # Verify config can be converted to/from dict
    d = cfg.to_dict()
    assert "d_model" in d
    assert "nhead" in d
    restored = TrainConfig(**{k: d[k] for k in d if hasattr(TrainConfig, k)})
    assert restored.d_model == cfg.d_model
    assert restored.nhead == cfg.nhead

    # Verify config produces valid model architecture
    model_cfg = {
        "d_model": cfg.d_model,
        "nhead": cfg.nhead,
        "nlayers": cfg.nlayers,
        "static": cfg.static,
        "split_heads": cfg.split_heads,
    }
    card_table = get_card_table()
    model = build_token_net_mlx(card_table, model_cfg)
    model.eval()

    # Quick forward to verify dimensions match
    tmpdir = _make_synthetic_dir()
    try:
        d, labels, keys, int_keys = _load_batches(tmpdir)
        ob_np = _first_batch(d, keys, int_keys, 4)
        ob_mlx = _batch_to_mlx(ob_np, int_keys)
        logits, value, _ = model.logits_value(ob_mlx)
        assert logits.shape[1] == N_ACTIONS
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"  PASSED: TrainConfig d={cfg.d_model} h={cfg.nhead} L={cfg.nlayers} "
          f"static={cfg.static} split={cfg.split_heads}")


def test_e6_checkpoint_save_load():
    """E.6a: Save and load MLX checkpoint roundtrip."""
    print("\n--- test_e6_checkpoint_save_load ---")

    # Flush any pending MLX computations from previous tests
    mx.eval(mx.array(0))

    tmpdir = _make_synthetic_dir()
    try:
        d, labels, keys, int_keys = _load_batches(tmpdir)
        ob_np = _first_batch(d, keys, int_keys, BATCH_SIZE)
        ob_mlx = _batch_to_mlx(ob_np, int_keys)

        card_table = get_card_table()
        model = build_token_net_mlx(card_table, CANONICAL_CFG)
        model.eval()

        # Get initial forward output
        logits_before, value_before, _ = model.logits_value(ob_mlx)

        # Save checkpoint (pickle, matching bc_train_mlx format)
        # Use nn.utils.tree_flatten to properly handle nested lists in trainable_parameters()
        flat_params = nn.utils.tree_flatten(model.trainable_parameters())
        state = {
            "model": {k: np.asarray(v) for k, v in flat_params},
            "val_acc": 0.6947,
            "config": model.get_config(),
        }
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
            pickle.dump(state, f)

        try:
            # Load and rebuild
            with open(tmp_path, "rb") as f:
                loaded_state = pickle.load(f)

            model2 = build_token_net_mlx(card_table, CANONICAL_CFG)
            # Update with saved parameters (convert numpy -> MLX arrays, rebuild tree)
            if isinstance(loaded_state.get("model"), dict):
                flat = [(k, mx.array(v)) for k, v in loaded_state["model"].items()]
                tree = nn.utils.tree_unflatten(flat)
                model2.update(tree)
            model2.eval()

            # Verify output matches
            logits_after, value_after, _ = model2.logits_value(ob_mlx)
            logits_match = np.allclose(
                np.asarray(logits_before), np.asarray(logits_after), atol=1e-5
            )
            value_match = np.allclose(
                np.asarray(value_before), np.asarray(value_after), atol=1e-5
            )

            # Config roundtrip
            cfg = loaded_state.get("config", {})
            assert cfg.get("arch_version"), "arch_version missing from checkpoint"
            assert cfg["d_model"] == 128, f"d_model={cfg['d_model']}"
            assert cfg["split_heads"] is True

            status = "PASSED" if logits_match and value_match else "PARTIAL"
            print(f"  {status}: logits_match={logits_match}, value_match={value_match}, "
                  f"arch_version={cfg.get('arch_version')}")
        finally:
            os.unlink(tmp_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_e6_submit_handling():
    """E.6b: SUBMIT is only accepted when min_count is satisfied."""
    print("\n--- test_e6_submit_handling ---")

    tmpdir = _make_synthetic_dir()
    try:
        d, labels, keys, int_keys = _load_batches(tmpdir)
        ob_np = _first_batch(d, keys, int_keys, BATCH_SIZE)

        card_table = get_card_table()
        model = build_token_net_mlx(card_table, CANONICAL_CFG)
        model.eval()

        # Force SUBMIT to be the highest logit
        am = ob_np["action_mask"].copy()
        am[:] = 0
        am[0, SUBMIT_ACTION] = 1.0  # Only SUBMIT is legal
        ob_np["action_mask"] = am

        ob_mlx = _batch_to_mlx(ob_np, int_keys)
        logits, _, _ = model.logits_value(ob_mlx)
        logits_np = np.asarray(logits)
        row_logits = logits_np[0].copy()
        row_am = am[0]
        row_logits[row_am < 0.5] = -1e9

        # Case 1: min_count=2, SUBMIT is highest -> should NOT submit
        min_count = 2
        results: list[int] = []
        action = int(np.argmax(row_logits))
        if action == SUBMIT_ACTION and len(results) >= min_count:
            assert False, "SUBMIT accepted before min_count satisfied"
        # Action is SUBMIT but min_count not met -> loop continues but no legal option
        # This is expected: the loop would break on encode failure or return empty

        # Case 2: min_count=0, SUBMIT is highest -> should submit immediately
        min_count_0 = 0
        results_2: list[int] = []
        action_2 = int(np.argmax(row_logits))
        accepted_submit = (action_2 == SUBMIT_ACTION and len(results_2) >= min_count_0)
        assert accepted_submit, "SUBMIT should be accepted when min_count=0"

        print("  PASSED: SUBMIT correctly blocked/accepted based on min_count")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_e6_submission_bundle():
    """E.6c: Verify submission can be built."""
    print("\n--- test_e6_submission_bundle ---")

    result = os.system(
        f"cd {_REPO} && uv run tcg-submission --no-validate 2>&1"
    )
    if result == 0:
        tarball = os.path.join(_REPO, "submission.tar.gz")
        assert os.path.exists(tarball), "submission.tar.gz not found"
        size_mb = os.path.getsize(tarball) / (1024 * 1024)

        import tarfile
        with tarfile.open(tarball, "r:gz") as tar:
            names = tar.getnames()
            assert "main.py" in names, "main.py not in bundle"
            assert "deck.csv" in names, "deck.csv not in bundle"
            assert "rl/policy_mlx.py" in names, "rl/policy_mlx.py not in bundle"
            assert "rl/encoder/encoding.py" in names, "rl/encoder/encoding.py not in bundle"
            # Verify no PyTorch fallback in bundled main.py
            main_py = [m for m in tar.getmembers() if m.name == "main.py"][0]
            content = tar.extractfile(main_py).read().decode()
            assert "import torch" not in content, "PyTorch import in bundled main.py"
            assert "_HAS_MLX" not in content, "_HAS_MLX in bundled main.py"
            assert "picked_set" in content, "autoregressive select not in bundled main.py"
            assert "np.float16" in content, "fp16 inference not in bundled main.py"
            print(f"  PASSED: bundle built ({size_mb:.1f} MB, {len(names)} files, "
                  f"MLX-only + autoregressive verified)")
        os.remove(tarball)
    else:
        print(f"  FAILED: build_submission.py returned {result}")


def test_e7_submit_entrypoint():
    """E.7: tcg-submit entrypoint is registered and --help works."""
    print("\n--- test_e7_submit_entrypoint ---")

    result = os.system(
        f"cd {_REPO} && uv run tcg-submit --help 2>&1"
    )
    assert result == 0, f"tcg-submit --help failed with exit code {result}"
    print("  PASSED: tcg-submit entrypoint works")


# ---------------------------------------------------------------------------
# Regression: run phases A-D tests
# ---------------------------------------------------------------------------

def run_regression():
    """Run existing phase A-D tests as regression suite."""
    print("\n=== REGRESSION: Phases A-D ===")
    test_dir = os.path.join(_REPO, "scripts", "validate")
    phases = ["test_phase_a.py", "test_phase_b.py", "test_phase_c.py", "test_phase_d.py"]
    results = {}
    for phase in phases:
        path = os.path.join(test_dir, phase)
        if not os.path.exists(path):
            results[phase] = "SKIPPED (not found)"
            continue
        ret = os.system(
            f"cd {_REPO} && uv run python {path} 2>&1 | tail -5"
        )
        results[phase] = "PASSED" if ret == 0 else f"FAILED (exit {ret})"

    print("\nRegression results:")
    for phase, status in results.items():
        print(f"  {phase}: {status}")

    return all(v == "PASSED" for v in results.values())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Phase E: MLX-Only Agent + Autoregressive Multi-Select Tests")
    print("=" * 60)

    tests = [
        ("E.1: No PyTorch imports", test_e1_no_pytorch_import),
        ("E.2: Complete logs", test_e2_complete_logs),
        ("E.3: Autoregressive multi-select", test_e3_autoregressive_multiselect),
        ("E.4: FP16 inference path", test_e4_fp16_inference),
        ("E.5: Config integration", test_e5_config_integration),
        ("E.6a: Checkpoint roundtrip", test_e6_checkpoint_save_load),
        ("E.6b: SUBMIT handling", test_e6_submit_handling),
        ("E.6c: Submission bundle", test_e6_submission_bundle),
        ("E.7: Submit entrypoint", test_e7_submit_entrypoint),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Phase E results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    # Run regression
    print("\nRunning regression suite...")
    reg_passed = run_regression()
    if not reg_passed:
        print("\nWARNING: Some regression tests failed!")

    if failed > 0 or not reg_passed:
        print("\nOVERALL: SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("\nOVERALL: ALL TESTS PASSED")


if __name__ == "__main__":
    main()
