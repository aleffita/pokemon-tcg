"""Package agent/ into submission.tar.gz for the PTCG AI Battle Challenge.

The competition requires main.py and deck.csv at the root of the archive.
This script also bundles:
  - rl/ (encoder, policy, etc. — needed by main.py)
  - the MLX checkpoint the agent actually loads (bc_best_mlx_final.pkl)
  - _vendor/ (MLX unpacked from wheels — the sandbox image has no mlx)

The cg/ SDK is NOT bundled — it's already in the Kaggle sandbox via kaggle_environments.

Usage:
    python scripts/build_submission.py            # -> submission.tar.gz
    python scripts/build_submission.py -o out.tar.gz
"""

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_DIR = os.path.join(ROOT, "agent")
RL_DIR = os.path.join(ROOT, "rl")
MODEL_DIR = os.path.join(ROOT, "model")
WHEELS_DIR = os.path.join(ROOT, "vendor", "wheels")
REQUIRED = ["main.py", "deck.csv"]

# Checkpoint candidates, in the order agent/main.py searches for them. The
# archive name must match one of those paths or the agent silently falls back
# to a random policy.
CHECKPOINT_CANDIDATES = [
    (os.path.join(MODEL_DIR, "bc_model", "bc_best_mlx_final.pkl"),
     "model/bc_model/bc_best_mlx_final.pkl"),
    (os.path.join(MODEL_DIR, "checkpoint", "bc_best_mlx.pkl"),
     "model/checkpoint/bc_best_mlx.pkl"),
]

# MLX is absent from the Kaggle image (see kaggle_requirements.txt in
# Kaggle/docker-python), so it ships inside the bundle. Both wheels unpack into
# the same directory: core.cpython-312-*.so has RUNPATH $ORIGIN/lib pointing at
# libmlx.so, which in turn has RUNPATH $ORIGIN/../../mlx_cpu.libs for BLAS.
# Wheels target the sandbox: Python 3.12, linux x86_64, glibc >= 2.35.
VENDOR_WHEELS = [
    "mlx-0.32.0-cp312-cp312-manylinux_2_35_x86_64.whl",
    "mlx_cpu-0.32.0-py3-none-manylinux_2_35_x86_64.whl",
]


def _collect_files(base_dir: str, prefix: str) -> list[tuple[str, str]]:
    """Walk a directory and return (abs_path, archive_name) pairs."""
    files = []
    for dirpath, _dirs, names in os.walk(base_dir):
        if "__pycache__" in dirpath:
            continue
        for name in names:
            if name.endswith(".pyc") or name == ".DS_Store":
                continue
            full = os.path.join(dirpath, name)
            arc = os.path.join(prefix, os.path.relpath(full, base_dir)).replace(os.sep, "/")
            files.append((full, arc))
    return files


def _unpack_vendor_wheels(dest: str) -> list[tuple[str, str]]:
    """Unpack the MLX wheels into `dest` and return (abs_path, archive_name) pairs.

    Both wheels extract into the same root so the RUNPATH chain
    (core.so -> mlx/lib/libmlx.so -> mlx_cpu.libs/) resolves inside the bundle.
    dist-info metadata is skipped: nothing imports it at runtime.
    """
    missing = [w for w in VENDOR_WHEELS if not os.path.exists(os.path.join(WHEELS_DIR, w))]
    if missing:
        raise SystemExit(
            f"ERROR: missing vendored wheel(s) in {WHEELS_DIR}: {missing}\n"
            "The Kaggle image has no MLX; the agent cannot import without them."
        )

    for wheel in VENDOR_WHEELS:
        with zipfile.ZipFile(os.path.join(WHEELS_DIR, wheel)) as zf:
            for member in zf.namelist():
                if ".dist-info/" in member or member.endswith("/"):
                    continue
                zf.extract(member, dest)

    files = _collect_files(dest, "_vendor")
    # The .so files lose their executable bit through extract(); harmless for
    # dlopen, but keep the mode tidy so the archive mirrors a real install.
    for full, _arc in files:
        if full.endswith(".so") or ".so." in os.path.basename(full):
            os.chmod(full, 0o755)
    return files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--out", default=os.path.join(ROOT, "submission.tar.gz"))
    p.add_argument("--no-validate", action="store_true", help="Skip agent validation")
    args = p.parse_args()

    files = []

    # 1. agent/ contents at top level (main.py, deck.csv)
    for full, arc in _collect_files(AGENT_DIR, ""):
        files.append((full, arc))

    # 2. rl/ directory (encoder, policy, lr_schedule, etc.)
    if os.path.isdir(RL_DIR):
        for full, arc in _collect_files(RL_DIR, "rl"):
            files.append((full, arc))
    else:
        print("WARNING: rl/ not found — submission may fail on Kaggle")

    # 3. MLX checkpoint, at the path agent/main.py searches for it
    for src, arc in CHECKPOINT_CANDIDATES:
        if os.path.exists(src):
            files.append((src, arc))
            print(f"Checkpoint: {arc}")
            break
    else:
        raise SystemExit(
            "ERROR: no MLX checkpoint found. Looked for:\n  "
            + "\n  ".join(src for src, _ in CHECKPOINT_CANDIDATES)
            + "\nWithout it the agent falls back to a random policy."
        )

    # Validate required files
    arcnames = {arc for _, arc in files}
    missing = [r for r in REQUIRED if r not in arcnames]
    if missing:
        raise SystemExit(f"ERROR: missing required file(s) at top level: {missing}")

    # Validate deck
    with open(os.path.join(AGENT_DIR, "deck.csv")) as f:
        n = len([ln for ln in f if ln.strip()])
    if n != 60:
        raise SystemExit(f"ERROR: deck.csv has {n} cards, expected 60.")

    # 4. EN_Card_Data.csv (needed by card_features.py at runtime)
    csv_path = os.path.join(ROOT, "EN_Card_Data.csv")
    if os.path.exists(csv_path):
        files.append((csv_path, "EN_Card_Data.csv"))
    else:
        print("WARNING: EN_Card_Data.csv not found — agent will fail at runtime")

    # 5. Vendored MLX + tarball, both inside the staging dir's lifetime
    staging = tempfile.mkdtemp(prefix="ptcg_vendor_")
    try:
        vendor_files = _unpack_vendor_wheels(staging)
        files.extend(vendor_files)
        print(f"Vendored MLX: {len(vendor_files)} files from {len(VENDOR_WHEELS)} wheels")

        with tarfile.open(args.out, "w:gz") as tar:
            for full, arc in sorted(files, key=lambda x: x[1]):
                tar.add(full, arcname=arc)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"Wrote {args.out} ({size_mb:.1f} MB)")
    n_vendor = sum(1 for _, arc in files if arc.startswith("_vendor/"))
    print(f"Contents ({len(files)} files, {n_vendor} vendored):")
    for _, arc in sorted(files, key=lambda x: x[1]):
        if not arc.startswith("_vendor/"):
            print(f"  {arc}")
    print(f"  _vendor/... ({n_vendor} files)")

    if size_mb > 197.7:
        raise SystemExit(f"\nERROR: submission is {size_mb:.1f} MB (limit is 197.7 MB)")

    if not args.no_validate:
        _validate_archive(args.out)


def _validate_archive(tar_path: str) -> None:
    """Check the archive's structure, then run the agent from the extracted copy.

    Extracting first is what makes this meaningful: it exercises the same flat
    layout Kaggle unpacks into /kaggle_simulations/agent/, so a path bug shows
    up here instead of on the ladder.

    The vendored MLX cannot be exercised locally — those wheels are linux
    x86_64 / cp312 binaries. This checks that they are present and correctly
    laid out; the agent itself runs against the development MLX.
    """
    import traceback

    print("\nValidating archive...")
    staging = tempfile.mkdtemp(prefix="ptcg_validate_")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(staging)

        # Structural checks: paths the agent resolves at import time.
        expected = [
            "main.py",
            "deck.csv",
            "EN_Card_Data.csv",
            "rl/encoder/encoding.py",
            "rl/policy_mlx.py",
            # RUNPATH chain: core.so -> mlx/lib/libmlx.so -> mlx_cpu.libs/
            "_vendor/mlx/core.cpython-312-x86_64-linux-gnu.so",
            "_vendor/mlx/lib/libmlx.so",
            "_vendor/mlx/nn/__init__.py",
        ]
        for rel in expected:
            if not os.path.exists(os.path.join(staging, rel)):
                raise SystemExit(f"  ERROR: archive is missing {rel}")
        if not os.path.isdir(os.path.join(staging, "_vendor", "mlx_cpu.libs")):
            raise SystemExit("  ERROR: archive is missing _vendor/mlx_cpu.libs/")
        print(f"  OK: {len(expected) + 1} required paths present")

        checkpoints = [
            arc for _, arc in CHECKPOINT_CANDIDATES
            if os.path.exists(os.path.join(staging, arc))
        ]
        if not checkpoints:
            raise SystemExit("  ERROR: archive has no checkpoint at a path main.py searches")
        print(f"  OK: checkpoint at {checkpoints[0]}")

        # Behavioural check: import and play one decision from the extracted copy.
        old_cwd = os.getcwd()
        old_path = list(sys.path)
        old_modules = set(sys.modules)
        try:
            os.chdir(staging)
            sys.path.insert(0, staging)
            for name in [m for m in sys.modules if m == "rl" or m.startswith("rl.")]:
                del sys.modules[name]

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "ptcg_submission_check", os.path.join(staging, "main.py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if module._MODEL_PATH is None:
                raise SystemExit("  ERROR: agent found no checkpoint and would play randomly")
            if module._LOADED_MODEL is None:
                raise SystemExit("  ERROR: checkpoint present but model failed to load")

            result = module.agent({"select": None})
            if not (isinstance(result, list) and len(result) == 60):
                raise SystemExit(f"  ERROR: deck submission returned {type(result)} len="
                                 f"{len(result) if hasattr(result, '__len__') else '?'}")
            print(f"  OK: model loaded and agent returned {len(result)} cards")
        finally:
            os.chdir(old_cwd)
            sys.path[:] = old_path
            for name in set(sys.modules) - old_modules:
                del sys.modules[name]
    except SystemExit:
        raise
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        raise SystemExit("  Submission would fail on Kaggle.")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
