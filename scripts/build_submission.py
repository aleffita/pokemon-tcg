"""Package agent/ into submission.tar.gz for the PTCG AI Battle Challenge.

The competition requires main.py and deck.csv at the root of the archive.
This script also bundles:
  - rl/ (encoder, policy, etc. — needed by main.py)
  - an MLX checkpoint converted strictly to PyTorch FP32, or an already
    validated PyTorch FP32 autoresearch candidate

The transient training JSON is deliberately excluded. Architecture, encoder
schema, would-KO inference settings, and training provenance travel inside the
converted model artifact.

The cg/ SDK is NOT bundled — it's already in the Kaggle sandbox via kaggle_environments.

This is the packaging implementation, driven by `scripts/submit.py`. Use the
entry point rather than calling it directly:

    uv run tcg-build --checkpoint experiments/autoresearch/<run>/candidate.pt
    uv run tcg-build --checkpoint experiments/autoresearch/<run>/candidate.pt --upload
"""

import argparse
import os
import shutil
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_DIR = os.path.join(ROOT, "agent")
RL_DIR = os.path.join(ROOT, "rl")
SUBMISSION_WORK_DIR = os.path.join(ROOT, "model", "submission_work")
REQUIRED = ["main.py", "deck.csv"]
SMOKE_CHECKPOINT_DIR = os.path.join(ROOT, "model", "checkpoint", "smoke")
SMOKE_TORCH_CHECKPOINT = os.path.join(
    ROOT, "model", "bc_model", "smoke", "bc_smoke_torch_fp32.pt"
)
MAIN_TORCH_CHECKPOINT = os.path.join(
    ROOT, "model", "bc_model", "bc_best_torch_fp32.pt"
)
SMOKE_SUBMISSION = os.path.join(
    ROOT,
    "public_agents",
    "submissions",
    "smoke",
    "submission_smoke.tar.gz",
)

# Checkpoint source paths are build-time conventions only. Runtime settings are
# embedded in the selected checkpoint, never read from train_config.json.
SOURCE_CHECKPOINT_CANDIDATES = [
    (os.path.join(ROOT, "model", "bc_model", "bc_best_mlx_final.pkl"),
     "bc_best_mlx_final.pkl"),
    (os.path.join(ROOT, "model", "checkpoint", "bc_best_mlx.pkl"),
     "bc_best_mlx.pkl"),
]
TORCH_CHECKPOINT_ARC = os.path.join(
    "model", "bc_model", "bc_best_torch_fp32.pt"
).replace(os.sep, "/")


def _is_within(path: str, directory: str) -> bool:
    return os.path.commonpath(
        (os.path.realpath(path), os.path.realpath(directory))
    ) == os.path.realpath(directory)


def _resolve_smoke_checkpoint(explicit: str | None) -> str:
    """Resolve one non-epoch smoke checkpoint without touching main models."""

    if explicit is not None:
        checkpoint = os.path.abspath(explicit)
        if not _is_within(checkpoint, SMOKE_CHECKPOINT_DIR):
            raise SystemExit(
                "ERROR: --smoke checkpoint must be inside "
                f"{SMOKE_CHECKPOINT_DIR}: {checkpoint}"
            )
        if not os.path.isfile(checkpoint):
            raise SystemExit(
                f"ERROR: explicit smoke checkpoint does not exist: {checkpoint}"
            )
        return checkpoint

    if not os.path.isdir(SMOKE_CHECKPOINT_DIR):
        raise SystemExit(
            f"ERROR: smoke checkpoint directory does not exist: "
            f"{SMOKE_CHECKPOINT_DIR}"
        )
    candidates = [
        os.path.join(SMOKE_CHECKPOINT_DIR, name)
        for name in sorted(os.listdir(SMOKE_CHECKPOINT_DIR))
        if name.endswith(".pkl")
        and "_epoch_" not in name
        and not name.endswith("_latest.pkl")
    ]
    if len(candidates) != 1:
        detail = "\n  ".join(candidates) if candidates else "(none)"
        raise SystemExit(
            "ERROR: --smoke requires exactly one canonical non-epoch MLX "
            f"checkpoint in {SMOKE_CHECKPOINT_DIR}; found:\n  {detail}"
        )
    return candidates[0]


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


def _fresh_project_work_dir(name: str) -> str:
    """Create one clean, repository-local build directory."""

    path = os.path.join(SUBMISSION_WORK_DIR, name)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return path


def _materialize_submission_checkpoint(
    source_checkpoint: str,
    building: str,
    card_table,
) -> tuple[dict, str]:
    """Create the exact portable FP32 artifact that will enter the archive."""
    from rl.policy_infer_torch import (
        load_torch_inference_checkpoint,
        save_torch_inference_checkpoint,
    )

    if source_checkpoint.endswith(".pt"):
        _model, metadata = load_torch_inference_checkpoint(source_checkpoint, card_table)
        shutil.copyfile(source_checkpoint, building)
        return metadata, "validated PyTorch FP32"
    metadata = save_torch_inference_checkpoint(source_checkpoint, building, card_table)
    return metadata, "converted MLX -> PyTorch FP32"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--out", default=None)
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Use only model/checkpoint/smoke and default to the isolated "
            "public_agents/submissions/smoke/submission_smoke.tar.gz"
        ),
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Explicit MLX trainer checkpoint to convert or PyTorch FP32 candidate "
            "to validate and package. When omitted, use the authoritative "
            "model/checkpoint directory candidates."
        ),
    )
    p.add_argument("--no-validate", action="store_true", help="Skip agent validation")
    args = p.parse_args()
    args.out = args.out or (
        SMOKE_SUBMISSION
        if args.smoke
        else os.path.join(ROOT, "submission.tar.gz")
    )

    files = []

    # 1. agent/ contents at top level (main.py, deck.csv)
    for full, arc in _collect_files(AGENT_DIR, ""):
        files.append((full, arc))

    # 2. rl/ directory (encoder, policy, lr_schedule, etc.)
    if os.path.isdir(RL_DIR):
        for full, arc in _collect_files(RL_DIR, "rl"):
            # Training remains MLX, but the submitted inference runtime is
            # PyTorch-only and must not import or ship the MLX policy.
            if arc not in (
                "rl/policy_mlx.py",
            ):
                files.append((full, arc))
    else:
        print("WARNING: rl/ not found — submission may fail on Kaggle")

    # 3. Resolve the MLX trainer checkpoint that will be converted below.
    source_checkpoint = None
    if args.smoke:
        source_checkpoint = _resolve_smoke_checkpoint(args.checkpoint)
    elif args.checkpoint:
        source_checkpoint = os.path.abspath(args.checkpoint)
        if not os.path.isfile(source_checkpoint):
            raise SystemExit(
                f"ERROR: explicit MLX checkpoint does not exist: {source_checkpoint}"
            )
    else:
        existing_candidates = []
        for src, _name in SOURCE_CHECKPOINT_CANDIDATES:
            if os.path.exists(src):
                existing_candidates.append(src)
        if len(existing_candidates) > 1:
            raise SystemExit(
                "ERROR: multiple checkpoint candidates exist; refusing to "
                "guess which model to submit. Pass --checkpoint explicitly:\n  "
                + "\n  ".join(existing_candidates)
            )
        if existing_candidates:
            source_checkpoint = existing_candidates[0]
    if source_checkpoint is None:
        raise SystemExit(
            "ERROR: no MLX checkpoint found. Looked for:\n  "
            + "\n  ".join(src for src, _ in SOURCE_CHECKPOINT_CANDIDATES)
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

    # 5. Convert into the canonical project model directory, then package that
    # exact artifact. The smoke and full paths mirror the real submission flow.
    from rl.encoder.card_features import get_card_table

    converted = (
        SMOKE_TORCH_CHECKPOINT if args.smoke else MAIN_TORCH_CHECKPOINT
    )
    os.makedirs(os.path.dirname(converted), exist_ok=True)
    building = converted + ".building"
    try:
        cfg, checkpoint_action = _materialize_submission_checkpoint(
            source_checkpoint,
            building,
            get_card_table(),
        )
        os.replace(building, converted)
    finally:
        if os.path.exists(building):
            os.unlink(building)
    files.append((converted, TORCH_CHECKPOINT_ARC))
    print(
        f"Checkpoint: {os.path.relpath(source_checkpoint, ROOT)} -> "
        f"{TORCH_CHECKPOINT_ARC} ({checkpoint_action}, nlayers={cfg['nlayers']}, "
        f"scratch={cfg['scratch_registers']})"
    )

    output_parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(output_parent, exist_ok=True)
    with tarfile.open(args.out, "w:gz") as tar:
        for full, arc in sorted(files, key=lambda x: x[1]):
            tar.add(full, arcname=arc)

    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"Wrote {args.out} ({size_mb:.1f} MB)")
    print(f"Contents ({len(files)} files):")
    for _, arc in sorted(files, key=lambda x: x[1]):
        print(f"  {arc}")

    if size_mb > 197.7:
        raise SystemExit(f"\nERROR: submission is {size_mb:.1f} MB (limit is 197.7 MB)")

    if not args.no_validate:
        _validate_archive(args.out)


def _validate_archive(tar_path: str) -> None:
    """Check the archive's structure, then run the agent from the extracted copy.

    Extracting first is what makes this meaningful: it exercises the same flat
    layout Kaggle unpacks into /kaggle_simulations/agent/, so a path bug shows
    up here instead of on the ladder.

    The behavioural check loads the pre-converted PyTorch FP32 artifact from
    the same flat layout used by the Kaggle sandbox.
    """
    import traceback

    print("\nValidating archive...")
    archive_label = os.path.basename(tar_path).removesuffix(".tar.gz")
    staging = _fresh_project_work_dir(f"validate_{archive_label}")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            archive_names = tar.getnames()
            wheels = [name for name in archive_names if name.endswith(".whl")]
            if wheels:
                raise SystemExit(
                    "  ERROR: PyTorch-only archive unexpectedly contains "
                    f"vendored wheels: {wheels}"
                )
            tar.extractall(staging)

        # Structural checks: paths the agent resolves at import time.
        expected = [
            "main.py",
            "deck.csv",
            "EN_Card_Data.csv",
            "rl/encoder/encoding.py",
            "rl/policy.py",
            "rl/policy_infer_torch.py",
            TORCH_CHECKPOINT_ARC,
        ]
        for rel in expected:
            if not os.path.exists(os.path.join(staging, rel)):
                raise SystemExit(f"  ERROR: archive is missing {rel}")
        forbidden = (
            "rl/policy_mlx.py",
            "configs/train_config.json",
            "configs/train_config.schema.json",
            "_vendor",
        )
        for rel in forbidden:
            if os.path.exists(os.path.join(staging, rel)):
                raise SystemExit(f"  ERROR: PyTorch-only archive unexpectedly contains {rel}")
        print(f"  OK: {len(expected)} required paths present")
        print(f"  OK: checkpoint at {TORCH_CHECKPOINT_ARC}")
        print("  OK: no MLX policy, MLX wheels, or vendored runtime")

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

            # The harness calls whichever callable is defined last, not the one
            # named `agent` (kaggle_environments/agent.py: get_last_callable).
            # Defining a helper below it silently swaps out the entry point.
            last_callable = [v for v in vars(module).values() if callable(v)][-1]
            if getattr(last_callable, "__name__", None) != "agent":
                raise SystemExit(
                    f"  ERROR: last callable in main.py is "
                    f"'{getattr(last_callable, '__name__', '?')}', not 'agent' — "
                    f"the harness would call the wrong function")

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
