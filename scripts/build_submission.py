"""Package agent/ into submission.tar.gz for the PTCG AI Battle Challenge.

The competition requires main.py and deck.csv at the root of the archive.
This script also bundles:
  - rl/ (encoder, policy, etc. — needed by main.py)
  - model checkpoint (bc_best_final.pt)

The cg/ SDK is NOT bundled — it's already in the Kaggle sandbox via kaggle_environments.

Usage:
    python scripts/build_submission.py            # -> submission.tar.gz
    python scripts/build_submission.py -o out.tar.gz
"""

import argparse
import os
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_DIR = os.path.join(ROOT, "agent")
RL_DIR = os.path.join(ROOT, "rl")
MODEL_DIR = os.path.join(ROOT, "model")
REQUIRED = ["main.py", "deck.csv"]


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

    # 3. Model checkpoint (bundled as model/bc_best_final.pt)
    checkpoint = os.path.join(MODEL_DIR, "bc_model", "bc_best_final.pt")
    if os.path.exists(checkpoint):
        files.append((checkpoint, "model/bc_best_final.pt"))
    else:
        # Try checkpoint dir
        checkpoint = os.path.join(MODEL_DIR, "checkpoint", "bc_best.pt")
        if os.path.exists(checkpoint):
            files.append((checkpoint, "model/bc_best_final.pt"))
        else:
            print("WARNING: no model checkpoint found — agent will use fallback policy")

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

    # Write tarball
    with tarfile.open(args.out, "w:gz") as tar:
        for full, arc in sorted(files, key=lambda x: x[1]):
            tar.add(full, arcname=arc)

    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"Wrote {args.out} ({size_mb:.1f} MB)")
    print(f"Contents ({len(files)} files):")
    for _, arc in sorted(files, key=lambda x: x[1]):
        print(f"  {arc}")

    if size_mb > 197.7:
        print(f"\nWARNING: submission is {size_mb:.1f} MB (limit is 197.7 MB)")

    # Validate agent loads correctly
    if not args.no_validate:
        import traceback
        print("\nValidating agent...")
        try:
            sys.path.insert(0, ROOT)
            from agent.main import agent as test_agent
            result = test_agent({"select": None})
            if isinstance(result, list) and len(result) == 60:
                print(f"  OK: agent returns {len(result)} cards")
            else:
                print(f"  WARNING: agent returned {type(result)} len={len(result) if hasattr(result, '__len__') else '?'}")
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            print(f"  Submission may fail on Kaggle!")


if __name__ == "__main__":
    main()
