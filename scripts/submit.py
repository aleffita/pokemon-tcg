"""Submit agent to the Pokemon TCG AI Battle Challenge on Kaggle.

Builds submission.tar.gz using the existing build logic, then uploads
to the competition via the Kaggle API.

Usage:
    uv run tcg-submit                     # build + submit
    uv run tcg-submit --message "v4"      # custom message
    uv run tcg-submit --dry-run           # build only, no upload
    uv run tcg-submit --help
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
from datetime import datetime
from pathlib import Path

from rich.console import Console

_ROOT = Path(__file__).resolve().parent.parent
console = Console()


def _get_api():
    """Create and authenticate a KaggleApi instance."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        console.print("[bold red]Error:[/] kaggle package not installed.")
        console.print("  Run:  uv add kaggle")
        sys.exit(1)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        console.print(f"[bold red]Kaggle authentication failed:[/] {exc}")
        console.print(
            "  Make sure ~/.kaggle/kaggle.json exists with your API key.\n"
            "  See: https://github.com/Kaggle/kaggle-api#api-credentials"
        )
        sys.exit(1)
    return api


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
            arc = os.path.join(prefix, os.path.relpath(full, base_dir)).replace(
                os.sep, "/"
            )
            files.append((full, arc))
    return files


def build_submission(out_path: str) -> bool:
    """Build submission.tar.gz. Returns True on success."""
    agent_dir = str(_ROOT / "agent")
    rl_dir = str(_ROOT / "rl")
    model_dir = str(_ROOT / "model")
    required = ["main.py", "deck.csv"]

    files: list[tuple[str, str]] = []

    # 1. agent/ contents at top level (main.py, deck.csv)
    files.extend(_collect_files(agent_dir, ""))

    # 2. rl/ directory (encoder, policy, lr_schedule, etc.)
    if os.path.isdir(rl_dir):
        files.extend(_collect_files(rl_dir, "rl"))
    else:
        console.print("[yellow]WARNING: rl/ not found — submission may fail on Kaggle[/]")

    # 3. Model checkpoint (prefer MLX, fallback PyTorch)
    checkpoint = None
    for candidate in [
        os.path.join(model_dir, "bc_model", "bc_best_mlx_final.pkl"),
        os.path.join(model_dir, "checkpoint", "bc_best_mlx.pkl"),
        os.path.join(model_dir, "bc_model", "bc_best_final.pkl"),
        os.path.join(model_dir, "checkpoint", "bc_best.pt"),
    ]:
        if os.path.exists(candidate):
            checkpoint = candidate
            break

    if checkpoint:
        # Determine archive name based on extension
        ext = os.path.splitext(checkpoint)[1]
        arc_name = f"model/bc_best{ext}"
        files.append((checkpoint, arc_name))
    else:
        console.print("[yellow]WARNING: no model checkpoint found — agent will use fallback policy[/]")

    # Validate required files
    arcnames = {arc for _, arc in files}
    missing = [r for r in required if r not in arcnames]
    if missing:
        console.print(f"[bold red]ERROR: missing required file(s) at top level: {missing}[/]")
        return False

    # Validate deck
    deck_path = os.path.join(agent_dir, "deck.csv")
    with open(deck_path) as f:
        n = len([ln for ln in f if ln.strip()])
    if n != 60:
        console.print(f"[bold red]ERROR: deck.csv has {n} cards, expected 60.[/]")
        return False

    # 4. EN_Card_Data.csv (needed by card_features.py at runtime)
    csv_path = str(_ROOT / "EN_Card_Data.csv")
    if os.path.exists(csv_path):
        files.append((csv_path, "EN_Card_Data.csv"))
    else:
        console.print("[yellow]WARNING: EN_Card_Data.csv not found[/]")

    # Write tarball
    with tarfile.open(out_path, "w:gz") as tar:
        for full, arc in sorted(files, key=lambda x: x[1]):
            tar.add(full, arcname=arc)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    console.print(f"[green]Wrote {out_path}[/] ({size_mb:.1f} MB, {len(files)} files)")

    if size_mb > 197.7:
        console.print(f"[bold yellow]WARNING: submission is {size_mb:.1f} MB (limit is 197.7 MB)[/]")

    return True


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-m", "--message",
        default=None,
        help="Submission message (default: auto-generated with date)",
    )
    p.add_argument(
        "--out",
        default=str(_ROOT / "submission.tar.gz"),
        help="Output path for submission archive (default: submission.tar.gz)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build only, do not upload to Kaggle",
    )
    p.add_argument(
        "--competition",
        default="pokemon-tcg-ai-battle",
        help="Kaggle competition slug (default: pokemon-tcg-ai-battle)",
    )
    args = p.parse_args()

    # 1. Build submission
    console.print("\n[bold cyan]Building submission...[/]")
    if not build_submission(args.out):
        sys.exit(1)

    if args.dry_run:
        console.print("\n[dim]Dry run — skipping upload[/]")
        sys.exit(0)

    # 2. Upload to Kaggle
    console.print("\n[bold cyan]Submitting to Kaggle...[/]")
    api = _get_api()

    message = args.message or f"MLX agent {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    try:
        result = api.competitions.submit(
            competition=args.competition,
            file_name=args.out,
            message=message,
        )
        console.print(f"\n[bold green]Submission successful![/]")
        console.print(f"  Submission ID: {result.ref}")
        console.print(f"  Status: {result.status}")
        console.print(f"  Message: {result.description}")
        console.print(
            f"\n  [link=https://www.kaggle.com/competitions/{args.competition}/submissions]"
            f"View on Kaggle[/link]"
        )
    except Exception as exc:
        console.print(f"[bold red]Submission failed:[/] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
