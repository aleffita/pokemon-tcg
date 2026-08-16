"""Build the agent bundle for the Pokemon TCG AI Battle Challenge, and
optionally upload it to Kaggle.

Packaging is delegated to scripts/build_submission.py — the single place that
knows how to bundle the agent (self-describing PyTorch FP32 checkpoint and archive layout)
and validate the result. This module only adds the upload step.

Usage:
    uv run tcg-build --checkpoint model/checkpoint/<model>.pkl
    uv run tcg-build --checkpoint model/checkpoint/<model>.pkl --upload
    uv run tcg-build --checkpoint model/checkpoint/<model>.pkl --upload -m "v4"
    uv run tcg-build --help

Building never uploads. Sending is opt-in through --upload.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

console = Console()

# Competition limits, from the competition's own submission FAQ.
SIZE_LIMIT_MIB = 197.7
DAILY_LIMIT = 5


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


def build_submission(
    out_path: str,
    checkpoint: str | None = None,
    *,
    smoke: bool = False,
) -> bool:
    """Build and validate the bundle via build_submission.py. True on success.

    That module owns packaging: PyTorch conversion, checkpoint paths, the
    size ceiling, and an extract-and-run validation. Duplicating any of it here
    is how this script previously drifted into shipping an unusable bundle.
    """
    from scripts.build_submission import main as build_main

    argv = sys.argv
    sys.argv = ["build_submission.py", "-o", out_path]
    if checkpoint is not None:
        sys.argv.extend(["--checkpoint", checkpoint])
    if smoke:
        sys.argv.append("--smoke")
    try:
        build_main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            console.print(f"[bold red]Build failed:[/] {exc}")
            return False
    finally:
        sys.argv = argv

    if not os.path.exists(out_path):
        console.print(f"[bold red]Build produced no archive at {out_path}[/]")
        return False
    return True


def _describe_bundle(out_path: str) -> None:
    """Print what is about to be uploaded, so it can be checked before sending."""
    import hashlib
    import tarfile

    size_mib = os.path.getsize(out_path) / (1024 * 1024)
    digest = hashlib.sha256(Path(out_path).read_bytes()).hexdigest()[:12]

    deck_path = _ROOT / "agent" / "deck.csv"
    deck_cards = [ln.strip() for ln in deck_path.read_text().splitlines() if ln.strip()]

    with tarfile.open(out_path, "r:gz") as tar:
        names = tar.getnames()
    checkpoint = next(
        (name for name in names if name.endswith((".pt", ".pkl"))),
        "none",
    )

    console.print("\n[bold]Bundle[/]")
    console.print(f"  archive     {out_path}")
    console.print(f"  size        {size_mib:.1f} MiB of {SIZE_LIMIT_MIB} MiB")
    console.print(f"  sha256      {digest}…")
    console.print(f"  checkpoint  {checkpoint}")
    console.print(f"  deck        {len(deck_cards)} cards from agent/deck.csv")
    backend = "PyTorch FP32" if checkpoint.endswith(".pt") else "unknown"
    console.print(f"  backend     {backend}")


def main():
    p = argparse.ArgumentParser(
        prog="tcg-build",
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
        default=None,
        help=(
            "Output path. Defaults to submission.tar.gz, or the isolated "
            "public_agents smoke path with --smoke."
        ),
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Explicit MLX checkpoint to convert or PyTorch FP32 candidate "
            "to validate and package"
        ),
    )
    p.add_argument(
        "--upload",
        action="store_true",
        help="Upload to Kaggle after building (default: build only)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Build only from model/checkpoint/smoke into the isolated local "
            "public-agents smoke artifact"
        ),
    )
    p.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip the upload confirmation prompt (only with --upload)",
    )
    p.add_argument(
        "--competition",
        default="pokemon-tcg-ai-battle",
        help="Kaggle competition slug (default: pokemon-tcg-ai-battle)",
    )
    args = p.parse_args()
    if args.smoke and args.upload:
        p.error("--smoke is a local validation artifact and cannot be uploaded")
    if args.out is None:
        args.out = str(
            _ROOT
            / (
                "public_agents/submissions/smoke/submission_smoke.tar.gz"
                if args.smoke
                else "submission.tar.gz"
            )
        )

    # 1. Build submission
    console.print("\n[bold cyan]Building submission...[/]")
    if not build_submission(
        args.out,
        checkpoint=args.checkpoint,
        smoke=args.smoke,
    ):
        sys.exit(1)

    _describe_bundle(args.out)

    if not args.upload:
        console.print("\n[green]Ready.[/] Upload with [bold]--upload[/], or manually at")
        console.print(f"  https://www.kaggle.com/competitions/{args.competition}/submissions")
        sys.exit(0)

    message = args.message or f"PyTorch FP32 agent {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    # 2. Confirm. Uploading is public, irreversible, and spends one of the
    # competition's daily submission slots.
    if not args.yes:
        console.print(
            f"\n[bold yellow]About to upload to '{args.competition}'.[/] "
            f"This is public, cannot be undone, and uses one of "
            f"{DAILY_LIMIT} submissions today."
        )
        console.print(f"  message: [italic]{message}[/]")
        try:
            reply = input("Type 'submit' to confirm: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Cancelled.[/]")
            sys.exit(1)
        if reply != "submit":
            console.print("[dim]Cancelled — nothing uploaded.[/]")
            sys.exit(1)

    # 3. Upload to Kaggle
    console.print("\n[bold cyan]Submitting to Kaggle...[/]")
    api = _get_api()

    try:
        # kaggle 2.x exposes this as a flat method; the old api.competitions.submit
        # namespace no longer exists.
        result = api.competition_submit(
            file_name=args.out,
            message=message,
            competition=args.competition,
        )
        console.print("\n[bold green]Submission successful![/]")
        for label, attr in (("Status", "status"), ("Message", "message"),
                            ("Ref", "ref"), ("Url", "url")):
            value = getattr(result, attr, None)
            if value:
                console.print(f"  {label}: {value}")
        console.print(
            f"\n  Track it at https://www.kaggle.com/competitions/"
            f"{args.competition}/submissions"
        )
    except Exception as exc:
        console.print(f"[bold red]Submission failed:[/] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
