#!/usr/bin/env python3
"""Kaggle dataset downloader with Rich progress bars.

Downloads Pokemon TCG replay episode datasets from Kaggle.
Idempotent: skips files that already exist with reasonable size.

Usage:
    uv run tcg-data --last                    # latest day only
    uv run tcg-data --date 2026-07-25         # specific date
    uv run tcg-data --range 2026-07-20 2026-07-25  # date range
    uv run tcg-data --all                     # all available
    uv run tcg-data --list                    # list available datasets
"""

import argparse
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)


MIN_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB — anything below this is suspect

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


def _dataset_slug(prefix: str, date_str: str) -> str:
    """Build the full dataset slug for a given date."""
    return f"{prefix}-{date_str}"


def _local_path(replay_zip_dir: str, date_str: str) -> Path:
    """Expected local zip path for a given date."""
    return Path(replay_zip_dir) / f"{date_str}.zip"


def _is_downloaded(path: Path) -> bool:
    """Check if a file already exists and looks complete."""
    if not path.is_file():
        return False
    return path.stat().st_size >= MIN_SIZE_BYTES





def show_catalog(db_path: str, replay_zip_dir: str) -> None:
    """Render the built-parquet dataset catalog from SQLite."""
    from pathlib import Path as _P
    if not _P(db_path).exists():
        console.print(f"[yellow]No catalog DB at {db_path}. Run tcg-rebuild-db first.[/]")
        return
    from rl.results_db import ResultsDB
    db = ResultsDB(db_path)
    try:
        rows = db.conn.execute(
            """
            SELECT d.id AS day_id, d.date, d.competition_day, d.n_matches,
                   d.source_zip, d.source_sha256,
                   ds.id AS dataset_id, ds.path, ds.rows AS parquet_rows,
                   ds.schema_version, ds.aux_targets, ds.sha256 AS parquet_sha256,
                   ds.created_at
            FROM days d
            LEFT JOIN datasets ds ON ds.day_id = d.id
            ORDER BY d.date
            """
        ).fetchall()
    finally:
        db.close()

    table = Table(title="Dataset Catalog (days ∙ parquets)")
    table.add_column("D#", justify="right")
    table.add_column("Date", style="cyan", no_wrap=True)
    table.add_column("Matches", justify="right", style="magenta")
    table.add_column("Zip", style="dim")
    table.add_column("Parquet", style="green")
    table.add_column("Rows", justify="right")
    table.add_column("Aux", justify="center")
    table.add_column("Schema", justify="center")

    total_matches = 0
    total_rows = 0
    parquets_present = 0
    parquets_registered = 0

    for r in rows:
        comp_day = f"D{r['competition_day']}" if r['competition_day'] else "?"
        matches = int(r['n_matches'] or 0)
        total_matches += matches
        zip_status = "[green]✓[/]" if r['source_zip'] else "[dim]—[/]"

        if r['dataset_id'] is None:
            parquet_status = "[dim]—[/]"
            rows_str = "—"
            aux_str = "—"
            schema_str = "—"
        else:
            parquets_registered += 1
            parquet_path = r['path']
            if _P(parquet_path).exists():
                parquets_present += 1
                size_mb = _P(parquet_path).stat().st_size / (1024 * 1024)
                parquet_status = f"[green]✓[/] {size_mb:.0f}MB"
            else:
                parquet_status = f"[red]missing[/]"
            rows_str = f"{int(r['parquet_rows'] or 0):,}"
            total_rows += int(r['parquet_rows'] or 0)
            aux_str = "[green]✓[/]" if r['aux_targets'] else "[dim]—[/]"
            schema_str = f"v{r['schema_version']}"

        table.add_row(comp_day, r['date'], f"{matches:,}", zip_status,
                      parquet_status, rows_str, aux_str, schema_str)

    console.print(table)
    console.print()
    console.print(
        f"[bold]Summary:[/] {len(rows)} days · "
        f"{total_matches:,} matches · "
        f"{parquets_registered} parquets registered "
        f"({parquets_present} present) · "
        f"{total_rows:,} training rows"
    )


def show_dataset_table(dates: list[str], replay_zip_dir: str) -> None:
    """Render a table of available datasets with local status."""
    table = Table(title="Available Kaggle Replay Datasets")
    table.add_column("Date", style="cyan", no_wrap=True)
    table.add_column("Local File", style="green")
    table.add_column("Status", justify="center")

    for date_str in dates:
        lp = _local_path(replay_zip_dir, date_str)
        if _is_downloaded(lp):
            size_mb = lp.stat().st_size / (1024 * 1024)
            status = f"[green]Downloaded[/] ({size_mb:.0f} MB)"
        elif lp.exists():
            size_kb = lp.stat().st_size / 1024
            status = f"[yellow]Partial[/] ({size_kb:.0f} KB)"
        else:
            status = "[dim]Not downloaded[/]"
        table.add_row(date_str, str(lp), status)

    console.print(table)


def download_dataset(api, prefix: str, date_str: str, replay_zip_dir: str, force: bool = False) -> bool:
    """Download a single dataset zip. Returns True on success or skip."""
    lp = _local_path(replay_zip_dir, date_str)
    slug = _dataset_slug(prefix, date_str)

    if not force and _is_downloaded(lp):
        size_mb = lp.stat().st_size / (1024 * 1024)
        console.print(
            f"  [green]Skip[/] {date_str}  "
            f"({size_mb:.0f} MB already present)"
        )
        return True

    lp.parent.mkdir(parents=True, exist_ok=True)

    # Download into a private per-date staging directory. Kaggle's
    # dataset_download_file() does not reliably save under the requested
    # local filename (it may keep the archive's own internal name, e.g.
    # "archive.zip"), so callers have historically had to fall back to
    # "whichever .zip is biggest in the target directory". Doing that scan
    # against the shared replay_zip_dir is unsafe: once more than one dated
    # zip already lives there, the fallback can pick a *pre-existing* file
    # instead of the freshly downloaded one and rename it, silently
    # discarding the real download and mislabeling old data under the new
    # date. Scoping the scan to an empty, date-exclusive staging directory
    # makes that ambiguity impossible.
    staging_dir = lp.parent / f".download_{date_str}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    console.print(f"  [cyan]Downloading[/] {slug} ...")

    try:
        # Create a Rich progress context for this download
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(date_str, total=None)

            try:
                api.dataset_download_file(
                    dataset=slug,
                    file_name=None,  # download the full dataset
                    path=str(staging_dir),
                    force=True,
                    quiet=True,
                )
            except Exception as exc:
                console.print(f"  [bold red]Failed:[/] {exc}")
                shutil.rmtree(staging_dir, ignore_errors=True)
                return False

            # The staging dir only ever contains files from this download,
            # so picking the single .zip in it is safe (no cross-date
            # collision is possible).
            candidates = list(staging_dir.glob("*.zip"))
            if not candidates:
                console.print(f"  [bold red]Error:[/] No zip file found after download")
                shutil.rmtree(staging_dir, ignore_errors=True)
                return False
            if len(candidates) > 1:
                console.print(
                    f"  [bold red]Error:[/] {len(candidates)} zip files found for "
                    f"{date_str}, expected exactly 1: {[c.name for c in candidates]}"
                )
                shutil.rmtree(staging_dir, ignore_errors=True)
                return False
            downloaded = candidates[0]
            downloaded.replace(lp)
            final_size = lp.stat().st_size
            progress.update(task_id, completed=final_size, total=final_size)
            shutil.rmtree(staging_dir, ignore_errors=True)

        # Post-download verification
        if final_size < MIN_SIZE_BYTES:
            console.print(
                f"  [yellow]Warning:[/] {date_str} is only {final_size / 1024 / 1024:.1f} MB "
                f"(expected >100 MB)"
            )

        size_mb = final_size / (1024 * 1024)
        console.print(f"  [green]Done[/] {date_str} ({size_mb:.0f} MB)")
        return True

    except Exception as exc:
        console.print(f"  [bold red]Download error:[/] {exc}")
        shutil.rmtree(staging_dir, ignore_errors=True)
        return False


def _date_range(start: str, end: str) -> list[str]:
    """Generate a list of YYYY-MM-DD strings from start to end inclusive."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    dates = []
    cur = s
    while cur <= e:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates


def main():
    parser = argparse.ArgumentParser(
        prog="tcg-data",
        description="Download Pokemon TCG replay datasets from Kaggle.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--last", action="store_true", help="Download only the most recent day (default)")
    group.add_argument("--date", metavar="YYYY-MM-DD", help="Download a specific date")
    group.add_argument("--range", nargs=2, metavar=("START", "END"), help="Download a date range (inclusive)")
    group.add_argument("--all", action="store_true", dest="all_", help="Download all available days")
    group.add_argument("--list", action="store_true", help="List available datasets (no download)")
    group.add_argument(
        "--catalog",
        action="store_true",
        help=(
            "Show the built-parquet dataset catalog from SQLite "
            "(days table + registered parquets + row/aux stats)"
        ),
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument("--config", metavar="PATH", default=None, help="Path to config.json")
    parser.add_argument(
        "--db",
        metavar="PATH",
        default="model/results.db",
        help="Path to results.db (used by --catalog)",
    )
    args = parser.parse_args()

    # Load project config
    from rl.train_config import load_config
    cfg = load_config(config_path=args.config)

    replay_zip_dir = cfg.replay_zip_dir
    prefix = cfg.kaggle_episodes_prefix

    console.print(f"[bold]Pokemon TCG Data Manager[/]")
    console.print(f"  Output dir : {replay_zip_dir}")
    console.print(f"  Prefix     : {prefix}")
    console.print()

    if args.catalog:
        show_catalog(args.db, replay_zip_dir)
        sys.exit(0)

    # Determine which dates to operate on purely locally
    if args.date:
        target_dates = [args.date]
    elif args.range:
        target_dates = _date_range(args.range[0], args.range[1])
    elif args.all_ or args.list:
        # 2026-07-14 is the epoch of our dataset collection
        target_dates = _date_range("2026-07-14", (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))
    else:
        # --last or default
        target_dates = [(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")]

    if args.list:
        show_dataset_table(target_dates, replay_zip_dir)
        sys.exit(0)

    # Pre-flight check on disk before EVER touching the network
    missing_dates = []
    skipped = 0
    console.print(f"[bold]Dates to process:[/] {len(target_dates)}")
    for d in target_dates:
        lp = _local_path(replay_zip_dir, d)
        if not args.force and _is_downloaded(lp):
            size_mb = lp.stat().st_size / (1024 * 1024)
            console.print(f"  [green]Skip[/] {d}  ({size_mb:.0f} MB already present)")
            skipped += 1
        else:
            missing_dates.append(d)

    if not missing_dates:
        console.print("\n[bold]Summary:[/]")
        console.print(f"  Skipped    : {skipped}")
        console.print("  [green]All requested dates are present. No network calls made.[/]")
        sys.exit(0)

    # Only instantiate API if we actually need to download something
    api = _get_api()

    # Download
    success = 0
    failed = 0

    for date_str in missing_dates:
        ok = download_dataset(api, prefix, date_str, replay_zip_dir, force=args.force)
        if ok:
            success += 1
        else:
            failed += 1
            break  # Crash-Early: abort queue on first network failure

    # Summary
    console.print()
    console.print("[bold]Summary:[/]")
    console.print(f"  Downloaded : {success}")
    console.print(f"  Skipped    : {skipped}")
    if failed:
        console.print(f"  [red]Failed    : {failed} (Queue Aborted)[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
