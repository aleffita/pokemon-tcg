#!/usr/bin/env python3
"""
Curriculum V1: 3-stage coarse->fine training pipeline for Pokémon TCG AI.

Stages:
  Stage 1: OFF top-elo, all days,     15ep,  80k rows/day, lr=1.5e-4
  Stage 2: ON top-600 real, last 5d,   5ep, 300k rows/day, lr=8e-5
  Stage 3: ON top-100 real, last 1d,  10ep, 300k rows/day, lr=3e-5

Usage:
  uv run python experiments/curriculum_v1.py
  # Or via CLI entrypoint:
  uv run tcg-curriculum-v1
"""

import sys
import os
import glob
import json
import shutil
import sqlite3
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Project root path resolution
ROOT_DIR = Path(__file__).resolve().parent.parent
EXP_ROOT = ROOT_DIR / "experiments" / "curriculum_v1"
REPORTS_DIR = EXP_ROOT / "reports"
MODELS_DIR = EXP_ROOT / "models"
BASE_CHECKPOINT = (
    ROOT_DIR / "model" / "checkpoint" / "suite_5d_10ep_OFF" / "5d_10ep_OFF.pkl"
)
CONFIG_FILE = ROOT_DIR / "configs" / "train_config.json"

OPPONENTS = [
    "public_agents/starters/lb510_mega_abomasnow_ex",
    "public_agents/starters/lb526_iono",
    "public_agents/starters/lb600_dragapult_ex",
    "public_agents/starters/lb600_mega_lucario_ex",
    "public_agents/lb826_alakazam_seok",
    "public_agents/lb945_multiply_ivan",
    "public_agents/lb1009_mega_lucario_ex_islet",
]

TOURNAMENT_GAMES = 30


def format_duration(seconds: float) -> str:
    s = int(seconds)
    hours, remainder = divmod(s, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    elif minutes > 0:
        return f"{minutes}m{secs:02d}s"
    else:
        return f"{secs}s"


def run_command(
    cmd: list[str], check: bool = True, log_file: Path | None = None
) -> int:
    """Runs a command, streaming output to stdout and optionally logging to a file."""
    print(f"\n[EXEC] {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)

    log_handle = None
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_file, "a", encoding="utf-8")

    process = subprocess.Popen(
        cmd,
        cwd=ROOT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_handle:
                log_handle.write(line)
                log_handle.flush()
        process.wait()
    finally:
        if log_handle:
            log_handle.close()

    if check and process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)
    return process.returncode


def pre_suite_enrichment():
    """Runs data download, DB competition day refresh, parquet build, card stats, and ELO reconciliation."""
    print("\n" + "=" * 70)
    print("  PRE-SUITE ENRICHMENT")
    print("=" * 70)

    # 1. Download missing Kaggle days
    zip_files = sorted(glob.glob(str(ROOT_DIR / "data" / "bc_replay_zip" / "*.zip")))
    latest_local = None
    if zip_files:
        latest_filename = Path(zip_files[-1]).name
        if len(latest_filename) >= 10:
            latest_local = latest_filename[:10]

    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()

    if latest_local and latest_local < yesterday:
        next_day = (
            (datetime.strptime(latest_local, "%Y-%m-%d") + timedelta(days=1))
            .date()
            .isoformat()
        )
        print(f"----- downloading Kaggle days {next_day} .. {yesterday} -----")
        run_command(
            [
                sys.executable,
                "-m",
                "scripts.data_manager",
                "--range",
                next_day,
                yesterday,
            ],
            check=False,
        )
    else:
        print(
            f"----- Kaggle days: nothing to download (latest local = {latest_local}) -----"
        )

    # 2. Refresh competition_day
    print("----- refreshing days.competition_day -----")
    from rl.results_db import ResultsDB

    db = ResultsDB(str(ROOT_DIR / "model" / "results.db"))
    db.refresh_competition_days()
    db.close()
    print("  refreshed competition_day for all days in calendar order")

    # 3. Build Parquets for new zips
    print("----- building Parquets for any new zips -----")
    run_command(
        [sys.executable, "-m", "scripts.bc.build_bc_from_zips", "--all", "--resume"],
        check=False,
    )

    # 4. Build card/deck ELO stats
    print("----- building card/deck Elo stats for any new days -----")
    db_path = ROOT_DIR / "model" / "results.db"
    conn = sqlite3.connect(db_path)
    have_stats = {
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT d.date FROM days d JOIN matches m ON m.day_id=d.id WHERE m.source='remote'"
        ).fetchall()
    }
    conn.close()

    parquet_files = sorted(Path(ROOT_DIR / "data" / "bc_data").glob("*.parquet"))
    parquet_days = [p.stem for p in parquet_files]
    missing = [d for d in parquet_days if d not in have_stats]

    if missing:
        start_day, end_day = missing[0], missing[-1]
        print(f"  new stats range: {start_day} .. {end_day}")
        run_command(
            [
                sys.executable,
                "-m",
                "scripts.build_card_stats",
                "--range",
                start_day,
                end_day,
            ],
            check=False,
        )
    else:
        print("  all days already have matches populated — skipping")

    # 5. Reconcile ELO
    print("----- reconciling agent_elo_daily(source='remote') with Kaggle LB -----")
    run_command([sys.executable, "-m", "scripts.reconcile_elo"], check=False)


def package_and_swap(pkl_path: Path, tag: str):
    """Packages the trained model checkpoint into a submission tarball."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst_tar = MODELS_DIR / f"{tag}.tar.gz"

    target_pkl = ROOT_DIR / "model" / "checkpoint" / "bc_best_mlx.pkl"
    target_pkl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(pkl_path, target_pkl)

    run_command(
        [
            sys.executable,
            "-m",
            "scripts.submit",
            "--checkpoint",
            str(target_pkl),
            "--out",
            str(dst_tar),
        ]
    )

    if not dst_tar.exists():
        raise RuntimeError(f"build_submission failed to produce {dst_tar}")
    print(f"  packaged {tag} -> {dst_tar}")


def run_tournament(tag: str, json_path: Path, extra_flags: list[str] | None = None):
    """Runs tournament evaluation for a given stage model."""
    if json_path.exists():
        print(f">>> tournament {tag} already done at {json_path} — skipping")
        return

    json_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = json_path.with_suffix(".log")

    cmd = [
        sys.executable,
        "-m",
        "scripts.tournament",
        "--games",
        str(TOURNAMENT_GAMES),
        "--note",
        f"curriculum_v1 {tag} {' '.join(extra_flags or [])}".strip(),
        "--report-json",
        str(json_path),
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    for opp in OPPONENTS:
        cmd.extend(["--opponent", opp])

    run_command(cmd, check=True, log_file=log_path)

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            wr = data.get("overall", {}).get("wr_pct", "?")
            print(f">>> tournament {tag} DONE: overall_wr={wr}%")


def run_stage_1():
    print("\n" + "=" * 70)
    print("  STAGE 1: OFF, all_days, 15ep, 30k/day, lr=1.5e-4")
    print(f"  resume from base: {BASE_CHECKPOINT}")
    print("=" * 70)

    s1_root = EXP_ROOT / "stage1"
    s1_final = s1_root / "curriculum_v1_stage1.pkl"
    s1_tourn = REPORTS_DIR / "stage1_tourn.json"

    s1_root.mkdir(parents=True, exist_ok=True)

    if s1_final.exists():
        print(
            f">>> stage 1 final checkpoint already exists at {s1_final} — skipping training"
        )
    else:
        # Clean start for stage 1 if final checkpoint is missing
        if s1_root.exists():
            shutil.rmtree(s1_root)
        s1_root.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        cmd = [
            sys.executable,
            "-m",
            "scripts.bc.bc_train_mlx",
            "--config",
            str(CONFIG_FILE),
            "--all-days",
            "--max-rows-per-day",
            "30000",
            "--epochs",
            "15",
            "--lr",
            "1.5e-4",
            "--resume",
            str(BASE_CHECKPOINT),
            "--optimizer-state",
            "reset",
            "--scheduler-state",
            "reset",
            "--out",
            str(s1_final),
            "--checkpoint-every-epochs",
            "1",
        ]
        run_command(cmd, check=True, log_file=s1_root / "_train.log")
        dur = time.time() - t0
        print(f">>> stage 1 training DONE in {format_duration(dur)}")

        # Clean cache spill
        cache_spill = s1_root / ".cache_spill"
        if cache_spill.exists():
            shutil.rmtree(cache_spill, ignore_errors=True)

    package_and_swap(s1_final, "stage1")
    run_tournament("stage1", s1_tourn, ["--no-sweep"])
    return s1_final


def run_stage_2(s1_final: Path):
    print("\n" + "=" * 70)
    print("  STAGE 2: ON top-600 (bronze+), last-5-days, 5ep, 300k/day, lr=8e-5")
    print(f"  resume from stage 1 final: {s1_final}")
    print("=" * 70)

    s2_root = EXP_ROOT / "stage2"
    s2_final = s2_root / "curriculum_v1_stage2.pkl"
    s2_tourn = REPORTS_DIR / "stage2_tourn.json"

    s2_root.mkdir(parents=True, exist_ok=True)

    if s2_final.exists():
        print(
            f">>> stage 2 final checkpoint already exists at {s2_final} — skipping training"
        )
    else:
        if s2_root.exists():
            shutil.rmtree(s2_root)
        s2_root.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        cmd = [
            sys.executable,
            "-m",
            "scripts.bc.bc_train_mlx",
            "--config",
            str(CONFIG_FILE),
            "--last-n-days",
            "5",
            "--max-rows-per-day",
            "300000",
            "--epochs",
            "5",
            "--top-elo",
            "600",
            "--lr",
            "8e-5",
            "--resume",
            str(s1_final),
            "--optimizer-state",
            "reset",
            "--scheduler-state",
            "reset",
            "--out",
            str(s2_final),
            "--checkpoint-every-epochs",
            "1",
        ]
        run_command(cmd, check=True, log_file=s2_root / "_train.log")
        dur = time.time() - t0
        print(f">>> stage 2 training DONE in {format_duration(dur)}")

        cache_spill = s2_root / ".cache_spill"
        if cache_spill.exists():
            shutil.rmtree(cache_spill, ignore_errors=True)

    package_and_swap(s2_final, "stage2")
    run_tournament("stage2", s2_tourn, ["--no-sweep"])
    return s2_final


def run_stage_3(s2_final: Path):
    print("\n" + "=" * 70)
    print("  STAGE 3: ON top-100 (elite), last-1-day, 10ep, 300k/day, lr=3e-5")
    print(f"  resume from stage 2 final: {s2_final}")
    print("=" * 70)

    s3_root = EXP_ROOT / "stage3"
    s3_final = s3_root / "curriculum_v1_stage3.pkl"
    s3_tourn_nosweep = REPORTS_DIR / "stage3_tourn_nosweep.json"
    s3_tourn_sweep = REPORTS_DIR / "stage3_tourn_sweep.json"

    s3_root.mkdir(parents=True, exist_ok=True)

    if s3_final.exists():
        print(
            f">>> stage 3 final checkpoint already exists at {s3_final} — skipping training"
        )
    else:
        if s3_root.exists():
            shutil.rmtree(s3_root)
        s3_root.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        cmd = [
            sys.executable,
            "-m",
            "scripts.bc.bc_train_mlx",
            "--config",
            str(CONFIG_FILE),
            "--last-n-days",
            "2",
            "--max-rows-per-day",
            "300000",
            "--epochs",
            "10",
            "--top-elo",
            "100",
            "--lr",
            "3e-5",
            "--resume",
            str(s2_final),
            "--optimizer-state",
            "reset",
            "--scheduler-state",
            "reset",
            "--out",
            str(s3_final),
            "--checkpoint-every-epochs",
            "1",
        ]
        run_command(cmd, check=True, log_file=s3_root / "_train.log")
        dur = time.time() - t0
        print(f">>> stage 3 training DONE in {format_duration(dur)}")

        cache_spill = s3_root / ".cache_spill"
        if cache_spill.exists():
            shutil.rmtree(cache_spill, ignore_errors=True)

    package_and_swap(s3_final, "stage3")
    run_tournament("stage3_nosweep", s3_tourn_nosweep, ["--no-sweep"])
    run_tournament("stage3_sweep", s3_tourn_sweep, ["--sweep-source", "remote"])


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pre_suite_enrichment()
    s1_final = run_stage_1()
    s2_final = run_stage_2(s1_final)
    run_stage_3(s2_final)

    print("\n" + "=" * 70)
    print("  Curriculum V1 complete.")
    print(f"  Reports:   {REPORTS_DIR}")
    print(f"  Models:    {MODELS_DIR}")
    print(f"  Checkpts:  {EXP_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
