#!/usr/bin/env python3
"""
Runs full deck sweep tournaments across ALL public opponents (no specific --opponent filter)
with --sweep-source remote (5 decks total, 10 games per deck = 50 games per opponent)
for the three checkpoints of the curriculum pipeline:
  1. Base Model (Pre-Stage 1): suite_5d_10ep_OFF.pkl
  2. Stage 1 (Post-Stage 1): curriculum_v1_stage1.pkl (25ep, all-days 30k)
  3. Stage 2 (Post-Stage 2): curriculum_v1_stage2.pkl (5ep, top-600)
"""

import sys
import json
import shutil
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
EXP_ROOT = ROOT_DIR / "experiments" / "curriculum_v1"
REPORTS_DIR = EXP_ROOT / "reports"
MODELS_DIR = EXP_ROOT / "models"


def run_cmd(cmd: list[str]):
    print(f"\n[EXEC] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def package_checkpoint(pkl_path: Path, tag: str):
    target_pkl = ROOT_DIR / "model" / "checkpoint" / "bc_best_mlx.pkl"
    target_pkl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(pkl_path, target_pkl)

    dst_tar = MODELS_DIR / f"{tag}.tar.gz"
    run_cmd([
        sys.executable, "-m", "scripts.submit",
        "--checkpoint", str(target_pkl),
        "--out", str(dst_tar)
    ])


def run_full_sweep_tournament(tag: str, report_json: Path, games_per_deck: int = 10, top_decks: int = 4):
    report_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "scripts.tournament",
        "--games", str(games_per_deck),
        "--top-decks", str(top_decks),
        "--sweep-source", "remote",
        "--note", f"curriculum_v1 {tag} full 5-deck sweep (50 games/opp)",
        "--report-json", str(report_json),
    ]
    run_cmd(cmd)


def print_comparison(reports: dict[str, Path]):
    data = {}
    all_opps = set()
    for name, path in reports.items():
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                r = json.load(f)
                data[name] = r
                for row in r.get("rows", []):
                    all_opps.add(row["opponent_label"])

    sorted_opps = sorted(all_opps)

    header = f"{'Opponent':<32}"
    for name in reports.keys():
        header += f" | {name:<18}"
    
    print("\n" + "=" * 90)
    print("  FULL 5-DECK SWEEP COMPARISON (50 GAMES/OPPONENT)")
    print("=" * 90)
    print(header)
    print("-" * 90)

    for opp in sorted_opps:
        line = f"{opp:<32}"
        for name in reports.keys():
            if name in data:
                rows = {r["opponent_label"]: r for r in data[name].get("rows", [])}
                wr = f"{rows[opp]['wr_pct']:.1f}%" if opp in rows else "N/A"
            else:
                wr = "N/A"
            line += f" | {wr:<18}"
        print(line)

    print("-" * 90)
    summary_line = f"{'OVERALL WIN RATE':<32}"
    for name in reports.keys():
        if name in data:
            owr = f"{data[name].get('overall', {}).get('wr_pct', 0):.1f}%"
        else:
            owr = "N/A"
        summary_line += f" | {owr:<18}"
    print(summary_line)
    print("=" * 90 + "\n")


def main():
    base_pkl = ROOT_DIR / "model" / "checkpoint" / "suite_5d_10ep_OFF" / "5d_10ep_OFF.pkl"
    s1_pkl = EXP_ROOT / "stage1" / "curriculum_v1_stage1.pkl"
    s2_pkl = EXP_ROOT / "stage2" / "curriculum_v1_stage2.pkl"

    r_base_json = REPORTS_DIR / "base_full_sweep.json"
    r1_json = REPORTS_DIR / "stage1_full_sweep.json"
    r2_json = REPORTS_DIR / "stage2_full_sweep.json"

    reports = {
        "Base (Pre-S1)": r_base_json,
        "Stage 1 (25ep OFF)": r1_json,
        "Stage 2 (5ep Top600)": r2_json,
    }

    if base_pkl.exists():
        print("\n>>> 1/3: Evaluating Base Model (Pre-Stage 1) with 5-deck sweep (30 games/deck)...")
        package_checkpoint(base_pkl, "base_model")
        run_full_sweep_tournament("base_model", r_base_json, games_per_deck=30, top_decks=4)

    if s1_pkl.exists():
        print("\n>>> 2/3: Evaluating Stage 1 (25ep OFF) with 5-deck sweep (30 games/deck)...")
        package_checkpoint(s1_pkl, "stage1")
        run_full_sweep_tournament("stage1", r1_json, games_per_deck=30, top_decks=4)

    if s2_pkl.exists():
        print("\n>>> 3/3: Evaluating Stage 2 (5ep Top-600) with 5-deck sweep (30 games/deck)...")
        package_checkpoint(s2_pkl, "stage2")
        run_full_sweep_tournament("stage2", r2_json, games_per_deck=30, top_decks=4)

    print_comparison(reports)


if __name__ == "__main__":
    main()
