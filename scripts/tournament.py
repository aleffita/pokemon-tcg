"""Mini-tournament: our agent vs multiple public agents.

Runs N games per opponent (alternating sides to cancel first-player advantage)
and reports a results table. Results are saved to SQLite (model/results.db)
with an optional backup append to eval_results.txt.

Usage:
  uv run tcg-tournament                              # all agents, 20 games each
  uv run tcg-tournament --games 50                   # 50 games each
  uv run tcg-tournament --opponent public_agents/lb826_alakazam_seok
"""

import argparse
import os
import sys
import time
from datetime import datetime

from scripts._common import AGENT_DIR, load_agent, make_env

PUBLIC_AGENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "public_agents")
RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "model", "eval_results.txt")


def find_agents() -> list[tuple[str, str]]:
    """Discover all public agents and submissions. Returns [(label, path), ...] sorted by score."""
    agents = []
    for root, dirs, files in os.walk(PUBLIC_AGENTS_DIR):
        if "main.py" in files and root != PUBLIC_AGENTS_DIR:
            # Label = relative path from public_agents/ (e.g. "lb826_alakazam_seok")
            rel = os.path.relpath(root, PUBLIC_AGENTS_DIR)
            if rel.startswith("starters/"):
                label = rel  # keep starters/ prefix
            elif rel.startswith("submissions/"):
                label = rel.replace("submissions/", "sub/")  # sub/lb881_alakazam_v1
            else:
                label = os.path.basename(root)
            path = os.path.join(root, "main.py")
            agents.append((label, path))
        # Also discover submission.tar.gz files
        for f in files:
            if f.endswith(".tar.gz") and "submissions" in root:
                rel = os.path.relpath(root, PUBLIC_AGENTS_DIR)
                label = rel.replace("submissions/", "sub/")
                agents.append((label, os.path.join(root, f)))

    # Sort by score extracted from label (lb{score}_...)
    def sort_key(item):
        label = item[0]
        try:
            # Extract number after "lb"
            return -int(label.split("_")[0].replace("lb", ""))
        except (ValueError, IndexError):
            return 0
    agents.sort(key=sort_key)
    return agents


def resolve(name: str):
    """Map a name to an agent callable."""
    if name in ("random", "first"):
        from kaggle_environments.envs.cabt.cabt import agents
        return agents[name]
    return load_agent(name)


def play(env, a, b) -> tuple[int, str]:
    """Run one game; return (result, html_replay). result: +1=P0 wins, -1=loses, 0=draw."""
    env.reset()
    env.run([a, b])
    r0, r1 = (s.reward for s in env.steps[-1])
    html = env.render(mode="html")
    return (1 if r0 > r1 else (-1 if r0 < r1 else 0)), html


def run_matchup(env, our_agent, opp_agent, n_games: int) -> tuple[int, int, int, str]:
    """Play n_games; return (wins, losses, draws, last_replay_html)."""
    wins = losses = draws = 0
    last_html = ""
    for i in range(n_games):
        if i % 2 == 0:
            r, html = play(env, our_agent, opp_agent)
        else:
            r, html = play(env, opp_agent, our_agent)
            r = -r
        if i == n_games - 1:
            last_html = html
        wins += r == 1
        losses += r == -1
        draws += r == 0
    return wins, losses, draws, last_html


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", "-n", type=int, default=20, help="Games per opponent")
    p.add_argument("--opponent", type=str, default=None,
                   help="Single opponent path (skip tournament)")
    p.add_argument("--note", type=str, default=None,
                   help="Annotation for this run (saved in SQLite)")
    p.add_argument("--txt-backup", action="store_true", default=False,
                   help="Also append results to eval_results.txt (backup)")
    args = p.parse_args()

    # Our agent
    our_path = os.path.join(AGENT_DIR, "main.py")
    our_agent = load_agent(our_path)
    env = make_env()

    # Baselines + public agents
    opponents = [("random", "random"), ("first", "first")]
    if args.opponent:
        label = os.path.basename(os.path.dirname(args.opponent)) if args.opponent.endswith("main.py") else os.path.basename(args.opponent)
        opponents.append((label, args.opponent))
    else:
        opponents.extend(find_agents())

    total_w = total_l = total_d = 0
    rows = []
    start_time = time.time()

    if args.note:
        print(f"Note: {args.note}", flush=True)

    for label, opp_path in opponents:
        try:
            opp_agent = resolve(opp_path)
        except Exception as e:
            rows.append((label, "ERROR", str(e)))
            continue

        t0 = time.time()
        w, l, d, replay_html = run_matchup(env, our_agent, opp_agent, args.games)
        elapsed = time.time() - t0
        wr = w / max(w + l, 1) * 100
        total_w += w; total_l += l; total_d += d
        rows.append((label, w, l, d, wr, elapsed, replay_html))
        print(f"  {label:40s} W={w:3d} L={l:3d} D={d:3d} wr={wr:5.1f}% ({elapsed:.0f}s)", flush=True)

    total_time = time.time() - start_time
    overall_wr = total_w / max(total_w + total_l, 1) * 100

    # Print table
    print()
    header = f"{'Opponent':40s} {'W':>4s} {'L':>4s} {'D':>4s} {'WinRate':>8s} {'Time':>6s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        if len(row) == 3:
            print(f"{row[0]:40s} {row[1]:>4s} — {row[2]}")
        else:
            print(f"{row[0]:40s} {row[1]:>4s} {row[2]:>4s} {row[3]:>4s} {row[4]:>8s} {row[5]:>6s}")
    print("-" * len(header))
    print(f"{'OVERALL':40s} {total_w:>4d} {total_l:>4d} {total_d:>4d} {overall_wr:>7.1f}% {total_time:>5.0f}s")

    # Build structured rows for SQLite (exclude error rows which have len==3)
    rows_with_stats = []
    for r in rows:
        if len(r) == 6:
            label, w, l, d, wr, t = r
            rows_with_stats.append((label, int(w), int(l), int(d), float(wr.rstrip('%')), t))

    # Save to SQLite (primary storage)
    from rl.results_db import ResultsDB
    db = ResultsDB()
    db.add_tournament(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M'),
        agent=our_path,
        games_per_opp=args.games,
        note=args.note or '',
        matchups=[{"opponent": label, "w": w, "l": l, "d": d, "wr": wr}
                  for label, w, l, d, wr, _ in rows_with_stats],
        overall={"w": total_w, "l": total_l, "d": total_d, "wr": overall_wr},
        total_time=total_time)
    db.close()
    print(f"\nResults saved to model/results.db")

    # Optional: also append to eval_results.txt (backup)
    if args.txt_backup:
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, "a") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"Tournament: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Agent: {our_path}\n")
            f.write(f"Games per opponent: {args.games}\n")
            if args.note:
                f.write(f"Note: {args.note}\n")
            f.write(f"{'='*70}\n")
            for row in rows:
                if len(row) == 3:
                    f.write(f"  {row[0]:40s} {row[1]:>4s} — {row[2]}\n")
                else:
                    f.write(f"  {row[0]:40s} W={row[1]:>3s} L={row[2]:>3s} D={row[3]:>3s} wr={row[4]:>6s}\n")
            f.write(f"  {'OVERALL':40s} W={total_w:3d} L={total_l:3d} D={total_d:3d} wr={overall_wr:5.1f}%\n")
            f.write(f"  Total time: {total_time:.0f}s\n")
        print(f"  Also appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
