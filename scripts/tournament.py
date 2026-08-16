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
from pathlib import Path

from scripts._common import AGENT_DIR, load_agent, make_env
from rl.results_db import ResultsDB as ProjectResultsDB

PUBLIC_AGENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "public_agents")
SMOKE_SUBMISSION = os.path.join(
    PUBLIC_AGENTS_DIR,
    "submissions",
    "smoke",
    "submission_smoke.tar.gz",
)
RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "model", "eval_results.txt")


def find_agents() -> list[tuple[str, str]]:
    """Discover all public agents under public_agents/."""
    agents = []
    for root, dirs, files in os.walk(PUBLIC_AGENTS_DIR):
        if "smoke" in root.split(os.sep):
            continue
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
                if "smoke" in f:
                    continue
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


def _parse_deck_lines(lines) -> list[int] | None:
    """Parse deck.csv lines into a sorted list of 60 card IDs, or None."""
    card_ids = []
    for line in lines:
        line = line.strip().rstrip(",")
        if line:
            try:
                card_ids.append(int(line))
            except ValueError:
                continue
    if len(card_ids) != 60:
        return None
    return sorted(card_ids)


def _read_deck_csv(path: str) -> list[int] | None:
    """Read a deck.csv file and return a sorted list of 60 card IDs, or None if not found."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return _parse_deck_lines(f)


def _read_deck_from_tar(tar_path: str) -> list[int] | None:
    """Read deck.csv from inside a submission tarball.

    Packaged submissions carry their deck at the archive root, so unlike a
    plain agent directory there is no sibling file to read.
    """
    import tarfile
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            member = next((m for m in tar.getmembers()
                           if os.path.basename(m.name) == "deck.csv"
                           and m.name.count("/") <= 1), None)
            if member is None:
                return None
            fh = tar.extractfile(member)
            if fh is None:
                return None
            return _parse_deck_lines(fh.read().decode("utf-8").splitlines())
    except (tarfile.TarError, OSError, UnicodeDecodeError):
        return None


def _find_or_create_deck(db, card_ids: list[int]) -> int:
    """Find an existing deck by card composition or create a new one. Returns deck_id.
    
    Delegates to the strict cryptographic SHA256 digest of the database to guarantee
    zero false-positive aggregations.
    """
    return db.get_or_create_deck(card_ids, source="local")


def _get_test_decks(db, source: str = "remote", n_top: int = 4, default_card_ids: list[int] | None = None) -> list[tuple[list[int], int | None]]:
    """Pick decks for an agent to sweep over.

    Includes the default deck plus up to ``n_top`` top decks from the ``deck_elo``
    table for the requested source tier.
    Decks with >=90% card overlap with an already-queued deck are deduplicated.
    Returns list of ``(card_ids, deck_id)`` tuples.
    """
    from collections import Counter

    decks = []
    if default_card_ids is None:
        default_card_ids = _read_deck_csv(os.path.join(AGENT_DIR, "deck.csv"))
    if default_card_ids:
        decks.append((default_card_ids, None))  # deck_id resolved later

    # Build Counter for default deck for dedup
    seen_counters = []
    if default_card_ids:
        seen_counters.append(Counter(default_card_ids))

    top = db.get_top_decks(n=n_top, source=source)
    for d in top:
        deck_id = d["id"]
        known = db.conn.execute(
            "SELECT card_id, quantity FROM deck_cards WHERE deck_id = ?",
            (deck_id,)
        ).fetchall()
        card_ids = []
        for cid, qty in known:
            card_ids.extend([cid] * qty)
        counter = Counter(card_ids)

        # Skip if composition matches an already-queued deck
        is_dup = False
        for seen_c in seen_counters:
            all_cards = set(counter) | set(seen_c)
            overlap = sum(min(counter.get(c, 0), seen_c.get(c, 0)) for c in all_cards)
            total = sum(max(counter.get(c, 0), seen_c.get(c, 0)) for c in all_cards)
            if total > 0 and overlap / total >= 0.9:
                is_dup = True
                break
        if not is_dup:
            seen_counters.append(counter)
            decks.append((card_ids, deck_id))

    return decks


def play(env, a, b) -> tuple[int, str]:
    """Run one game; return (result, html_replay). result: +1=P0 wins, -1=loses, 0=draw."""
    env.reset()
    env.run([a, b])
    r0, r1 = (s.reward for s in env.steps[-1])
    html = env.render(mode="html")
    return (1 if r0 > r1 else (-1 if r0 < r1 else 0)), html


from collections import Counter


def _identify_deck(db, card_ids):
    """Identify a deck from a 60-card list. Returns deck_id.

    Uses quantity-aware matching: overlap = sum(min(c1, c2)) for each card,
    total = sum(max(c1, c2)) across all cards in both decks.
    """
    card_counter = Counter(card_ids)
    decks = db.conn.execute("SELECT id FROM decks").fetchall()
    for (deck_id,) in decks:
        known = db.conn.execute("SELECT card_id, quantity FROM deck_cards WHERE deck_id = ?", (deck_id,)).fetchall()
        known_counter = Counter({cid: qty for cid, qty in known})
        all_cards = set(card_counter) | set(known_counter)
        overlap = sum(min(card_counter.get(c, 0), known_counter.get(c, 0)) for c in all_cards)
        total = sum(max(card_counter.get(c, 0), known_counter.get(c, 0)) for c in all_cards)
        if total > 0 and overlap / total >= 0.9:
            return deck_id
    # Create new deck
    name = f"arena_deck_{hash(frozenset(card_counter.items())) % 100000}"
    deck_id = db.add_deck(name, "arena", archetype=None)
    db.add_deck_cards(deck_id, list(card_counter.items()))
    return deck_id


def _clean_agent_label(path: str) -> str:
    """Extract clean agent name from file path, handling generic filenames like submission.tar.gz."""
    if not path:
        return "Unknown"
    base = os.path.basename(path)
    if base in ("submission.tar.gz", "submission.tgz", "submission", "main.py"):
        parent = os.path.basename(os.path.dirname(path))
        if parent and parent not in ("submissions", "models"):
            return parent
    return base


def _agent_deck_path(agent_path: str, agent_module) -> str:
    """Return the deck shipped with the loaded agent.

    A packaged submission is extracted to a temporary directory by
    ``load_agent``.  Using the repository's ``agent/deck.csv`` for that
    submission silently labels tournament rows with the wrong deck.
    """
    module_file = getattr(agent_module, "__file__", None)
    if module_file:
        return os.path.join(os.path.dirname(os.path.abspath(module_file)), "deck.csv")
    if os.path.isfile(agent_path):
        return os.path.join(os.path.dirname(os.path.abspath(agent_path)), "deck.csv")
    return os.path.join(os.path.abspath(agent_path), "deck.csv")


def _resolve_deck_human_info(db, deck_id: int | None, is_our_deck: bool = False) -> dict:
    """Resolve human info for deck_id. If is_our_deck is True, brands cleanly as Agent Submission Deck."""
    if deck_id is None:
        return {"nick": "Submission Deck", "remote_elo": None, "local_elo": None, "local_games": 0, "archetype": "N/A"}

    d_row = db.conn.execute("SELECT name, archetype, source FROM decks WHERE id = ?", (deck_id,)).fetchone()
    archetype = d_row['archetype'] if d_row and d_row['archetype'] else "N/A"
    raw_name = d_row['name'] if d_row and d_row['name'] else f"deck_{deck_id}"

    if is_our_deck:
        nick = "Submission Deck"
        remote_elo = None
    else:
        team_row = db.conn.execute("""
            SELECT t.display_name, COUNT(*) as cnt
            FROM match_participants mp
            JOIN teams t ON mp.team_id = t.id
            WHERE mp.deck_id = ?
            GROUP BY t.display_name
            ORDER BY cnt DESC
            LIMIT 1
        """, (deck_id,)).fetchone()

        nick = team_row['display_name'] if team_row and team_row['display_name'] else raw_name
        if nick.startswith("replay_deck_"):
            nick = f"Deck #{deck_id}"

        remote_row = db.conn.execute("SELECT elo FROM deck_elo WHERE deck_id = ? AND source = 'remote'", (deck_id,)).fetchone()
        remote_elo = remote_row['elo'] if remote_row else None

    local_row = db.conn.execute("SELECT elo, games_played FROM deck_elo WHERE deck_id = ? AND source = 'local'", (deck_id,)).fetchone()
    local_elo = local_row['elo'] if local_row else None
    local_games = local_row['games_played'] if local_row else 0

    inv_data = db.get_invariant_deck_elo(deck_id, source="local") if hasattr(db, "get_invariant_deck_elo") else {}
    local_elo_inv = inv_data.get("elo_invariant", local_elo)

    return {
        "nick": nick,
        "remote_elo": remote_elo,
        "local_elo": local_elo,
        "local_elo_invariant": local_elo_inv,
        "local_games": local_games,
        "archetype": archetype
    }


def _record_match_card_usage(db, match_id, side, action):
    """Persist observed local-deck cards without violating catalog FKs."""

    for card_id, quantity in Counter(action).items():
        card_id = int(card_id)
        db.conn.execute(
            """
            INSERT INTO cards (id, name)
            VALUES (?, ?)
            ON CONFLICT(id) DO NOTHING
            """,
            (card_id, f"Card {card_id}"),
        )
        db.conn.execute(
            """
            INSERT OR IGNORE INTO match_card_usage (
                match_id, card_id, player_side, quantity
            ) VALUES (?, ?, ?, ?)
            """,
            (match_id, card_id, side, quantity),
        )


def save_match_replay(
    db,
    matchup_id,
    game_index,
    our_side,
    result,
    replay_json,
    our_deck_id=None,
    opp_deck_id=None,
    our_agent_path="agent/main.py",
):
    """Save full replay data from a completed game to SQLite.

    Args:
        db: ResultsDB instance.
        matchup_id: ID of the parent matchup row.
        game_index: Index of this game within the matchup.
        our_side: Which player index was ours (0 or 1).
        result: +1 win, -1 loss, 0 draw from our perspective.
        replay_json: Parsed JSON dict from env.render(mode='json').
        our_deck_id: Optional deck ID for our deck.
        opp_deck_id: Optional deck ID for opponent's deck.
    Returns:
        match_id: The inserted match ID.
    """
    steps = replay_json.get('steps', [])

    # Create match
    match_id = db.add_match(
        matchup_id=matchup_id,
        game_index=game_index,
        source='local',
        our_agent=our_agent_path,
        our_deck_id=our_deck_id,
        opp_agent='opponent',
        opp_deck_id=opp_deck_id,
        our_side=our_side,
        result=result,
        n_steps=len(steps)
    )

    if not match_id:
        return match_id

    if our_deck_id is not None:
        # Pre-computed deck ID(s) — use them and save card usage from replay
        for step in steps:
            for side, player_data in enumerate(step):
                if isinstance(player_data, dict):
                    action = player_data.get('action', [])
                else:
                    action = getattr(player_data, 'action', [])
                if len(action) == 60:
                    _record_match_card_usage(db, match_id, side, action)
            # Only need the first step with a deck action
            if any(
                len(s[0].get('action', []) if isinstance(s[0], dict) else getattr(s[0], 'action', []) or []) == 60
                for s in steps[:1]
            ):
                break
        db.conn.execute(
            "UPDATE matches SET our_deck_id = ?, opp_deck_id = ? WHERE id = ?",
            (our_deck_id, opp_deck_id, match_id))
    else:
        # No pre-computed IDs — identify decks from replay content
        deck_ids = {}
        for step in steps:
            for side, player_data in enumerate(step):
                if isinstance(player_data, dict):
                    action = player_data.get('action', [])
                else:
                    action = getattr(player_data, 'action', [])
                if len(action) == 60:
                    _record_match_card_usage(db, match_id, side, action)
                    deck_ids[side] = _identify_deck(db, action)
            if deck_ids:
                break
        if deck_ids:
            our_id = deck_ids.get(our_side)
            opp_id = deck_ids.get(1 - our_side)
            db.conn.execute(
                "UPDATE matches SET our_deck_id = ?, opp_deck_id = ? WHERE id = ?",
                (our_id, opp_id, match_id))

    # Save steps
    for step_num, step in enumerate(steps):
        for player_idx, player_data in enumerate(step):
            # player_data may be a dict (from JSON) or a state object
            if isinstance(player_data, dict):
                obs = player_data.get('observation', {})
                action_val = player_data.get('action', [])
                status_val = player_data.get('status', 'UNKNOWN')
                reward_val = player_data.get('reward', 0)
            else:
                obs = getattr(player_data, 'observation', {}) or {}
                action_val = getattr(player_data, 'action', []) or []
                status_val = getattr(player_data, 'status', 'UNKNOWN')
                reward_val = getattr(player_data, 'reward', 0) or 0

            current = obs.get('current') or {}
            select = obs.get('select') or {}
            logs = obs.get('logs') or []

            # Normalize action to list
            if isinstance(action_val, (int, float)):
                action_list = [int(action_val)]
            elif isinstance(action_val, list):
                action_list = action_val
            else:
                action_list = []

            cursor = db.conn.execute(
                """INSERT OR IGNORE INTO match_steps (match_id, step_num, player_idx, turn, select_type,
                   select_context, n_options, action, status, reward)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (match_id, step_num, player_idx,
                 current.get('turn', 0) if current else 0,
                 select.get('type') if select else None,
                 select.get('context') if select else None,
                 len(select.get('option', [])) if select and 'option' in select else 0,
                 str(action_list),
                 str(status_val),
                 int(reward_val) if reward_val is not None else 0)
            )

            if cursor.rowcount == 0:
                row = db.conn.execute(
                    "SELECT id FROM match_steps WHERE match_id = ? AND step_num = ? AND player_idx = ?",
                    (match_id, step_num, player_idx)
                ).fetchone()
                step_id = row['id'] if row else None
            else:
                step_id = cursor.lastrowid

            if not step_id:
                continue

            # Save options
            if select and 'option' in select:
                for opt_idx, opt in enumerate(select['option']):
                    opt_type = opt.get('type', 0) if isinstance(opt, dict) else 0
                    was_selected = 1 if opt_idx in action_list else 0
                    db.conn.execute(
                        "INSERT OR IGNORE INTO step_options (step_id, option_idx, option_type, was_selected) VALUES (?, ?, ?, ?)",
                        (step_id, opt_idx, opt_type, was_selected)
                    )

            # Save events (logs)
            for log in logs:
                if isinstance(log, dict):
                    db.conn.execute(
                        """INSERT OR IGNORE INTO step_events (step_id, event_type, player_idx, card_id, serial,
                           target_card_id, target_serial, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (step_id, log.get('type', 0), log.get('playerIndex'),
                         log.get('cardId'), log.get('serial'),
                         log.get('cardIdTarget'), log.get('serialTarget'),
                         log.get('value'))
                    )

            # Save board snapshot
            if current and 'players' in current:
                for pidx, player in enumerate(current['players']):
                    cursor = db.conn.execute(
                        """INSERT OR IGNORE INTO board_snapshots (step_id, player_idx, turn, deck_count,
                           hand_count, prize_count, discard_count, poisoned, burned, asleep,
                           paralyzed, confused) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (step_id, pidx, current.get('turn', 0),
                         player.get('deckCount', 0),
                         player.get('handCount', 0),
                         len([p for p in player.get('prize', []) if p is not None]) if 'prize' in player else 0,
                         len(player.get('discard', [])) if 'discard' in player else 0,
                         int(player.get('poisoned', False)),
                         int(player.get('burned', False)),
                         int(player.get('asleep', False)),
                         int(player.get('paralyzed', False)),
                         int(player.get('confused', False)))
                    )

                    if cursor.rowcount == 0:
                        row = db.conn.execute(
                            "SELECT id FROM board_snapshots WHERE step_id = ? AND player_idx = ?",
                            (step_id, pidx)
                        ).fetchone()
                        snapshot_id = row['id'] if row else None
                    else:
                        snapshot_id = cursor.lastrowid

                    if not snapshot_id:
                        continue

                    # Save Pokemon on field
                    for slot_name in ('active', 'bench'):
                        for slot_idx, pokemon in enumerate(player.get(slot_name, [])):
                            if pokemon is None:
                                continue
                            db.conn.execute(
                                """INSERT OR IGNORE INTO pokemon_on_field (snapshot_id, slot, slot_idx, card_id,
                                   serial, hp, max_hp, n_energies, n_tools, n_preevo)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (snapshot_id, slot_name, slot_idx,
                                 pokemon.get('id', 0), pokemon.get('serial', 0),
                                 pokemon.get('hp', 0), pokemon.get('maxHp', 0),
                                 len(pokemon.get('energies', [])) if 'energies' in pokemon else 0,
                                 len(pokemon.get('tools', [])) if 'tools' in pokemon else 0,
                                 len(pokemon.get('preEvolution', [])) if 'preEvolution' in pokemon else 0)
                            )

    db.conn.commit()
    return match_id


def run_matchup(env, our_agent, opp_agent, n_games: int):
    """Play n_games; return (wins, losses, draws, last_replay_html, game_results).

    game_results is a list of dicts with keys: game_index, our_side, result, replay_json."""
    import json as _json
    wins = losses = draws = 0
    last_html = ""
    game_results = []
    for i in range(n_games):
        if i % 2 == 0:
            r, html = play(env, our_agent, opp_agent)
            our_side = 0
        else:
            r, html = play(env, opp_agent, our_agent)
            r = -r
            our_side = 1
        if i == n_games - 1:
            last_html = html
        wins += r == 1
        losses += r == -1
        draws += r == 0

        # Capture full replay JSON after each game
        replay_json = _json.loads(env.render(mode='json'))
        game_results.append({
            "game_index": i,
            "our_side": our_side,
            "result": r,
            "replay_json": replay_json,
        })
    return wins, losses, draws, last_html, game_results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", type=str, default=None, action="append",
                   help="Custom main agent path (submission tarball or main.py). May be repeated to evaluate multiple agents.")
    p.add_argument("--games", "-n", type=int, default=20, help="Games per opponent")
    p.add_argument("--workers", "-w", type=int, default=4, help="Parallel worker processes for games execution (default: 4)")
    p.add_argument("--opponent", type=str, default=None, action="append",
                   help="Opponent path (submission tarball or agent main.py). "
                        "May be repeated to select a custom opponent subset "
                        "instead of iterating public_agents/.")
    p.add_argument("--note", type=str, default=None,
                   help="Annotation for this run (saved in SQLite)")
    p.add_argument("--no-sweep", action="store_true", default=False,
                   help="Disable deck sweep (only use default deck)")
    p.add_argument("--sweep-source", choices=["remote", "local"], default="remote",
                   help="Elo tier the sweep pulls its top decks from "
                        "(default: remote, populated from Kaggle replays)")
    p.add_argument("--skip-baselines", action="store_true", default=False,
                   help="Skip the built-in random+first baseline opponents. "
                        "Useful for round-robin runs where the noise floor "
                        "was already measured elsewhere.")
    p.add_argument("--new-season", action="store_true", default=False,
                   help="Start a new sequential season in SQLite (deactivates current active season)")
    p.add_argument("--reset-local-elo", action="store_true", default=False,
                   help="Reset local Elo metrics to INITIAL_ELO without touching remote Kaggle Elo data")
    p.add_argument("--clear-local-matches", action="store_true", help="Delete local match history without touching remote Kaggle replay matches")
    p.add_argument("--top-decks", type=int, default=4, help="Number of top remote/local decks to include in sweep (default: 4)")
    p.add_argument("--opp-top-decks", type=int, default=0,
                   help="Number of top remote/local decks to include in sweep for opponents (default: 0)")
    p.add_argument("--emit-best-performing-deck", nargs="?", const=True, default=False,
                   help="Export the 60-card CSV of the best performing deck for our agent to its directory or optional path")
    p.add_argument("--report-json", type=str, default=None,
                   help="Write the structured per-opponent/per-deck result "
                        "table to this JSON path, so callers can consume "
                        "results without scraping stdout.")
    p.add_argument("--txt-backup", action="store_true", default=False,
                   help="Also append results to eval_results.txt (backup)")
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Use only public_agents/submissions/smoke/submission_smoke.tar.gz "
            "as our agent and force a no-sweep run"
        ),
    )
    args = p.parse_args()

    root_db_path = Path(__file__).resolve().parent.parent / "model" / "results.db"
    _db = ProjectResultsDB(db_path=root_db_path)

    if args.new_season:
        season = _db.start_new_season()
        print(f"✓ Started new active season: {season['name']} (ID: {season['id']})")

    if args.reset_local_elo:
        count = _db.reset_local_elo()
        print(f"✓ Reset local Elo metrics for {count} decks to 600.0 (remote Elo preserved)")

    if args.clear_local_matches:
        count = _db.clear_local_matches()
        print(f"Cleared {count} local matches.")

    if (args.new_season or args.reset_local_elo or args.clear_local_matches) and not args.agent and not args.smoke:
        _db.close()
        return

    n_sync = _db.sync_kaggle_leaderboard_elos()
    if n_sync > 0:
        print(f"✓ Synchronized official Kaggle Leaderboard ratings for {n_sync} decks in database.")

    _remote_count = _db.conn.execute("SELECT COUNT(*) FROM card_elo WHERE source='remote'").fetchone()[0]
    _db.close()
    if _remote_count == 0:
        print("\n[auto-update] No remote card Elo data. Processing Kaggle replays...")
        import subprocess as _sp
        today = datetime.now().strftime("%Y-%m-%d")
        _sp.run([sys.executable, "-m", "scripts.build_card_stats", "--date", today],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), check=False)
        print("[auto-update] Remote data updated.\n")

    # Determine agent paths to evaluate
    if args.agent:
        agent_paths = args.agent if isinstance(args.agent, list) else [args.agent]
    elif args.smoke:
        agent_paths = [SMOKE_SUBMISSION]
    else:
        agent_paths = [os.path.join(AGENT_DIR, "main.py")]

    for agent_idx, our_path in enumerate(agent_paths, start=1):
        if len(agent_paths) > 1:
            clean_name = _clean_agent_label(our_path)
            print(f"\n=======================================================")
            print(f"=== Tournament [{agent_idx}/{len(agent_paths)}]: {clean_name} ({our_path})")
            print(f"=======================================================\n", flush=True)

        if args.smoke and not os.path.isfile(our_path):
            p.error(
                "--smoke requires the isolated local artifact at "
                f"{SMOKE_SUBMISSION}; build it with `uv run tcg-build --smoke`"
            )

        _run_single_tournament(our_path, args, root_db_path)


def _run_single_tournament(our_path: str, args: argparse.Namespace, root_db_path: Path):
    our_agent, our_module = load_agent(our_path, return_module=True)
    our_deck_path = _agent_deck_path(our_path, our_module)
    env = make_env()

    # Baselines + public agents. ``--opponent`` may be repeated to select a
    # custom subset instead of iterating everything under public_agents/.
    opponents = (
        [] if args.skip_baselines
        else [("random", "random"), ("first", "first")]
    )
    if args.opponent:
        for opp_path in args.opponent:
            label = _clean_agent_label(opp_path)
            opponents.append((label, opp_path))
    else:
        opponents.extend(
            (label, path)
            for label, path in find_agents()
            if os.path.realpath(path) != os.path.realpath(our_path)
        )

    # Prepare test decks for sweep
    db = ProjectResultsDB(db_path=root_db_path)
    # The sweep belongs exclusively to our agent. Opponent agents are resolved
    # once below and their own deck files/callables remain fixed for every
    # alternative deck we test against them.
    if args.smoke:
        default_card_ids = _read_deck_from_tar(our_path)
        our_test_decks = (
            [(default_card_ids, None)] if default_card_ids is not None else []
        )
    else:
        our_test_decks = _get_test_decks(db, source=args.sweep_source, n_top=args.top_decks)
        default_card_ids = _read_deck_csv(our_deck_path)

    # Read original deck to restore after sweep
    original_deck = (
        open(our_deck_path).read()
        if not args.smoke and os.path.exists(our_deck_path)
        else None
    )

    # Read opponent deck IDs (deck.csv from their agent dir, if available)
    opp_deck_ids = {}
    for label, opp_path in opponents:
        if opp_path in ("random", "first"):
            opp_deck_ids[label] = None
            continue
        if opp_path.endswith((".tar.gz", ".tgz")):
            # Packaged submission: the deck lives inside the archive.
            opp_card_ids = _read_deck_from_tar(opp_path)
        else:
            # "…/agent/main.py" sits next to its deck.csv; a bare directory is
            # already the right place to look.
            opp_dir = opp_path if os.path.isdir(opp_path) else os.path.dirname(opp_path)
            opp_card_ids = _read_deck_csv(os.path.join(opp_dir, "deck.csv"))
        opp_deck_ids[label] = _find_or_create_deck(db, opp_card_ids) if opp_card_ids else None

    total_w = total_l = total_d = 0
    rows = []
    structured_rows: list[dict] = []
    all_game_results = []  # (label, deck_id, game_results) tuples
    start_time = time.time()

    if args.note:
        print(f"Note: {args.note}", flush=True)

    do_sweep = (
        not args.smoke
        and not args.no_sweep
        and len(our_test_decks) > 1
    )
    total_blocks = len(opponents) * (len(our_test_decks) if do_sweep else 1)
    completed_blocks = 0

    # Identify default deck ID
    default_deck_id = None
    if our_test_decks:
        first_cards, _first_id = our_test_decks[0]
        if first_cards:
            default_deck_id = _find_or_create_deck(db, first_cards)

    if do_sweep:
        # Pre-resolve opponents and their test decks
        resolved_opponents = []
        for label, opp_path in opponents:
            try:
                opp_agent = resolve(opp_path)
            except Exception as e:
                rows.append((label, "ERROR", str(e)))
                structured_rows.append({
                    "opponent_label": label,
                    "opponent_path": opp_path,
                    "deck_id": None,
                    "wins": 0, "losses": 0, "draws": 0,
                    "wr_pct": None,
                    "elapsed_s": 0.0,
                    "error": str(e),
                })
                continue

            if args.opp_top_decks > 0 and opp_path not in ("random", "first"):
                opp_cards = _read_deck_from_tar(opp_path) if opp_path.endswith((".tar.gz", ".tgz")) else _read_deck_csv(os.path.join(opp_path if os.path.isdir(opp_path) else os.path.dirname(opp_path), "deck.csv"))
                opp_test_decks = _get_test_decks(db, source=args.sweep_source, n_top=args.opp_top_decks, default_card_ids=opp_cards)
            else:
                opp_test_decks = [(None, opp_deck_ids.get(label))]

            resolved_opponents.append((label, opp_path, opp_agent, opp_test_decks))

        total_blocks = sum(len(opp_test_decks) for _, _, _, opp_test_decks in resolved_opponents) * len(our_test_decks)
        print(f"Sweep: {len(our_test_decks)} OUR decks per opponent ({total_blocks} total matchup blocks)\n", flush=True)

        for card_ids, deck_id in our_test_decks:
            if deck_id is None and default_card_ids is not None:
                deck_id = _find_or_create_deck(db, default_card_ids)

            # Swap agent/deck.csv ONCE for our agent for this entire deck batch
            deck_csv = our_deck_path
            try:
                if card_ids:
                    with open(deck_csv, "w") as f:
                        f.write("\n".join(str(c) for c in card_ids) + "\n")
                    played = our_module.reload_deck(deck_csv)
                    if sorted(played) != sorted(card_ids):
                        raise RuntimeError(
                            f"deck reload mismatch for deck {deck_id}: agent holds "
                            f"{len(played)} cards, expected {len(card_ids)}")

                is_default = (deck_id == default_deck_id)
                our_info = _resolve_deck_human_info(db, deck_id, is_our_deck=is_default)
                rem_elo = f"{our_info['remote_elo']:.1f}" if our_info['remote_elo'] else "N/A"
                loc_elo = f"{our_info.get('local_elo_invariant', our_info['local_elo']):.1f}" if our_info.get('local_elo_invariant', our_info['local_elo']) else "N/A"

                if is_default:
                    deck_header = f"OUR AGENT DECK [NATIVO]: {our_info['nick']}"
                    elo_header = f"Local Elo: {loc_elo} ({our_info['local_games']} partidas) | Origem: Baralho Nativo da Submissao"
                else:
                    deck_header = f"BARALHO EM TESTE #{deck_id}: {our_info['nick']}"
                    elo_header = f"Elo Remoto: {rem_elo} | Elo Local: {loc_elo} ({our_info['local_games']} partidas) | Archetype: {our_info['archetype']}"

                print("=" * 105, flush=True)
                print(deck_header, flush=True)
                print(elo_header, flush=True)
                print("=" * 105, flush=True)
                print(f"{'OPONENTE':28s} {'BARALHO OPONENTE':28s} {'WIN':>5s} {'LOSS':>6s} {'DRAW':>6s}  {'WIN%':>6s}   {'TIME':>6s}   {'PROGRESS'}", flush=True)
                print("-" * 105, flush=True)

                deck_w = deck_l = deck_d = 0

                for label, opp_path, opp_agent, opp_test_decks in resolved_opponents:
                    for _opp_cards, opp_d_id in opp_test_decks:
                        opp_info = _resolve_deck_human_info(db, opp_d_id)
                        opp_nick = opp_info['nick']
                        if opp_d_id is None:
                            opp_deck_disp = "Nativo (Tarball)"
                        elif opp_nick and opp_nick != label and not opp_nick.startswith("Deck #"):
                            opp_deck_disp = f"Deck #{opp_d_id} ({opp_nick})"
                        else:
                            opp_deck_disp = f"Deck #{opp_d_id}"

                        t0 = time.time()
                        w, l, d, replay_html, game_results = run_matchup(
                            env, our_agent, opp_agent, args.games)
                        elapsed = time.time() - t0

                        wr = w / max(w + l, 1) * 100
                        total_w += w; total_l += l; total_d += d
                        deck_w += w; deck_l += l; deck_d += d

                        deck_label = f"{label} [{opp_deck_disp}] vs {our_info['nick']}"
                        rows.append((deck_label, w, l, d, wr, elapsed, replay_html))
                        structured_rows.append({
                            "opponent_label": label,
                            "opponent_path": opp_path,
                            "deck_id": deck_id,
                            "opp_deck_id": opp_d_id,
                            "wins": w, "losses": l, "draws": d,
                            "wr_pct": wr,
                            "elapsed_s": elapsed,
                            "error": None,
                        })
                        all_game_results.append((label, deck_id, game_results))
                        completed_blocks += 1

                        elapsed_suite = time.time() - start_time
                        avg_block = elapsed_suite / completed_blocks
                        rem_blocks = max(0, total_blocks - completed_blocks)
                        eta_sec = int(avg_block * rem_blocks)
                        eta_m, eta_s = divmod(eta_sec, 60)
                        eta_h, eta_m = divmod(eta_m, 60)
                        eta_fmt = f"{eta_h}h{eta_m:02d}m" if eta_h else f"{eta_m}m{eta_s:02d}s"

                        print(f"{label:28s} {opp_deck_disp:28s} {w:5d} {l:6d} {d:6d}  {wr:5.1f}%  {elapsed:5.1f}s   [{completed_blocks}/{total_blocks}] ETA: {eta_fmt}", flush=True)

                deck_wr = deck_w / max(deck_w + deck_l, 1) * 100
                print("-" * 88, flush=True)
                print(f"SUBTOTAL ({our_info['nick']}): {deck_w:5d} {deck_l:6d} {deck_d:6d}  {deck_wr:5.1f}%", flush=True)
                print("=" * 88 + "\n", flush=True)
            finally:
                if original_deck is not None:
                    with open(deck_csv, "w") as f:
                        f.write(original_deck)
                    our_module.reload_deck(deck_csv)
    else:
        # No sweep: run with default deck only
        for label, opp_path in opponents:
            opp_agent = resolve(opp_path)
            t0 = time.time()
            w, l, d, replay_html, game_results = run_matchup(
                env, our_agent, opp_agent, args.games)
            elapsed = time.time() - t0
            wr = w / max(w + l, 1) * 100
            total_w += w; total_l += l; total_d += d

            if default_card_ids:
                deck_id = _find_or_create_deck(db, default_card_ids)
            else:
                deck_id = None
            opp_deck_id = opp_deck_ids.get(label)

            deck_label = f"{label} [deck:{deck_id}]" if deck_id else label
            rows.append((deck_label, w, l, d, wr, elapsed, replay_html))
            structured_rows.append({
                "opponent_label": label,
                "opponent_path": opp_path,
                "deck_id": deck_id,
                "opp_deck_id": opp_deck_id,
                "wins": w, "losses": l, "draws": d,
                "wr_pct": wr,
                "elapsed_s": elapsed,
                "error": None,
            })
            all_game_results.append((label, deck_id, game_results))
            print(f"{deck_label:32s} {w:5d} {l:6d} {d:6d}  {wr:5.1f}%  {elapsed:5.1f}s", flush=True)
        completed_blocks += 1
        elapsed_suite = time.time() - start_time
        avg_block = elapsed_suite / completed_blocks
        rem_blocks = max(0, total_blocks - completed_blocks)
        eta_sec = int(avg_block * rem_blocks)
        eta_m, eta_s = divmod(eta_sec, 60)
        eta_h, eta_m = divmod(eta_m, 60)
        eta_fmt = f"{eta_h}h{eta_m:02d}m" if eta_h else f"{eta_m}m{eta_s:02d}s"
        print(f"  {deck_label:40s} W={w:3d} L={l:3d} D={d:3d} wr={wr:5.1f}% ({elapsed:.0f}s) | [{completed_blocks}/{total_blocks} ETA: {eta_fmt}]",
              flush=True)

    total_time = time.time() - start_time
    overall_wr = total_w / max(total_w + total_l, 1) * 100

    # Identify default deck ID
    default_deck_id = None
    if our_test_decks:
        first_cards, _first_id = our_test_decks[0]
        if first_cards:
            default_deck_id = _find_or_create_deck(db, first_cards)

    # Print table
    print()
    header = f"{'Opponent':40s} {'W':>4s} {'L':>4s} {'D':>4s} {'WinRate':>8s} {'Time':>6s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        if len(row) == 3:
            print(f"{row[0]:40s} {str(row[1]):>4s} — {row[2]}")
        else:
            print(f"{row[0]:40s} {str(row[1]):>4s} {str(row[2]):>4s} {str(row[3]):>4s} {row[4]:>7.1f}% {row[5]:>5.0f}s")
    print("-" * len(header))

    # Calculate best deck ID
    best_deck_id = None
    best_wr = -1.0
    deck_stats = {}  # deck_id -> {w, l, d, n_opponents}
    if do_sweep and all_game_results:
        for _label, deck_id, game_results in all_game_results:
            if deck_id not in deck_stats:
                deck_stats[deck_id] = {"w": 0, "l": 0, "d": 0, "n_opponents": 0,
                                       "opponents": set()}
            ds = deck_stats[deck_id]
            for gr in game_results:
                r = gr["result"]
                ds["w"] += r == 1
                ds["l"] += r == -1
                ds["d"] += r == 0
            if _label not in ds["opponents"]:
                ds["opponents"].add(_label)
                ds["n_opponents"] = len(ds["opponents"])

        for deck_id, ds in deck_stats.items():
            if deck_id is not None:
                wr = ds["w"] / max(ds["w"] + ds["l"], 1) * 100
                if wr > best_wr:
                    best_wr = wr
                    best_deck_id = deck_id

    # Print disaggregated OVERALL rows
    if do_sweep and default_deck_id in deck_stats:
        d_ds = deck_stats[default_deck_id]
        d_wr = d_ds["w"] / max(d_ds["w"] + d_ds["l"], 1) * 100
        print(f"{'OVERALL (DEFAULT DECK: #' + str(default_deck_id) + ')':40s} {d_ds['w']:>4d} {d_ds['l']:>4d} {d_ds['d']:>4d} {d_wr:>7.1f}% {total_time:>5.0f}s")

    if do_sweep and best_deck_id is not None and best_deck_id != default_deck_id:
        b_ds = deck_stats[best_deck_id]
        b_wr = b_ds["w"] / max(b_ds["w"] + b_ds["l"], 1) * 100
        print(f"{'OVERALL (BEST DECK:    #' + str(best_deck_id) + ')':40s} {b_ds['w']:>4d} {b_ds['l']:>4d} {b_ds['d']:>4d} {b_wr:>7.1f}% {total_time:>5.0f}s")

    print(f"{'OVERALL (SWEEP TOTAL:  All Decks)':40s} {total_w:>4d} {total_l:>4d} {total_d:>4d} {overall_wr:>7.1f}% {total_time:>5.0f}s")
    print("-" * len(header))

    # Structured report for programmatic consumers
    if args.report_json:
        import json as _json
        report = {
            "our_agent": our_path,
            "sweep": do_sweep,
            "sweep_source": args.sweep_source if do_sweep else None,
            "games_per_opponent": args.games,
            "note": args.note,
            "rows": structured_rows,
            "overall": {
                "wins": total_w,
                "losses": total_l,
                "draws": total_d,
                "wr_pct": overall_wr,
                "elapsed_s": total_time,
            },
        }
        os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
        tmp = f"{args.report_json}.tmp"
        with open(tmp, "w") as f:
            _json.dump(report, f, indent=2)
        os.replace(tmp, args.report_json)
        print(f"[tournament] report written: {args.report_json}", flush=True)

    # Deck performance summary with rich metadata
    if do_sweep and deck_stats:
        print()
        print("=" * 105)
        print(f"{'DECK PERFORMANCE SUMMARY':^105s}")
        print("=" * 105)
        header_fmt = f"{'Deck ID':>8s}  {'Deck Name / Archetype':<48s} {'Remote Elo':>10s} {'W':>5s} {'L':>5s} {'D':>5s} {'WinRate':>9s} {'Opps':>5s}"
        print(header_fmt)
        print("-" * 105)
        for deck_id, ds in sorted(deck_stats.items(),
                                   key=lambda x: x[1]["w"] / max(x[1]["w"] + x[1]["l"], 1),
                                   reverse=True):
            wr = ds["w"] / max(ds["w"] + ds["l"], 1) * 100
            if deck_id == default_deck_id:
                deck_row = db.conn.execute("SELECT name, archetype FROM decks WHERE id = ?", (deck_id,)).fetchone()
                elo_row = db.conn.execute("SELECT elo FROM deck_elo WHERE deck_id = ? AND source = 'remote'", (deck_id,)).fetchone()
                name_part = deck_row["name"] if deck_row and deck_row["name"] else f"deck_{deck_id}"
                label_str = f"[NATIVE] {name_part}"
                elo_str = f"{elo_row['elo']:.0f}" if elo_row and elo_row["elo"] else "-"
            else:
                deck_row = db.conn.execute("SELECT name, archetype FROM decks WHERE id = ?", (deck_id,)).fetchone()
                elo_row = db.conn.execute("SELECT elo FROM deck_elo WHERE deck_id = ? AND source = 'remote'", (deck_id,)).fetchone()
                name_part = deck_row["name"] if deck_row and deck_row["name"] else f"deck_{deck_id}"
                arch_part = f" [{deck_row['archetype']}]" if deck_row and deck_row["archetype"] else ""
                label_str = f"{name_part}{arch_part}"
                elo_str = f"{elo_row['elo']:.0f}" if elo_row and elo_row["elo"] else "-"

            print(f"  {deck_id:>6d}  {label_str:<48.48s} {elo_str:>10s} {ds['w']:>5d} {ds['l']:>5d} {ds['d']:>5d} {wr:>8.1f}% {ds['n_opponents']:>5d}")
        print("-" * 105)

        if args.emit_best_performing_deck and best_deck_id is not None:
            known = db.conn.execute(
                "SELECT card_id, quantity FROM deck_cards WHERE deck_id = ?",
                (best_deck_id,)
            ).fetchall()
            best_card_ids = []
            for cid, qty in known:
                best_card_ids.extend([cid] * qty)

            if isinstance(args.emit_best_performing_deck, str):
                out_csv = args.emit_best_performing_deck
            else:
                agent_dir = os.path.dirname(our_path) if os.path.isfile(our_path) else our_path
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                agent_stem = Path(our_path).name.replace(".tar.gz", "").replace(".pkl", "")
                out_csv = os.path.join(agent_dir, f"top_deck_{agent_stem}_{ts}.csv")

            os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
            with open(out_csv, "w") as f:
                f.write("\n".join(str(c) for c in best_card_ids) + "\n")
            print(f"\n[best-deck] Exported best performing deck (Deck ID: {best_deck_id}, WinRate: {best_wr:.1f}%) to {out_csv}", flush=True)

    # Build structured rows for SQLite (exclude error rows which have len==3)
    rows_with_stats = []
    for r in rows:
        if len(r) == 7:
            label, w, l, d, wr, t, _html = r
            rows_with_stats.append((label, int(w), int(l), int(d), float(wr), t))

    # Save to SQLite (primary storage)
    tournament_id = db.add_tournament(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M'),
        agent=our_path,
        games_per_opp=args.games,
        note=args.note or '',
        matchups=[{"opponent": label, "w": w, "l": l, "d": d, "wr": wr}
                  for label, w, l, d, wr, _ in rows_with_stats],
        overall={"w": total_w, "l": total_l, "d": total_d, "wr": overall_wr},
        total_time=total_time)

    # Save full replay data for each game
    # Fetch matchup IDs in insertion order (rowid) to handle duplicate opponent names
    all_matchup_rows = db.conn.execute(
        "SELECT id, opponent FROM matchups WHERE tournament_id = ? ORDER BY rowid",
        (tournament_id,)).fetchall()
    matchup_iter = iter(all_matchup_rows)
    n_matches_saved = 0
    for label, deck_id, game_results in all_game_results:
        # Get next matchup in insertion order
        matchup_row = next(matchup_iter, None)
        if not matchup_row:
            continue
        matchup_id = matchup_row["id"]
        opp_did = opp_deck_ids.get(label)

        for gr in game_results:
            save_match_replay(
                db=db,
                matchup_id=matchup_id,
                game_index=gr["game_index"],
                our_side=gr["our_side"],
                result=gr["result"],
                replay_json=gr["replay_json"],
                our_deck_id=deck_id,
                opp_deck_id=opp_did,
                our_agent_path=our_path,
            )
            n_matches_saved += 1

    # Compute Elo for local matches
    if n_matches_saved > 0:
        db.compute_card_elo(source='local')
        db.compute_deck_elo(source='local')

    db.close()
    print(f"\nResults saved to model/results.db ({n_matches_saved} match replays)")

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
