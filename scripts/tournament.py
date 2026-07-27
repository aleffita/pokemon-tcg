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


def _read_deck_csv(path: str) -> list[int] | None:
    """Read a deck.csv file and return a sorted list of 60 card IDs, or None if not found."""
    if not os.path.exists(path):
        return None
    card_ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    card_ids.append(int(line))
                except ValueError:
                    continue
    if len(card_ids) != 60:
        return None
    return sorted(card_ids)


def _find_or_create_deck(db, card_ids: list[int]) -> int:
    """Find an existing deck by card composition or create a new one. Returns deck_id.

    Uses quantity-aware matching: overlap = sum(min(c1, c2)) for each card,
    total = sum(max(c1, c2)) across all cards in both decks.
    """
    from collections import Counter
    card_counter = Counter(card_ids)
    decks = db.conn.execute("SELECT id FROM decks").fetchall()
    for (deck_id,) in decks:
        known = db.conn.execute(
            "SELECT card_id, quantity FROM deck_cards WHERE deck_id = ?", (deck_id,)
        ).fetchall()
        known_counter = Counter({cid: qty for cid, qty in known})
        # Quantity-aware overlap: sum of min quantities per card
        overlap = sum(min(card_counter.get(c, 0), known_counter.get(c, 0))
                      for c in set(card_counter) | set(known_counter))
        total = sum(max(card_counter.get(c, 0), known_counter.get(c, 0))
                    for c in set(card_counter) | set(known_counter))
        if total > 0 and overlap / total >= 0.9:
            return deck_id
    name = f"arena_deck_{hash(frozenset(card_counter.items())) % 100000}"
    deck_id = db.add_deck(name, "arena", archetype=None)
    db.add_deck_cards(deck_id, list(card_counter.items()))
    return deck_id


def _get_test_decks(db) -> list[tuple[list[int], int | None]]:
    """Get test decks for sweep: [default_deck] + top 3 from local deck Elo.

    Returns list of (card_ids, deck_id) tuples. deck_id may be None for the
    default deck if it hasn't been added to the DB yet.
    Deduplicates using Counter-based matching (same as _find_or_create_deck).
    """
    from collections import Counter

    decks = []
    default_card_ids = _read_deck_csv(os.path.join(AGENT_DIR, "deck.csv"))
    if default_card_ids:
        decks.append((default_card_ids, None))  # deck_id resolved later

    # Build Counter for default deck for dedup
    seen_counters = []
    if default_card_ids:
        seen_counters.append(Counter(default_card_ids))

    top = db.get_top_decks(n=3, source='local')
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


def save_match_replay(db, matchup_id, game_index, our_side, result, replay_json, our_deck_id=None, opp_deck_id=None):
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
        our_agent='agent/main.py',
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
                    for card_id, qty in Counter(action).items():
                        db.conn.execute(
                            "INSERT OR IGNORE INTO match_card_usage (match_id, card_id, player_side, quantity) VALUES (?, ?, ?, ?)",
                            (match_id, int(card_id), side, qty))
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
                    for card_id, qty in Counter(action).items():
                        db.conn.execute(
                            "INSERT OR IGNORE INTO match_card_usage (match_id, card_id, player_side, quantity) VALUES (?, ?, ?, ?)",
                            (match_id, int(card_id), side, qty))
                    from rl.results_db import ResultsDB
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

            # Create step
            step_id = db.conn.execute(
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
            ).lastrowid

            if not step_id:
                continue

            # Save options
            if select and 'option' in select:
                for opt_idx, opt in enumerate(select['option']):
                    opt_type = opt.get('type', 0) if isinstance(opt, dict) else 0
                    was_selected = 1 if opt_idx in action_list else 0
                    db.conn.execute(
                        "INSERT INTO step_options (step_id, option_idx, option_type, was_selected) VALUES (?, ?, ?, ?)",
                        (step_id, opt_idx, opt_type, was_selected)
                    )

            # Save events (logs)
            for log in logs:
                if isinstance(log, dict):
                    db.conn.execute(
                        """INSERT INTO step_events (step_id, event_type, player_idx, card_id, serial,
                           target_card_id, target_serial, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (step_id, log.get('type', 0), log.get('playerIndex'),
                         log.get('cardId'), log.get('serial'),
                         log.get('cardIdTarget'), log.get('serialTarget'),
                         log.get('value'))
                    )

            # Save board snapshot
            if current and 'players' in current:
                for pidx, player in enumerate(current['players']):
                    snapshot_id = db.conn.execute(
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
                    ).lastrowid

                    if not snapshot_id:
                        continue

                    # Save Pokemon on field
                    for slot_name in ('active', 'bench'):
                        for slot_idx, pokemon in enumerate(player.get(slot_name, [])):
                            if pokemon is None:
                                continue
                            db.conn.execute(
                                """INSERT INTO pokemon_on_field (snapshot_id, slot, slot_idx, card_id,
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
    p.add_argument("--games", "-n", type=int, default=20, help="Games per opponent")
    p.add_argument("--opponent", type=str, default=None,
                   help="Single opponent path (skip tournament)")
    p.add_argument("--note", type=str, default=None,
                   help="Annotation for this run (saved in SQLite)")
    p.add_argument("--no-sweep", action="store_true", default=False,
                   help="Disable deck sweep (only use default deck)")
    p.add_argument("--txt-backup", action="store_true", default=False,
                   help="Also append results to eval_results.txt (backup)")
    args = p.parse_args()

    # Our agent
    our_path = os.path.join(AGENT_DIR, "main.py")
    our_agent = load_agent(our_path)
    env = make_env()

    # Auto-update: ensure remote card/deck data is populated
    from rl.results_db import ResultsDB as _DB
    _db = _DB()
    _remote_count = _db.conn.execute("SELECT COUNT(*) FROM card_elo WHERE source='remote'").fetchone()[0]
    _db.close()
    if _remote_count == 0:
        print("\n[auto-update] No remote card Elo data. Processing Kaggle replays...")
        import subprocess as _sp
        today = datetime.now().strftime("%Y-%m-%d")
        _sp.run([sys.executable, "-m", "scripts.build_card_stats", "--date", today],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), check=False)
        print("[auto-update] Remote data updated.\n")

    # Baselines + public agents
    opponents = [("random", "random"), ("first", "first")]
    if args.opponent:
        label = os.path.basename(os.path.dirname(args.opponent)) if args.opponent.endswith("main.py") else os.path.basename(args.opponent)
        opponents.append((label, args.opponent))
    else:
        opponents.extend(find_agents())

    # Prepare test decks for sweep
    from rl.results_db import ResultsDB
    db = ResultsDB()
    test_decks = _get_test_decks(db)
    default_card_ids = _read_deck_csv(os.path.join(AGENT_DIR, "deck.csv"))

    # Read original deck to restore after sweep
    original_deck = open(os.path.join(AGENT_DIR, "deck.csv")).read() if os.path.exists(os.path.join(AGENT_DIR, "deck.csv")) else None

    # Read opponent deck IDs (deck.csv from their agent dir, if available)
    opp_deck_ids = {}
    for label, opp_path in opponents:
        if opp_path in ("random", "first"):
            opp_deck_ids[label] = None
            continue
        opp_dir = os.path.dirname(opp_path) if opp_path.endswith(".tar.gz") else opp_path
        if os.path.isdir(opp_dir):
            opp_card_ids = _read_deck_csv(os.path.join(opp_dir, "deck.csv"))
            if opp_card_ids:
                opp_deck_ids[label] = _find_or_create_deck(db, opp_card_ids)
            else:
                opp_deck_ids[label] = None
        else:
            opp_deck_ids[label] = None

    total_w = total_l = total_d = 0
    rows = []
    all_game_results = []  # (label, deck_id, game_results) tuples
    start_time = time.time()

    if args.note:
        print(f"Note: {args.note}", flush=True)

    do_sweep = (not args.no_sweep) and len(test_decks) > 1
    if do_sweep:
        print(f"Sweep: {len(test_decks)} decks per opponent "
              f"(default + {len(test_decks) - 1} from deck_elo)\n", flush=True)

    for label, opp_path in opponents:
        try:
            opp_agent = resolve(opp_path)
        except Exception as e:
            rows.append((label, "ERROR", str(e)))
            continue

        if do_sweep:
            # Run each test deck against this opponent
            for card_ids, deck_id in test_decks:
                if deck_id is None and default_card_ids is not None:
                    deck_id = _find_or_create_deck(db, default_card_ids)
                deck_label = f"{label} [deck:{deck_id}]"

                # Swap agent/deck.csv
                deck_csv = os.path.join(AGENT_DIR, "deck.csv")
                try:
                    with open(deck_csv, "w") as f:
                        f.write("\n".join(str(c) for c in card_ids) + "\n")

                    t0 = time.time()
                    w, l, d, replay_html, game_results = run_matchup(
                        env, our_agent, opp_agent, args.games)
                    elapsed = time.time() - t0
                finally:
                    # Always restore original deck
                    if original_deck is not None:
                        with open(deck_csv, "w") as f:
                            f.write(original_deck)

                wr = w / max(w + l, 1) * 100
                total_w += w; total_l += l; total_d += d
                rows.append((deck_label, w, l, d, wr, elapsed, replay_html))
                all_game_results.append((label, deck_id, game_results))
                print(f"  {deck_label:40s} W={w:3d} L={l:3d} D={d:3d} wr={wr:5.1f}% ({elapsed:.0f}s)",
                      flush=True)
        else:
            # No sweep: run with default deck only
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

            deck_label = f"{label} [deck:{deck_id}]" if deck_id else label
            rows.append((deck_label, w, l, d, wr, elapsed, replay_html))
            all_game_results.append((label, deck_id, game_results))
            print(f"  {deck_label:40s} W={w:3d} L={l:3d} D={d:3d} wr={wr:5.1f}% ({elapsed:.0f}s)",
                  flush=True)

    total_time = time.time() - start_time
    overall_wr = total_w / max(total_w + total_l, 1) * 100

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
    print(f"{'OVERALL':40s} {total_w:>4d} {total_l:>4d} {total_d:>4d} {overall_wr:>7.1f}% {total_time:>5.0f}s")

    # Deck performance summary (when sweep is active)
    if do_sweep and all_game_results:
        deck_stats = {}  # deck_id -> {w, l, d, n_opponents}
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

        print()
        print(f"{'DECK PERFORMANCE SUMMARY':^70s}")
        print("-" * 70)
        print(f"{'Deck ID':>8s} {'W':>4s} {'L':>4s} {'D':>4s} {'WinRate':>8s} {'Opps':>5s}")
        print("-" * 70)
        for deck_id, ds in sorted(deck_stats.items(),
                                   key=lambda x: x[1]["w"] / max(x[1]["w"] + x[1]["l"], 1),
                                   reverse=True):
            total_games = ds["w"] + ds["l"] + ds["d"]
            wr = ds["w"] / max(ds["w"] + ds["l"], 1) * 100
            print(f"  {deck_id:>6d} {ds['w']:>4d} {ds['l']:>4d} {ds['d']:>4d} "
                  f"{wr:>7.1f}% {ds['n_opponents']:>5d}")
        print("-" * 70)

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
