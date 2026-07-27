"""Pokemon TCG MLX — Complete Dashboard (Streamlit).

8 tabs: Overview, Cards, Decks, Deck Builder, Agents, Arena, Replays, Config.
Reads all data from SQLite (model/results.db). No sidebar.

Usage:
    uv run tcg-dashboard
"""
import json
import os
import re
import sys
from pathlib import Path

# Event type mapping
EVENT_TYPES = {
    0: "TURN_START", 1: "MULLIGAN", 2: "DRAW", 3: "END_TURN",
    6: "MOVE_CARD", 10: "PLAY", 11: "ATTACH", 12: "EVOLVE",
    15: "ATTACK", 16: "DAMAGE"
}

# Select type mapping
SELECT_TYPES = {
    0: "NONE", 1: "CHOOSE_ACTIVE", 2: "CHOOSE_BENCH",
    3: "CHOOSE_HAND", 4: "CHOOSE_DISCARD", 5: "CHOOSE_DECK",
    6: "CHOOSE_PRIZE", 7: "CHOOSE_ATTACK", 8: "CHOOSE_ENERGY",
    9: "CHOOSE_TARGET", 10: "SUBMIT"
}


def main():
    """Launch the Streamlit dashboard (entrypoint)."""
    app_path = str(Path(__file__).resolve())
    os.execvp(sys.executable, [sys.executable, "-m", "streamlit", "run", app_path,
        "--server.headless", "true", "--theme.base", "dark",
        "--theme.primaryColor", "#d4a574", "--theme.backgroundColor", "#171717",
        "--theme.secondaryBackgroundColor", "#232329", "--theme.textColor", "#e8e6e3"])


def run_app():
    """Run the Streamlit dashboard UI (only called by streamlit runtime)."""
    import streamlit as st
    try:
        import pandas as pd
    except ImportError:
        st.error("pandas is required. Run: uv add pandas")
        st.stop()

    st.set_page_config(page_title="Pokemon TCG MLX Dashboard", layout="wide",
                       initial_sidebar_state="collapsed")

    # ── paths ────────────────────────────────────────────────────
    ROOT = Path(__file__).resolve().parent.parent
    SMOKE_CONFIG = ROOT / "configs" / "smoke.json"
    TRAIN_CONFIG = ROOT / "configs" / "train_config.json"
    SCHEMA_FILE = ROOT / "configs" / "train_config.schema.json"

    # ── data loading helpers ─────────────────────────────────────
    @st.cache_data(ttl=5)
    def load_runs():
        from rl.results_db import ResultsDB
        db = ResultsDB()
        runs = db.get_all_runs()
        db.close()
        return runs

    @st.cache_data(ttl=5)
    def load_elos():
        from rl.results_db import ResultsDB
        db = ResultsDB()
        elos = db.compute_elos()
        db.close()
        return elos

    @st.cache_data(ttl=5)
    def load_top_cards(n=50, source="remote"):
        from rl.results_db import ResultsDB
        db = ResultsDB()
        cards = db.get_top_cards(n, source)
        db.close()
        return cards

    @st.cache_data(ttl=5)
    def load_dual_elos(n=50):
        """Load cards with both local and remote Elo side by side."""
        from rl.results_db import ResultsDB
        db = ResultsDB()
        remote = db.get_top_cards(n, "remote")
        local = db.get_top_cards(n, "local")
        db.close()
        # Index by card id
        remote_map = {c["id"]: c for c in remote}
        local_map = {c["id"]: c for c in local}
        # Merge: start from remote, fill local
        all_ids = sorted(set(list(remote_map.keys()) + list(local_map.keys())),
                         key=lambda cid: remote_map.get(cid, {}).get("elo", 0), reverse=True)
        result = []
        for cid in all_ids:
            r = remote_map.get(cid, {})
            l = local_map.get(cid, {})
            result.append({
                "id": cid,
                "name": r.get("name") or l.get("name", "?"),
                "category": r.get("category") or l.get("category", ""),
                "energy_type": r.get("energy_type") or l.get("energy_type", ""),
                "elo_remote": r.get("elo", 0) or 0,
                "elo_local": l.get("elo", 0) or 0,
                "games_remote": r.get("games_played", 0) or 0,
                "games_local": l.get("games_played", 0) or 0,
                "win_rate_remote": r.get("win_rate", 0) or 0,
                "win_rate_local": l.get("win_rate", 0) or 0,
            })
        return result

    @st.cache_data(ttl=5)
    def load_top_decks(n=20, source="remote"):
        from rl.results_db import ResultsDB
        db = ResultsDB()
        decks = db.get_top_decks(n, source)
        db.close()
        return decks

    @st.cache_data(ttl=5)
    def load_all_cards():
        from rl.results_db import ResultsDB
        db = ResultsDB()
        rows = db.conn.execute("SELECT id, name, category, energy_type, hp FROM cards ORDER BY name").fetchall()
        db.close()
        return [{"id": r[0], "name": r[1], "category": r[2], "energy_type": r[3], "hp": r[4]} for r in rows]

    @st.cache_data(ttl=5)
    def load_all_decks():
        from rl.results_db import ResultsDB
        db = ResultsDB()
        rows = db.conn.execute("SELECT id, name, source, archetype, card_count FROM decks ORDER BY name").fetchall()
        db.close()
        return [{"id": r[0], "name": r[1], "source": r[2], "archetype": r[3], "card_count": r[4]} for r in rows]

    @st.cache_data(ttl=5)
    def load_deck_cards(deck_id):
        from rl.results_db import ResultsDB
        db = ResultsDB()
        rows = db.conn.execute(
            """SELECT c.id, c.name, c.category, c.energy_type, dc.quantity
               FROM deck_cards dc JOIN cards c ON dc.card_id = c.id
               WHERE dc.deck_id = ? ORDER BY c.category, c.name""",
            (deck_id,)).fetchall()
        db.close()
        return [{"id": r[0], "name": r[1], "category": r[2], "energy_type": r[3], "qty": r[4]} for r in rows]

    @st.cache_data(ttl=5)
    def load_card_usage(card_id):
        from rl.results_db import ResultsDB
        db = ResultsDB()
        elo_rows = db.conn.execute(
            "SELECT elo, games_played, wins, losses, win_rate, source FROM card_elo WHERE card_id = ?",
            (card_id,)).fetchall()
        decks = db.conn.execute(
            """SELECT DISTINCT d.name FROM deck_cards dc JOIN decks d ON dc.deck_id = d.id
               WHERE dc.card_id = ? LIMIT 20""",
            (card_id,)).fetchall()
        db.close()
        elos = [dict(r) for r in elo_rows]
        best = max(elos, key=lambda e: e["elo"]) if elos else None
        return best, [r[0] for r in decks]

    @st.cache_data(ttl=5)
    def load_matches():
        from rl.results_db import ResultsDB
        db = ResultsDB()
        rows = db.conn.execute(
            """SELECT m.id, m.game_index, m.our_agent, m.opp_agent, m.our_side, m.result,
                      m.n_steps, m.created_at, d.name as deck_name
               FROM matches m LEFT JOIN decks d ON m.our_deck_id = d.id
               ORDER BY m.created_at DESC LIMIT 200"""
        ).fetchall()
        db.close()
        return [dict(r) for r in rows]

    @st.cache_data(ttl=5)
    def load_match_detail(match_id):
        from rl.results_db import ResultsDB
        db = ResultsDB()
        match = db.conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)).fetchone()
        steps = db.conn.execute(
            "SELECT * FROM match_steps WHERE match_id = ? ORDER BY step_num, player_idx",
            (match_id,)).fetchall()
        db.close()
        return dict(match) if match else None, [dict(s) for s in steps]

    @st.cache_data(ttl=5)
    def load_step_detail(step_id):
        from rl.results_db import ResultsDB
        db = ResultsDB()
        options = db.conn.execute(
            "SELECT * FROM step_options WHERE step_id = ? ORDER BY option_idx",
            (step_id,)).fetchall()
        events = db.conn.execute(
            "SELECT * FROM step_events WHERE step_id = ?", (step_id,)).fetchall()
        snapshot = db.conn.execute(
            "SELECT * FROM board_snapshots WHERE step_id = ? LIMIT 1", (step_id,)).fetchone()
        pokemon = []
        if snapshot:
            snap_id = snapshot["id"]
            pokemon = db.conn.execute(
                "SELECT * FROM pokemon_on_field WHERE snapshot_id = ? ORDER BY slot, slot_idx",
                (snap_id,)).fetchall()
        db.close()
        return ([dict(o) for o in options], [dict(e) for e in events],
                dict(snapshot) if snapshot else None, [dict(p) for p in pokemon])

    @st.cache_data(ttl=5)
    def load_agent_stats():
        from rl.results_db import ResultsDB
        db = ResultsDB()
        rows = db.conn.execute(
            """SELECT m.our_agent, m.opp_agent, d.name as deck_name,
                      COUNT(*) as games,
                      SUM(CASE WHEN m.result = 1 THEN 1 ELSE 0 END) as wins,
                      SUM(CASE WHEN m.result = -1 THEN 1 ELSE 0 END) as losses,
                      SUM(CASE WHEN m.result = 0 THEN 1 ELSE 0 END) as draws
               FROM matches m LEFT JOIN decks d ON m.our_deck_id = d.id
               GROUP BY m.our_agent, m.opp_agent, d.name"""
        ).fetchall()
        db.close()
        return [dict(r) for r in rows]

    @st.cache_data(ttl=5)
    def load_matchup_matrix():
        from rl.results_db import ResultsDB
        db = ResultsDB()
        rows = db.conn.execute(
            """SELECT m.our_agent, m.opp_agent,
                      COUNT(*) as games,
                      SUM(CASE WHEN m.result = 1 THEN 1 ELSE 0 END) as wins,
                      SUM(CASE WHEN m.result = -1 THEN 1 ELSE 0 END) as losses
               FROM matches m
               GROUP BY m.our_agent, m.opp_agent"""
        ).fetchall()
        db.close()
        return [dict(r) for r in rows]

    @st.cache_data(ttl=30)
    def load_config(path):
        p = Path(path)
        if not p.exists():
            return {}
        return json.loads(p.read_text())

    @st.cache_data(ttl=30)
    def load_schema():
        if not SCHEMA_FILE.exists():
            return {}
        return json.loads(SCHEMA_FILE.read_text())

    @st.cache_data(ttl=30)
    def load_builder_cards():
        """Load all cards from SQLite for the deck builder with dual Elo."""
        from rl.results_db import ResultsDB
        db = ResultsDB()
        rows = db.conn.execute(
            "SELECT id, name, category, stage, hp, energy_type FROM cards ORDER BY name"
        ).fetchall()
        elo_remote_rows = db.conn.execute(
            "SELECT card_id, elo, games_played, win_rate FROM card_elo WHERE source = 'remote'"
        ).fetchall()
        elo_local_rows = db.conn.execute(
            "SELECT card_id, elo, games_played, win_rate FROM card_elo WHERE source = 'local'"
        ).fetchall()
        db.close()
        elo_remote = {r[0]: {"elo": r[1], "games": r[2], "wr": r[3]} for r in elo_remote_rows}
        elo_local = {r[0]: {"elo": r[1], "games": r[2], "wr": r[3]} for r in elo_local_rows}
        cards = []
        for r in rows:
            er = elo_remote.get(r[0], {})
            el = elo_local.get(r[0], {})
            cards.append({
                "id": r[0], "name": r[1], "category": r[2],
                "stage": r[3], "hp": r[4], "energy_type": r[5],
                "elo_remote": er.get("elo", 0) or 0,
                "elo_local": el.get("elo", 0) or 0,
                "elo": max(er.get("elo", 0) or 0, el.get("elo", 0) or 0),
                "games_remote": er.get("games", 0) or 0,
                "games_local": el.get("games", 0) or 0,
            })
        return cards

    def extract_lb_score(label):
        m = re.search(r"lb(\d+)", label)
        return int(m.group(1)) if m else None

    @st.cache_data(ttl=60)
    def load_base_deck():
        """Load the current agent/deck.csv as a dict {card_id: quantity}."""
        deck_path = ROOT / "agent" / "deck.csv"
        if not deck_path.exists():
            return {}
        deck = {}
        for line in deck_path.read_text().strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                cid = int(line)
                deck[cid] = deck.get(cid, 0) + 1
            except ValueError:
                continue
        return deck

    @st.cache_data(ttl=60)
    def load_agent_deck():
        """Load agent/deck.csv as a sorted card-id list, for identity comparison."""
        deck_path = ROOT / "agent" / "deck.csv"
        if not deck_path.exists():
            return []
        cards = []
        for line in deck_path.read_text().splitlines():
            line = line.strip().rstrip(",")
            if line:
                try:
                    cards.append(int(line))
                except ValueError:
                    continue
        return sorted(cards)

    @st.cache_data(ttl=60)
    def load_arena_deck_ranking():
        """Rank decks by local arena Elo, with their compositions.

        Local evidence only: remote games say nothing about how a deck performs
        against the agents we actually run here. Every deck that has played at
        least one local game ranks — they all start at 600, so a short record
        simply sits near its starting point instead of being hidden.
        """
        from rl.results_db import ResultsDB
        with ResultsDB() as db:
            rows = db.conn.execute(
                "SELECT de.deck_id, d.name, de.elo, de.games_played, de.wins, "
                "       de.losses, de.win_rate "
                "FROM deck_elo de JOIN decks d ON de.deck_id = d.id "
                "WHERE de.source = 'local' AND de.games_played > 0 "
                "ORDER BY de.elo DESC"
            ).fetchall()
            ranking = []
            for r in rows:
                cards = []
                for cid, qty in db.conn.execute(
                    "SELECT card_id, quantity FROM deck_cards WHERE deck_id = ?",
                    (r["deck_id"],)
                ):
                    cards.extend([cid] * qty)
                if len(cards) != 60:
                    continue  # incomplete composition: cannot be shipped
                ranking.append({
                    "id": r["deck_id"], "name": r["name"], "elo": r["elo"],
                    "games_played": r["games_played"], "wins": r["wins"],
                    "losses": r["losses"], "win_rate": r["win_rate"] * 100,
                    "cards": sorted(cards),
                })
            return ranking

    def compute_deck_strength(builder_deck, all_cards):
        """Compute average local and remote Elo for cards in the deck."""
        if not builder_deck:
            return 0.0, 0.0
        card_map = {c["id"]: c for c in all_cards}
        total_qty = sum(builder_deck.values())
        if total_qty == 0:
            return 0.0, 0.0
        weighted_local = 0.0
        weighted_remote = 0.0
        for cid, qty in builder_deck.items():
            c = card_map.get(cid)
            if c:
                weighted_local += c["elo_local"] * qty
                weighted_remote += c["elo_remote"] * qty
        return weighted_local / total_qty, weighted_remote / total_qty

    # ── main tabs ────────────────────────────────────────────────
    tab_overview, tab_cards, tab_decks, tab_deck_builder, tab_agents, tab_arena, tab_replays, tab_config = st.tabs(
        ["Overview", "Cards", "Decks", "Deck Builder", "Agents", "Arena", "Replays", "Config"])

    # ════════════════════════════════════════════════════════════════
    # TAB 1: OVERVIEW
    # ════════════════════════════════════════════════════════════════
    with tab_overview:
        runs = load_runs()
        elos = load_elos()

        if not runs:
            st.warning("No tournament results yet. Run `uv run tcg-tournament` first.")
            st.stop()

        latest = runs[-1]
        best = max(runs, key=lambda r: r["win_rate"])
        best_elo = max(elos.values()) if elos else 0
        best_elo_label = max(elos, key=elos.get) if elos else "N/A"

        # Latest model stats
        st.subheader("Latest Model Stats")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Elo", f"{elos.get(latest['timestamp'], 0):.0f}")
        with c2:
            st.metric("Win Rate", f"{latest['win_rate']:.1f}%")
        with c3:
            st.metric("Games", f"{latest['total_w'] + latest['total_l'] + latest['total_d']}")
        with c4:
            st.metric("Note", latest["note"] or "—")

        st.divider()

        # Best model ever
        st.subheader("Best Model Ever")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Best Elo", f"{best_elo:.0f}", help=f"Label: {best_elo_label}")
        with c2:
            st.metric("Best Win Rate", f"{best['win_rate']:.1f}%")
        with c3:
            st.metric("Best Run", best["timestamp"])

        st.divider()

        # Elo over time
        st.subheader("Elo Over Time")
        if len(runs) > 1:
            elo_df = pd.DataFrame({
                "Date": pd.to_datetime([r["timestamp"] for r in runs]),
                "Elo": [elos.get(r["timestamp"], 0) for r in runs]
            })
            st.line_chart(elo_df, x="Date", y="Elo", color="#d4a574")
        else:
            st.info("Need at least 2 runs to show Elo history.")

        st.divider()

        # Latest matchup grid
        st.subheader(f"Latest Matchups — {latest['timestamp']}")
        if latest["matchups"]:
            cols = st.columns(min(4, len(latest["matchups"])))
            for i, m in enumerate(latest["matchups"]):
                with cols[i % len(cols)]:
                    wr = m["wr"]
                    st.metric(
                        label=m["opponent"],
                        value=f"{wr:.0f}%",
                        delta=f"W{m['w']} L{m['l']} D{m['d']}",
                        delta_color="normal" if wr >= 50 else "inverse"
                    )
        else:
            st.info("No matchups recorded for this run.")

    # ════════════════════════════════════════════════════════════════
    # TAB 2: CARDS
    # ════════════════════════════════════════════════════════════════
    with tab_cards:
        st.subheader("Cards by Elo (Dual Source)")

        # Filters
        c1, c2 = st.columns(2)
        with c1:
            category_filter = st.selectbox("Category", ["All", "Pokemon", "Trainer", "Energy"], index=0, key="cards_cat")
        with c2:
            energy_types_cards = ["All", "Fire", "Water", "Grass", "Lightning", "Psychic", "Fighting",
                           "Darkness", "Metal", "Fairy", "Dragon", "Colorless"]
            energy_filter = st.selectbox("Energy Type", energy_types_cards, index=0, key="cards_energy")

        # Load dual Elo data
        all_cards = load_dual_elos(200)
        if not all_cards:
            st.info("No card Elo data. Run matches first.")
        else:
            df = pd.DataFrame(all_cards)
            if category_filter != "All":
                df = df[df["category"] == category_filter]
            if energy_filter != "All":
                df = df[df["energy_type"] == energy_filter]

            if df.empty:
                st.info("No cards match the filters.")
            else:
                st.dataframe(
                    df[["name", "category", "energy_type",
                        "elo_remote", "elo_local",
                        "games_remote", "games_local",
                        "win_rate_remote", "win_rate_local"]].rename(columns={
                        "name": "Name", "category": "Category", "energy_type": "Type",
                        "elo_remote": "Elo (remote)", "elo_local": "Elo (local)",
                        "games_remote": "Games (remote)", "games_local": "Games (local)",
                        "win_rate_remote": "Win Rate (remote)", "win_rate_local": "Win Rate (local)"
                    }),
                    use_container_width=True, hide_index=True
                )

                st.divider()

                # Card detail
                card_names = df["name"].tolist()
                selected_card = st.selectbox("Select a card for details", card_names, key="cards_detail_select")

                if selected_card:
                    card_row = df[df["name"] == selected_card].iloc[0]
                    card_id = int(card_row["id"])

                    st.subheader(f"Card: {selected_card}")
                    c1, c2, c3, c4, c5, c6 = st.columns(6)
                    with c1:
                        st.metric("Elo (remote)", f"{card_row['elo_remote']:.1f}")
                    with c2:
                        st.metric("Elo (local)", f"{card_row['elo_local']:.1f}")
                    with c3:
                        st.metric("Games (remote)", f"{card_row['games_remote']}")
                    with c4:
                        st.metric("Games (local)", f"{card_row['games_local']}")
                    with c5:
                        wr = card_row['win_rate_remote'] or 0
                        st.metric("Win Rate (remote)", f"{wr:.1%}" if wr else "—")
                    with c6:
                        st.metric("Category", card_row['category'] or "—")

                    elo_data, decks = load_card_usage(card_id)
                    if decks:
                        st.markdown(f"**Used in decks:** {', '.join(decks)}")
                    else:
                        st.info("No deck usage data.")

    # ════════════════════════════════════════════════════════════════
    # TAB 3: DECKS
    # ════════════════════════════════════════════════════════════════
    with tab_decks:
        st.subheader("Decks by Elo")

        # Filters
        c1, c2 = st.columns(2)
        with c1:
            all_decks = load_all_decks()
            sources = ["All"] + sorted(set(d["source"] for d in all_decks if d["source"]))
            source_filter = st.selectbox("Source", sources, index=0)
        with c2:
            source_elo = st.selectbox("Elo Source", ["remote", "local", "replay"], index=0, key="deck_elo_source")

        # Load decks
        top_decks = load_top_decks(100, source_elo)
        if not top_decks:
            st.info("No deck Elo data. Run matches first.")
        else:
            df = pd.DataFrame(top_decks)
            if source_filter != "All":
                # Merge with source info from all_decks
                deck_sources = {d["name"]: d["source"] for d in all_decks}
                df["source"] = df["name"].map(deck_sources)
                df = df[df["source"] == source_filter]

            if df.empty:
                st.info("No decks match the filter.")
            else:
                st.dataframe(
                    df[["name", "source", "archetype", "elo", "games_played", "win_rate"]].rename(columns={
                        "name": "Name", "source": "Source", "archetype": "Archetype",
                        "elo": "Elo", "games_played": "Games", "win_rate": "Win Rate"
                    }),
                    use_container_width=True, hide_index=True
                )

                st.divider()

                # Deck detail
                deck_names = df["name"].tolist()
                selected_deck = st.selectbox("Select a deck for details", deck_names, key="deck_select")

                if selected_deck:
                    deck_row = df[df["name"] == selected_deck].iloc[0]
                    deck_id = int(deck_row["id"])

                    st.subheader(f"Deck: {selected_deck}")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Elo", f"{deck_row['elo']:.1f}")
                    with c2:
                        st.metric("Games", f"{deck_row['games_played']}")
                    with c3:
                        st.metric("Archetype", deck_row["archetype"] or "—")

                    # Deck composition
                    deck_cards = load_deck_cards(deck_id)
                    if deck_cards:
                        cards_df = pd.DataFrame(deck_cards)
                        st.markdown("**Deck Composition:**")
                        st.dataframe(
                            cards_df[["name", "category", "energy_type", "qty"]].rename(columns={
                                "name": "Card", "category": "Category",
                                "energy_type": "Type", "qty": "Qty"
                            }),
                            use_container_width=True, hide_index=True
                        )

                        # Load in Builder button
                        if st.button("Load in Builder", key=f"load_in_builder_{deck_id}", type="primary"):
                            st.session_state.builder_deck = {c["id"]: c["qty"] for c in deck_cards}
                            st.session_state.builder_deck_name = selected_deck
                            st.success(f"Deck '{selected_deck}' loaded — go to Deck Builder tab")
                    else:
                        st.info("No card composition data for this deck.")

    # ════════════════════════════════════════════════════════════════
    # TAB 4: DECK BUILDER
    # ════════════════════════════════════════════════════════════════
    with tab_deck_builder:
        # Initialize session state for deck builder
        if "builder_deck" not in st.session_state:
            st.session_state.builder_deck = {}
        if "builder_deck_name" not in st.session_state:
            st.session_state.builder_deck_name = ""
        if "builder_show_compare" not in st.session_state:
            st.session_state.builder_show_compare = False

        deck = st.session_state.builder_deck

        # Load card data
        all_builder_cards = load_builder_cards()
        all_stages = sorted(set(c["stage"] for c in all_builder_cards if c["stage"]))
        energy_types = ["All", "Fire", "Water", "Grass", "Lightning", "Psychic",
                        "Fighting", "Darkness", "Metal", "Fairy", "Dragon", "Colorless"]

        # Compute deck strength
        avg_local, avg_remote = compute_deck_strength(deck, all_builder_cards)

        # ── Left panel: deck management (sidebar) ─────────────────
        with st.sidebar:
            st.header("Deck Builder")

            # Save controls
            st.subheader("Save Deck")
            save_col1, save_col2 = st.columns([2, 1])
            with save_col1:
                deck_name_input = st.text_input(
                    "Deck name", value=st.session_state.builder_deck_name,
                    key="builder_name_input", label_visibility="collapsed",
                    placeholder="Enter deck name...")
            with save_col2:
                if st.button("Save", key="builder_save_btn", type="primary"):
                    total = sum(deck.values())
                    if total != 60:
                        st.error(f"Deck must have exactly 60 cards (has {total})")
                    elif not deck_name_input.strip():
                        st.error("Enter a deck name")
                    else:
                        try:
                            from rl.results_db import ResultsDB
                            db = ResultsDB()
                            deck_id = db.add_deck(
                                deck_name_input.strip(), "builder", card_count=60)
                            card_qtys = [(cid, qty) for cid, qty in deck.items()]
                            db.add_deck_cards(deck_id, card_qtys)
                            db.close()
                            st.session_state.builder_deck_name = deck_name_input.strip()
                            st.success(f"Saved '{deck_name_input.strip()}'")
                            st.cache_data.clear()
                        except Exception as e:
                            st.error(f"Error saving: {e}")

            # Load controls
            st.subheader("Load Deck")
            all_decks_data = load_all_decks()
            deck_names = ["— select —"] + [d["name"] for d in all_decks_data]
            load_choice = st.selectbox("Existing decks", deck_names,
                                       key="builder_load_select", label_visibility="collapsed")
            if load_choice != "— select —" and st.button("Load selected deck",
                                                          key="builder_load_btn"):
                target = next(d for d in all_decks_data if d["name"] == load_choice)
                cards_data = load_deck_cards(target["id"])
                deck.clear()
                for c in cards_data:
                    deck[c["id"]] = c["qty"]
                st.session_state.builder_deck_name = load_choice
                st.success(f"Loaded '{load_choice}' ({sum(deck.values())} cards)")
                st.rerun()

            st.divider()

            # Deck composition
            total_cards = sum(deck.values())
            st.subheader(f"Deck ({total_cards}/60)")

            # Validation messages
            if total_cards > 0:
                over4 = {cid: q for cid, q in deck.items() if q > 4}
                if over4:
                    names = ", ".join(
                        f"{next((c['name'] for c in all_builder_cards if c['id'] == cid), str(cid))} x{q}"
                        for cid, q in over4.items())
                    st.warning(f"Max 4 copies: {names}")

            if not deck:
                st.caption("Click a card to add it")
            else:
                # Sort by name
                sorted_deck = sorted(deck.items(),
                                     key=lambda x: next((c["name"] for c in all_builder_cards if c["id"] == x[0]), ""))
                for cid, qty in sorted_deck:
                    card_info = next((c for c in all_builder_cards if c["id"] == cid), None)
                    cname = card_info["name"] if card_info else str(cid)
                    cstage = card_info["stage"] if card_info else ""
                    c1, c2, c3 = st.columns([4, 2, 1])
                    with c1:
                        st.caption(f"**{cname}**")
                    with c2:
                        st.caption(f"{qty}x {cstage[:12]}")
                    with c3:
                        if st.button("x", key=f"builder_rm_{cid}"):
                            deck[cid] -= 1
                            if deck[cid] <= 0:
                                del deck[cid]
                            st.rerun()

            st.divider()

            # Deck Strength Estimation
            if total_cards > 0:
                st.subheader("Deck Strength")
                s1, s2 = st.columns(2)
                with s1:
                    st.metric("Avg Elo (local)", f"{avg_local:.0f}")
                with s2:
                    st.metric("Avg Elo (remote)", f"{avg_remote:.0f}")

            st.divider()

            # Action buttons
            act1, act2, act3 = st.columns(3)
            with act1:
                if st.button("Clear", key="builder_clear_btn"):
                    deck.clear()
                    st.session_state.builder_deck_name = ""
                    st.rerun()
            with act2:
                if total_cards == 60:
                    csv_lines = []
                    for cid, qty in deck.items():
                        for _ in range(qty):
                            csv_lines.append(str(cid))
                    csv_text = "\n".join(csv_lines) + "\n"
                    st.download_button("Export CSV", csv_text, "deck.csv", "text/csv",
                                       key="builder_export_btn")
                else:
                    st.button("Export CSV", key="builder_export_disabled",
                              disabled=True)
            with act3:
                st.toggle("Deck only", key="builder_deck_only",
                          help="Show only cards in deck")

            # Compare with base deck
            if total_cards > 0:
                st.divider()
                if st.button("Compare with base deck", key="builder_compare_btn"):
                    st.session_state.builder_show_compare = True

                if st.session_state.get("builder_show_compare", False):
                    base = load_base_deck()
                    if not base:
                        st.info("No agent/deck.csv found.")
                    else:
                        card_map = {c["id"]: c for c in all_builder_cards}
                        all_ids = sorted(set(list(deck.keys()) + list(base.keys())),
                                         key=lambda cid: card_map.get(cid, {}).get("name", ""))
                        rows = []
                        for cid in all_ids:
                            name = card_map.get(cid, {}).get("name", str(cid))
                            builder_qty = deck.get(cid, 0)
                            base_qty = base.get(cid, 0)
                            diff = builder_qty - base_qty
                            elo_l = card_map.get(cid, {}).get("elo_local", 0)
                            elo_r = card_map.get(cid, {}).get("elo_remote", 0)
                            changed = builder_qty != base_qty
                            rows.append({
                                "Card": name,
                                "Builder": builder_qty,
                                "Base": base_qty,
                                "Diff": diff,
                                "Elo (local)": elo_l,
                                "Elo (remote)": elo_r,
                                "_changed": changed,
                            })
                        compare_df = pd.DataFrame(rows)
                        st.markdown("**Builder vs Base Deck (agent/deck.csv):**")
                        # Highlight changed rows
                        def highlight_diff(row):
                            if row.get("_changed", False):
                                return ["background-color: #3d2b1f"] * len(row)
                            return [""] * len(row)
                        st.dataframe(
                            compare_df[["Card", "Builder", "Base", "Diff", "Elo (local)", "Elo (remote)"]],
                            use_container_width=True, hide_index=True,
                            row_config={"Diff": st.column_config.NumberColumn(
                                "Diff", help="Positive = in builder but not base, negative = vice versa")}
                        )
                        st.info("Yellow-highlighted rows are cards that differ between builder and base deck.")
                    if st.button("Close comparison", key="builder_close_compare"):
                        st.session_state.builder_show_compare = False
                        st.rerun()

            # Use this deck button
            if total_cards == 60:
                st.divider()
                st.caption("Overwrites agent/deck.csv — the deck the submission ships. "
                           "The previous one is kept as deck.csv.bak.")
                if st.button("Use this deck in tournament", key="builder_use_deck", type="primary"):
                    csv_lines = []
                    for cid, qty in deck.items():
                        for _ in range(qty):
                            csv_lines.append(str(cid))
                    csv_text = "\n".join(csv_lines) + "\n"
                    deck_path = ROOT / "agent" / "deck.csv"
                    # Back up first: this button silently replaced a tournament-
                    # validated deck once, with no way to tell afterwards.
                    if deck_path.exists():
                        backup = deck_path.with_suffix(".csv.bak")
                        backup.write_text(deck_path.read_text())
                        st.info(f"Previous deck backed up to {backup.name}")
                    deck_path.write_text(csv_text)
                    st.success(f"Deck saved to {deck_path} ({total_cards} cards)")

        # ── Right panel: card browser ─────────────────────────────
        st.subheader("Card Browser")

        # Filters
        f1, f2, f3 = st.columns(3)
        with f1:
            search_query = st.text_input("Search cards",
                                         key="builder_search",
                                         placeholder="Name or ID...")
        with f2:
            category_filter = st.selectbox("Category",
                ["All", "Pokemon", "Trainer", "Energy"], key="builder_cat")
        with f3:
            energy_filter = st.selectbox("Energy Type", energy_types,
                                         key="builder_energy")

        # Filter cards
        filtered = all_builder_cards
        if search_query:
            q = search_query.lower()
            filtered = [c for c in filtered
                       if q in c["name"].lower() or q in str(c["id"])]
        if category_filter != "All":
            filtered = [c for c in filtered if c["category"] == category_filter]
        if energy_filter != "All":
            filtered = [c for c in filtered if c["energy_type"] == energy_filter]
        if st.session_state.get("builder_deck_only", False):
            filtered = [c for c in filtered if deck.get(c["id"], 0) > 0]

        st.caption(f"Showing {len(filtered)} of {len(all_builder_cards)} cards")

        if not filtered:
            st.info("No cards match the current filters.")
        else:
            CARDS_PER_ROW = 4
            for row_start in range(0, len(filtered), CARDS_PER_ROW):
                row_cards = filtered[row_start:row_start + CARDS_PER_ROW]
                cols = st.columns(CARDS_PER_ROW)
                for i, card in enumerate(row_cards):
                    with cols[i]:
                        card_id = card["id"]
                        img_path = ROOT / "scripts" / "deck_builder" / "card_images" / f"{card_id}.jpg"
                        if img_path.exists():
                            st.image(str(img_path), use_container_width=True)
                        else:
                            st.caption(f"[No image {card_id}]")

                        st.markdown(f"**{card['name']}**")
                        meta_parts = [str(card_id)]
                        if card["category"]:
                            meta_parts.append(card["category"])
                        if card["hp"]:
                            meta_parts.append(f"{card['hp']}HP")
                        if card["energy_type"]:
                            meta_parts.append(card["energy_type"])
                        if card["elo_remote"] and card["elo_remote"] > 0:
                            meta_parts.append(f"R:{card['elo_remote']:.0f}")
                        if card["elo_local"] and card["elo_local"] > 0:
                            meta_parts.append(f"L:{card['elo_local']:.0f}")
                        st.caption(" | ".join(meta_parts))

                        current_qty = deck.get(card_id, 0)
                        btn_label = f"Add ({current_qty}/4)" if current_qty > 0 else "Add"
                        if current_qty >= 4:
                            st.button(btn_label, key=f"builder_add_{card_id}",
                                      disabled=True, use_container_width=True)
                        elif sum(deck.values()) >= 60:
                            st.button("Deck full", key=f"builder_full_{card_id}",
                                      disabled=True, use_container_width=True)
                        else:
                            if st.button(btn_label, key=f"builder_add_{card_id}",
                                         use_container_width=True):
                                deck[card_id] = deck.get(card_id, 0) + 1
                                st.rerun()

    # ════════════════════════════════════════════════════════════════
    # TAB 5: AGENTS
    # ════════════════════════════════════════════════════════════════
    with tab_agents:
        st.subheader("Agent Performance")

        agent_stats = load_agent_stats()
        if not agent_stats:
            st.info("No match data available.")
        else:
            df = pd.DataFrame(agent_stats)
            df["win_rate"] = df["wins"] / df["games"].replace(0, float("nan"))

            # Group by our_agent
            agent_summary = df.groupby("our_agent").agg({
                "games": "sum",
                "wins": "sum",
                "losses": "sum",
                "draws": "sum"
            }).reset_index()
            agent_summary["win_rate"] = agent_summary["wins"] / agent_summary["games"].replace(0, float("nan"))

            st.markdown("### Our Agents")
            st.dataframe(
                agent_summary.rename(columns={
                    "our_agent": "Agent", "games": "Games", "wins": "Wins",
                    "losses": "Losses", "draws": "Draws", "win_rate": "Win Rate"
                }).sort_values("Win Rate", ascending=False),
                use_container_width=True, hide_index=True
            )

            st.divider()

            # Agent vs opponent breakdown
            st.markdown("### Agent vs Opponent")
            agent_filter = st.selectbox("Select Agent", df["our_agent"].unique(), key="agent_filter")
            if agent_filter:
                agent_df = df[df["our_agent"] == agent_filter].copy()
                agent_df["win_rate"] = agent_df["wins"] / agent_df["games"].replace(0, float("nan"))

                st.dataframe(
                    agent_df[["opp_agent", "deck_name", "games", "wins", "losses", "draws", "win_rate"]].rename(columns={
                        "opp_agent": "Opponent", "deck_name": "Deck", "games": "Games",
                        "wins": "Wins", "losses": "Losses", "draws": "Draws", "win_rate": "Win Rate"
                    }).sort_values("Games", ascending=False),
                    use_container_width=True, hide_index=True
                )

    # ════════════════════════════════════════════════════════════════
    # TAB 6: ARENA
    # ════════════════════════════════════════════════════════════════
    with tab_arena:
        st.subheader("Arena Deck")

        current_deck = load_agent_deck()
        arena_decks = load_arena_deck_ranking()

        if not arena_decks:
            st.info("No local arena evidence yet. Run a tournament to rank decks.")
        else:
            current_id = None
            for d in arena_decks:
                if d["cards"] == current_deck:
                    current_id = d["id"]
                    break

            rank_df = pd.DataFrame([{
                "": "▶" if d["id"] == current_id else "",
                "Deck": d["name"],
                "Elo": round(d["elo"], 1),
                "Games": d["games_played"],
                "W": d["wins"],
                "L": d["losses"],
                "Win Rate": d["win_rate"],
            } for d in arena_decks])
            st.dataframe(rank_df, use_container_width=True, hide_index=True,
                         column_config={"Win Rate": st.column_config.NumberColumn(format="%.1f%%")})
            st.caption("Local arena evidence only. ▶ marks the deck in agent/deck.csv — "
                       "the one the submission ships.")

            best = arena_decks[0]
            if current_id == best["id"]:
                st.success(f"agent/deck.csv already holds the top deck: **{best['name']}** "
                           f"(Elo {best['elo']:.0f}, {best['wins']}W-{best['losses']}L)")
            else:
                current_name = next((d["name"] for d in arena_decks if d["id"] == current_id),
                                    "not ranked yet")
                st.warning(
                    f"Top arena deck is **{best['name']}** "
                    f"(Elo {best['elo']:.0f}, {best['wins']}W-{best['losses']}L), "
                    f"but agent/deck.csv holds **{current_name}**."
                )
                if st.button(f"Use best arena deck ({best['name']})",
                             key="arena_use_best", type="primary"):
                    deck_path = ROOT / "agent" / "deck.csv"
                    if deck_path.exists():
                        backup = deck_path.with_suffix(".csv.bak")
                        backup.write_text(deck_path.read_text())
                    deck_path.write_text("\n".join(str(c) for c in best["cards"]) + "\n")
                    st.success(f"agent/deck.csv now holds {best['name']} "
                               f"({len(best['cards'])} cards). Previous deck saved as deck.csv.bak.")
                    st.rerun()

        st.divider()
        st.subheader("Arena Matchup Matrix")

        matchup_data = load_matchup_matrix()
        if not matchup_data:
            st.info("No matchup data available.")
        else:
            df = pd.DataFrame(matchup_data)
            df["win_rate"] = df["wins"] / df["games"].replace(0, float("nan"))

            # Create pivot table
            pivot = df.pivot_table(
                index="our_agent", columns="opp_agent",
                values="win_rate", aggfunc="first"
            )

            if not pivot.empty:
                st.markdown("### Win Rate Matrix (Our Agent vs Opponent)")
                st.dataframe(
                    pivot.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1).format("{:.1%}"),
                    use_container_width=True
                )

                st.divider()

                # Games matrix
                pivot_games = df.pivot_table(
                    index="our_agent", columns="opp_agent",
                    values="games", aggfunc="first"
                )
                st.markdown("### Games Played Matrix")
                st.dataframe(pivot_games, use_container_width=True)

            # Self-play vs submissions
            st.divider()
            st.markdown("### Self-Play Analysis")
            self_play = df[df["our_agent"] == df["opp_agent"]]
            if not self_play.empty:
                st.dataframe(
                    self_play[["our_agent", "games", "wins", "losses", "win_rate"]].rename(columns={
                        "our_agent": "Agent", "games": "Games", "wins": "Wins",
                        "losses": "Losses", "win_rate": "Win Rate"
                    }),
                    use_container_width=True, hide_index=True
                )
            else:
                st.info("No self-play matches recorded.")

            # ── Deck Performance Summary ────────────────────────────
            st.divider()
            st.subheader("Deck Performance (from matches)")

            try:
                from rl.results_db import ResultsDB
                db = ResultsDB()
                deck_perf_rows = db.conn.execute(
                    """SELECT d.name as deck_name,
                              m.opp_agent,
                              COUNT(*) as games,
                              SUM(CASE WHEN m.result = 1 THEN 1 ELSE 0 END) as wins,
                              SUM(CASE WHEN m.result = -1 THEN 1 ELSE 0 END) as losses,
                              SUM(CASE WHEN m.result = 0 THEN 1 ELSE 0 END) as draws
                       FROM matches m
                       LEFT JOIN decks d ON m.our_deck_id = d.id
                       WHERE m.our_deck_id IS NOT NULL
                       GROUP BY d.name, m.opp_agent"""
                ).fetchall()
                db.close()

                if deck_perf_rows:
                    deck_df = pd.DataFrame([dict(r) for r in deck_perf_rows])
                    deck_df["win_rate"] = deck_df["wins"] / deck_df["games"].replace(0, float("nan"))

                    # Best deck by overall win rate
                    deck_summary = deck_df.groupby("deck_name").agg({
                        "games": "sum", "wins": "sum", "losses": "sum", "draws": "sum"
                    }).reset_index()
                    deck_summary["win_rate"] = deck_summary["wins"] / deck_summary["games"].replace(0, float("nan"))
                    deck_summary = deck_summary.sort_values("win_rate", ascending=False)

                    best_deck = deck_summary.iloc[0]
                    st.metric("Best Deck", best_deck["deck_name"],
                              f"{best_deck['win_rate']:.1%} win rate ({best_deck['games']} games)")

                    st.markdown("**Per-Deck Breakdown:**")
                    st.dataframe(
                        deck_df[["deck_name", "opp_agent", "games", "wins", "losses", "draws", "win_rate"]].rename(columns={
                            "deck_name": "Deck", "opp_agent": "Opponent", "games": "Games",
                            "wins": "Wins", "losses": "Losses", "draws": "Draws", "win_rate": "Win Rate"
                        }).sort_values(["Deck", "Win Rate"], ascending=[True, False]),
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.info("No deck-specific match data yet.")
            except Exception as e:
                st.info(f"Deck performance unavailable: {e}")

    # ════════════════════════════════════════════════════════════════
    # TAB 7: REPLAYS
    # ════════════════════════════════════════════════════════════════
    with tab_replays:
        st.subheader("Match Replays")

        matches = load_matches()
        if not matches:
            st.info("No matches recorded yet.")
        else:
            # Match list
            matches_df = pd.DataFrame(matches)
            matches_df["result_str"] = matches_df["result"].map({1: "WIN", -1: "LOSS", 0: "DRAW"})
            matches_df["label"] = matches_df.apply(
                lambda r: f"#{r['id']} vs {r['opp_agent']} - {r['result_str']} ({r['n_steps']} steps)",
                axis=1
            )

            selected_label = st.selectbox("Select a match", matches_df["label"].tolist())

            if selected_label:
                match_id = int(matches_df[matches_df["label"] == selected_label].iloc[0]["id"])
                match_data, steps = load_match_detail(match_id)

                if not match_data:
                    st.error("Match not found.")
                else:
                    # Match header
                    result_map = {1: "WIN", -1: "LOSS", 0: "DRAW"}
                    st.markdown(f"### Match #{match_id}: {result_map.get(match_data['result'], '?')}")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Our Agent", match_data["our_agent"])
                    with c2:
                        st.metric("Opponent", match_data["opp_agent"])
                    with c3:
                        st.metric("Steps", f"{match_data['n_steps']}")
                    with c4:
                        st.metric("Date", match_data["created_at"][:10] if match_data["created_at"] else "—")

                    st.divider()

                    if not steps:
                        st.info("No step data for this match.")
                    else:
                        # Step viewer
                        step_nums = sorted(set(s["step_num"] for s in steps))
                        max_step = len(step_nums) - 1 if step_nums else 0

                        step_idx = st.slider("Step", 0, max_step, 0, key="step_slider")
                        current_step_num = step_nums[step_idx] if step_nums else 0

                        st.markdown(f"**Step {current_step_num}** (of {len(step_nums)} unique decision points)")

                        # Get steps for this step_num
                        step_data = [s for s in steps if s["step_num"] == current_step_num]

                        for step in step_data:
                            st.markdown(f"#### Player {step['player_idx']} - Turn {step['turn']}")

                            # Select info
                            select_type_name = SELECT_TYPES.get(step.get("select_type"), "UNKNOWN")
                            st.markdown(f"**Select Type:** {select_type_name} | **Options:** {step['n_options']} | **Status:** {step['status']}")

                            # Load step details
                            options, events, snapshot, pokemon = load_step_detail(step["id"])

                            # Board state
                            if snapshot:
                                st.markdown("**Board State:**")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Deck", snapshot["deck_count"])
                                with col2:
                                    st.metric("Hand", snapshot["hand_count"])
                                with col3:
                                    st.metric("Prizes", snapshot["prize_count"])
                                with col4:
                                    st.metric("Discard", snapshot["discard_count"])

                                # Pokemon on field
                                if pokemon:
                                    poke_df = pd.DataFrame(pokemon)
                                    st.markdown("**Pokemon on Field:**")
                                    st.dataframe(
                                        poke_df[["slot", "card_id", "hp", "max_hp", "n_energies"]].rename(columns={
                                            "slot": "Slot", "card_id": "Card ID", "hp": "HP",
                                            "max_hp": "Max HP", "n_energies": "Energies"
                                        }),
                                        use_container_width=True, hide_index=True
                                    )

                            # Events
                            if events:
                                st.markdown("**Events:**")
                                for evt in events:
                                    evt_type = EVENT_TYPES.get(evt["event_type"], f"TYPE_{evt['event_type']}")
                                    card_info = f"Card {evt['card_id']}" if evt.get("card_id") else ""
                                    value_info = f"Value: {evt['value']}" if evt.get("value") else ""
                                    st.text(f"  [{evt_type}] {card_info} {value_info}")

                            # Options
                            if options:
                                st.markdown("**Available Options:**")
                                opts_df = pd.DataFrame(options)
                                st.dataframe(
                                    opts_df[["option_idx", "option_type", "was_selected"]].rename(columns={
                                        "option_idx": "Index", "option_type": "Type", "was_selected": "Selected"
                                    }),
                                    use_container_width=True, hide_index=True
                                )

                            st.divider()

    # ════════════════════════════════════════════════════════════════
    # TAB 8: CONFIG
    # ════════════════════════════════════════════════════════════════
    with tab_config:
        st.subheader("Configuration")

        smoke = load_config(str(SMOKE_CONFIG))
        train = load_config(str(TRAIN_CONFIG))
        schema = load_schema()
        props = schema.get("properties", {})

        # Side by side configs
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Smoke Config")
            if smoke:
                smoke_rows = []
                for k, v in smoke.items():
                    if k == "$schema":
                        continue
                    desc = props.get(k, {}).get("description", "")
                    smoke_rows.append({"Parameter": k, "Value": v, "Description": desc})
                st.dataframe(pd.DataFrame(smoke_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No smoke config found.")

        with col2:
            st.markdown("### Train Config")
            if train:
                train_rows = []
                for k, v in train.items():
                    if k == "$schema":
                        continue
                    desc = props.get(k, {}).get("description", "")
                    train_rows.append({"Parameter": k, "Value": v, "Description": desc})
                st.dataframe(pd.DataFrame(train_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No train config found.")

        st.divider()

        # Entrypoints
        st.markdown("### Entrypoints")
        entrypoints = [
            ("tcg-data", "Kaggle dataset downloader"),
            ("tcg-build-bc", "BC dataset builder"),
            ("tcg-build-daily", "Single-replay dataset builder"),
            ("tcg-train", "MLX Metal GPU trainer"),
            ("tcg-evaluate", "1v1 evaluation"),
            ("tcg-tournament", "Multi-opponent tournament"),
            ("tcg-submission", "Build submission.tar.gz"),
            ("tcg-submit", "Submit to Kaggle"),
            ("tcg-dashboard", "This dashboard"),
        ]
        st.dataframe(
            pd.DataFrame([{"Command": f"uv run {c}", "Description": d} for c, d in entrypoints]),
            use_container_width=True, hide_index=True
        )


if __name__ == "__main__":
    run_app()
