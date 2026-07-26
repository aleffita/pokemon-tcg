"""Pokémon TCG MLX — Elo Dashboard (Streamlit).

Reads tournament results from SQLite (model/results.db) and displays
Elo rankings, matchup history, and configuration.

Usage:
    uv run tcg-dashboard
"""
import json
import os
import re
import sys
from pathlib import Path


def main():
    """Launch the Streamlit dashboard (entrypoint)."""
    app_path = str(Path(__file__).resolve())
    argv = [sys.executable, "-m", "streamlit", "run", app_path,
            "--server.headless", "true",
            "--theme.base", "dark",
            "--theme.primaryColor", "#d4a574",
            "--theme.backgroundColor", "#171717",
            "--theme.secondaryBackgroundColor", "#232329",
            "--theme.textColor", "#e8e6e3"]
    os.execvp(sys.executable, argv)


def run_app():
    """Run the Streamlit dashboard UI (only called by streamlit runtime)."""
    import streamlit as st
    try:
        import pandas as pd
    except ImportError:
        st.error("pandas is required. Run: uv add pandas")
        st.stop()

    # ── paths ────────────────────────────────────────────────────
    ROOT = Path(__file__).resolve().parent.parent
    SMOKE_CONFIG = ROOT / "configs" / "smoke.json"
    TRAIN_CONFIG = ROOT / "configs" / "train_config.json"
    SCHEMA_FILE = ROOT / "configs" / "train_config.schema.json"

    # ── elo helpers ──────────────────────────────────────────────
    def extract_lb_score(label):
        m = re.search(r"lb(\d+)", label)
        return int(m.group(1)) if m else None

    # ── data loading from SQLite ─────────────────────────────────
    @st.cache_data(ttl=5)
    def load_results():
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

    # ── page config ──────────────────────────────────────────────
    st.set_page_config(page_title="Pokémon TCG MLX — Elo Dashboard", page_icon="🧬",
                       layout="wide", initial_sidebar_state="expanded")

    # ── sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.title("🧬 Pokémon TCG MLX")
        st.caption("Elo Dashboard")
        st.divider()
        runs = load_results() or []
        elos = load_elos() if runs else {}
        if runs:
            latest = runs[-1]
            st.metric("Latest Run", latest["timestamp"])
            if latest["overall"]:
                st.metric("Win Rate", f'{latest["overall"]["wr"]:.1f}%')
                st.metric("Elo", f'{elos.get(latest["timestamp"], 0):.0f}')
        else:
            st.info("No tournament results yet.")
        st.divider()
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.subheader("Files")
        st.code(f"Results: model/results.db", language=None)
        st.code(f"Smoke:   {SMOKE_CONFIG}", language=None)
        st.code(f"Train:   {TRAIN_CONFIG}", language=None)

    # ── tabs ─────────────────────────────────────────────────────
    tab_overview, tab_elo, tab_matchups, tab_history, tab_config = st.tabs(
        ["📊 Overview", "🏆 Elo Rankings", "🎯 Matchups", "📈 History", "⚙️ Config"])

    with tab_overview:
        if not runs:
            st.warning("No tournament results yet. Run `uv run tcg-tournament` first.")
            st.info("Results appear here automatically after each tournament run.")
            st.stop()
        latest = runs[-1]
        best = max(runs, key=lambda r: r["overall"]["wr"] if r["overall"] else 0)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Latest Elo", f'{elos.get(latest["timestamp"], 0):.0f}')
        with c2:
            st.metric("Latest Win Rate", f'{latest["overall"]["wr"]:.1f}%')
        with c3:
            st.metric("Best Elo Ever", f'{max((elos.get(r["timestamp"], 0) for r in runs), default=0):.0f}')
        with c4:
            st.metric("Best Win Rate", f'{best["overall"]["wr"]:.1f}%')
        st.divider()
        st.subheader("Elo Over Time")
        elo_df = pd.DataFrame({"Date": pd.to_datetime([r["timestamp"] for r in runs]),
                                "Elo": [elos.get(r["timestamp"], 0) for r in runs]})
        st.line_chart(elo_df, x="Date", y="Elo", color="#d4a574")
        st.divider()
        st.subheader(f"Latest Matchups — {latest['timestamp']}")
        cols = st.columns(min(4, len(latest["matchups"])))
        for i, m in enumerate(latest["matchups"]):
            with cols[i % len(cols)]:
                st.metric(label=m["opponent"], value=f'{m["wr"]:.0f}%',
                          delta=f'W{m["w"]} L{m["l"]} D{m["d"]}',
                          delta_color="normal" if m["wr"] >= 50 else "inverse")

    with tab_elo:
        st.subheader("Elo Rankings")
        if runs:
            model_elos = {r["timestamp"]: elos.get(r["timestamp"], 0) for r in runs}
            opp_elos = {k: v for k, v in elos.items() if k not in model_elos}
            st.markdown("### 🤖 Models")
            st.dataframe(pd.DataFrame([{"Model": n, "Elo": f"{e:.0f}",
                                         "Win Rate": f'{next((r["overall"]["wr"] for r in runs if r["timestamp"]==n),0):.1f}%'}
                                        for n, e in sorted(model_elos.items(), key=lambda x: -x[1])]),
                         use_container_width=True, hide_index=True)
            st.markdown("### ⚔️ Opponents")
            st.dataframe(pd.DataFrame([{"Opponent": n, "Elo": f"{e:.0f}", "LB Score": extract_lb_score(n) or "—"}
                                        for n, e in sorted(opp_elos.items(), key=lambda x: -x[1])]),
                         use_container_width=True, hide_index=True)

    with tab_matchups:
        st.subheader("Matchup Matrix")
        if runs:
            sel = st.selectbox("Select run", range(len(runs)),
                               format_func=lambda i: f'{runs[i]["timestamp"]} — {runs[i]["note"] or "no note"}',
                               index=len(runs) - 1)
            run = runs[sel]
            mdf = pd.DataFrame([{"Opponent": m["opponent"], "LB Score": extract_lb_score(m["opponent"]) or "—",
                                  "W": m["w"], "L": m["l"], "D": m["d"], "Win Rate": m["wr"]}
                                 for m in run["matchups"]])
            st.dataframe(mdf.style.background_gradient(subset=["Win Rate"], cmap="RdYlGn", vmin=0, vmax=100),
                         use_container_width=True, hide_index=True)
            st.bar_chart(mdf, x="Opponent", y="Win Rate", color="#d4a574")

    with tab_history:
        st.subheader("Tournament History")
        if runs:
            wr_df = pd.DataFrame({"Date": pd.to_datetime([r["timestamp"] for r in runs]),
                                   "Win Rate": [r["overall"]["wr"] if r["overall"] else 0 for r in runs]})
            st.bar_chart(wr_df, x="Date", y="Win Rate")
            st.divider()
            hdf = pd.DataFrame([{"Date": r["timestamp"], "Agent": Path(r["agent"]).name,
                                  "Games/Opp": r["games_per_opp"],
                                  "W": r["overall"]["w"] if r["overall"] else 0,
                                  "L": r["overall"]["l"] if r["overall"] else 0,
                                  "Win Rate": f'{r["overall"]["wr"]:.1f}%' if r["overall"] else "—",
                                  "Note": r["note"]} for r in reversed(runs)])
            st.dataframe(hdf, use_container_width=True, hide_index=True)

    with tab_config:
        st.subheader("Configuration")
        smoke = load_config(str(SMOKE_CONFIG))
        train = load_config(str(TRAIN_CONFIG))
        schema = load_schema()
        props = schema.get("properties", {})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🔥 Smoke Config")
            if smoke:
                st.dataframe(pd.DataFrame([{"Parameter": k, "Value": v,
                                             "Description": props.get(k, {}).get("description", "")}
                                            for k, v in smoke.items() if k != "$schema"]),
                             use_container_width=True, hide_index=True)
        with c2:
            st.markdown("### 🚂 Train Config")
            if train:
                st.dataframe(pd.DataFrame([{"Parameter": k, "Value": v,
                                             "Description": props.get(k, {}).get("description", "")}
                                            for k, v in train.items() if k != "$schema"]),
                             use_container_width=True, hide_index=True)
        st.divider()
        st.markdown("### 🏗️ Entrypoints")
        eps = [("tcg-data", "Kaggle dataset downloader"), ("tcg-build-bc", "BC dataset builder"),
               ("tcg-build-daily", "Single-replay dataset builder"), ("tcg-train", "MLX Metal GPU trainer"),
               ("tcg-evaluate", "1v1 evaluation"), ("tcg-tournament", "Multi-opponent tournament"),
               ("tcg-submission", "Build submission.tar.gz"), ("tcg-submit", "Submit to Kaggle"),
               ("tcg-dashboard", "This dashboard")]
        st.dataframe(pd.DataFrame([{"Command": f"uv run {c}", "Description": d} for c, d in eps]),
                     use_container_width=True, hide_index=True)


# ── only run UI when streamlit executes this script ──────────────
if __name__ == "__main__":
    run_app()
