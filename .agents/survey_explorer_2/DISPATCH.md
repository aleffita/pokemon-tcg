## 2026-08-14T14:09:35Z

You are Survey Explorer 2 (R2 Elite Pool Dataset & Oracles & Database Parity).
Your working directory is: `/Users/alefita/workdir/pokemon-tcg/.agents/survey_explorer_2/`
Project root: `/Users/alefita/workdir/pokemon-tcg`
Original user request: `/Users/alefita/workdir/pokemon-tcg/.agents/ORIGINAL_REQUEST.md`

Your mission:
Survey the dataset, database, and oracles:
1. Local replay archives in `data/bc_replay_zip/`, match filtering criteria (Elo >= 1100, ~100k target matches), parquet caching, vehicle draft sequence ingestion.
2. Corrected auxiliary target heads (`aux_ko`, `aux_prize_delta`, `aux_terminal`, `aux_return`) and C++ `bc_would_ko` damage oracles / annotations. Check `scripts/`, `model/`, `cpp/` or relevant files.
3. Database `model/results.db` schema and physical parity against raw JSON/ZIP archives on disk (check `docs/database_schema.md`, `rl/results_db.py`, `scripts/tournament.py`).
4. Tournament harness execution: 500-match benchmarks against `first_sub_kaggle_2707`, deck sweeps, Yan (#633) archetype metrics.
5. Enumerate all data pipelines, scripts, database tables, and validation requirements.

Rules:
- You are read-only: do NOT write or modify source code or database.
- Write your comprehensive findings to `/Users/alefita/workdir/pokemon-tcg/.agents/survey_explorer_2/analysis.md` and `/Users/alefita/workdir/pokemon-tcg/.agents/survey_explorer_2/handoff.md`.
- Send a completion message to parent when done.
