## 2026-08-14T14:15:06Z
You are DB Integrity Explorer for Milestone 2.
Your working directory is: `/Users/alefita/workdir/pokemon-tcg/.agents/explorer_m2_db/`
Project workspace root: `/Users/alefita/workdir/pokemon-tcg`
Mandatory reading:
- `/Users/alefita/workdir/pokemon-tcg/.agents/ORIGINAL_REQUEST.md`
- `/Users/alefita/workdir/pokemon-tcg/PROJECT.md`
- `/Users/alefita/workdir/pokemon-tcg/.agents/sub_orch_m2/SCOPE.md`
- `/Users/alefita/workdir/pokemon-tcg/.agents/survey_explorer_2/analysis.md`
- `/Users/alefita/workdir/pokemon-tcg/docs/database_schema.md`
- `/Users/alefita/workdir/pokemon-tcg/rl/results_db.py`

Your task:
1. Conduct an empirical investigation of SQLite database `model/results.db`.
2. Inspect `PRAGMA foreign_key_check;` and diagnose the exact orphaned records in `match_steps` (~2,488,290) and `match_card_usage` (~458,046). Check if there are cascade relations or other tables (e.g. `pokemon_on_field`, `step_events`, `step_options`, `board_snapshots`) affected.
3. Formulate the exact, transaction-safe, atomic SQL purge script and python execution procedure (using `uv run`).
4. Verify total physical match counts, table counts, and parity before and after the proposed cleanup.
5. Write your comprehensive report to `/Users/alefita/workdir/pokemon-tcg/.agents/explorer_m2_db/analysis.md` and write a self-contained handoff to `/Users/alefita/workdir/pokemon-tcg/.agents/explorer_m2_db/handoff.md`.
6. Send a completion message back to parent when done.
