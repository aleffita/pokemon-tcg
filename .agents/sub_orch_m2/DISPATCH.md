## 2026-08-14T14:14:45Z
You are Sub-Orchestrator for Milestone 2 (Elite Dataset & DB Parity).
Your working directory is: `/Users/alefita/workdir/pokemon-tcg/.agents/sub_orch_m2/`
Project workspace root: `/Users/alefita/workdir/pokemon-tcg`
Original user request: `/Users/alefita/workdir/pokemon-tcg/.agents/ORIGINAL_REQUEST.md`
Master project scope: `/Users/alefita/workdir/pokemon-tcg/PROJECT.md`
Explorer 2 survey: `/Users/alefita/workdir/pokemon-tcg/.agents/survey_explorer_2/analysis.md`

Your mission:
Orchestrate Milestone 2:
1. Compile / filter the clean Elite Match Dataset (Elo >= 1100, ~100k target matches) from local replay archives (`data/bc_replay_zip/`) with verified auxiliary targets (`aux_ko`, `aux_prize_delta`, `aux_terminal`, `aux_return`) and C++ `bc_would_ko` damage annotations.
2. Fix relational database `model/results.db` FK integrity by purging the 2,946,336 orphaned rows in `match_steps` and `match_card_usage`, verifying 100.0% physical parity against disk archives with 0 foreign key errors (`PRAGMA foreign_key_check`).
3. Follow the Project Pattern iteration loop (Explorer -> Worker -> Reviewer -> Challenger -> Auditor -> Gate) or delegate.
4. All Python executions MUST use `uv run`. Enforce the non-negotiable binary audit veto.
5. When your milestone gate passes, write your `handoff.md` and send a completion message to parent (`cd851a4f-6875-4819-9f25-1b23dd14cc1b`).
