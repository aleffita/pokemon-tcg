## 2026-08-16T18:57:54Z

You are the Project Orchestrator for the task defined in `/Users/alefita/workdir/pokemon-tcg/.agents/ORIGINAL_REQUEST.md`.

Your working directory is: `/Users/alefita/workdir/pokemon-tcg/.agents/orchestrator/`.
Please create your `BRIEFING.md` and `plan.md` in that working directory.

### Task Summary:
Tactical and adversarial engineering of a closed 60-card deck for Kaggle Pokémon TCG AI Challenge, maximizing win rate and invariant robustness during frozen ladder evaluation (Aug 16-31, 2026), integrated with Codex (GPT-5.6-Luna-Max) autoresearch protocol.

### Rigid Constraints:
1. ZERO GPU/MPS/Metal usage. 100% of compute belongs to Codex on Apple Silicon M3 Pro. Swarm work is strictly cognitive, combinatorial, and read-only SQLite analysis.
2. Package management: ALWAYS use `uv run python` if executing scripts.
3. Database queries: Read-only on `model/results.db`. Consult `docs/database_schema.md` first.
4. Deliverables & Acceptance Criteria:
   - `agent/deck.json`: Exactly 60 valid integer Card IDs matching SQLite IDs.
   - `experiments/decks/deck_supreme_60.json`: Deck capsule with archetype metadata, energy curve, and calculated probabilities.
   - `experiments/decks/DECK_SUPREME_60.md`: Monograph detailing each slot rationale, hypergeometric proof (P(Setup) >= 92%, P(Mulligan) <= 8%), and matchup matrix against the 6 panel archetypes.
   - `read-this-agent/08_DECK_SWARM_PROTOCOL.md`: Synchronized protocol with deck hashes and location.

Please decompose the task, spawn specialist subagents, coordinate progress in `progress.md`, and report back upon completion.
