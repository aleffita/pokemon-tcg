## 2026-08-16T19:03:09Z

You are the Milestone 1 Worker for the Pokémon TCG AI project.
Your working directory is: /Users/alefita/workdir/pokemon-tcg/.agents/worker_m1/

You MUST read /Users/alefita/workdir/pokemon-tcg/.agents/ORIGINAL_REQUEST.md before starting work.

Also read the survey handoff reports:
- /Users/alefita/workdir/pokemon-tcg/.agents/survey_miner_1/handoff.md (SQLite Card Miner: Deck #633, Deck #251, high-Elo card synergies, legal card IDs)
- /Users/alefita/workdir/pokemon-tcg/.agents/survey_miner_2/handoff.md (Opponent Panel Miner: 6 panel archetypes, threat vectors, counter-strategies)
- /Users/alefita/workdir/pokemon-tcg/.agents/survey_miner_3/handoff.md (Hypergeometric Modeler: P(Setup) >= 92% proof, energy curves, 7-prize asymmetry)

### MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

### MANDATORY HARD CONSTRAINTS:
1. ZERO GPU/MPS/Metal usage. 100% of compute belongs to Codex on Apple Silicon M3 Pro. All scripts must run on CPU in Python.
2. Package management: ALWAYS use `uv run python` if executing scripts.
3. Database queries: Read-only on `model/results.db`. Consult `docs/database_schema.md` first.

### DELIVERABLES:
1. Build `agent/deck.json`:
   - A valid JSON file containing an array of EXACTLY 60 integer Card IDs.
   - Every single integer Card ID MUST exist in the `cards` table of `model/results.db`.
   - The deck must obey Pokémon TCG deckbuilding rules:
     - Maximum 4 copies of any card with the same name (except Basic Energy).
     - Exactly 1 ACE SPEC card in the entire 60-card deck (e.g. Unfair Stamp [ID 1080]).
     - At least 10 Basic Pokémon to guarantee P(Setup within 1 mulligan) >= 92.0% and P(Mulligan) <= 8.0%.

2. Build `experiments/decks/deck_supreme_60.json`:
   - A comprehensive JSON deck capsule containing:
     - `deck_name`: "Deck Supreme 60 — Teal Mask Ogerpon ex / Turbo Acceleration & Psychic Counter Hybrid"
     - `archetype`: "Teal Mask Ogerpon ex / Grass Turbo Ramp / Anti-Meta Control"
     - `card_count`: 60
     - `card_list`: Array of card objects with fields: `id`, `name`, `category`, `stage`, `type`, `hp`, `rule`, `quantity`, and `role`
     - `energy_curve`: Breakdown of basic/special energy counts and turn-by-turn attachment expectations
     - `hypergeometric_probabilities`: Exact float and rational values for P(Setup n=7), P(Setup within 1 mulligan), P(Mulligan), P(T1 Energy), P(T1 Search Engine Access)
     - `matchup_profiles`: Strategic summary against the 6 panel archetypes (lb826 Alakazam, lb1009/945 Mega Lucario ex, lb814/600 Dragapult/Crustle, first_sub 2707, lb510 Mega Abomasnow, Deck #633)

3. Verify:
   - Run a python validation script via `uv run python` to confirm:
     - `agent/deck.json` has length 60, all elements are ints, all exist in `model/results.db`.
     - `experiments/decks/deck_supreme_60.json` is valid JSON and its card IDs sum to 60 and match `agent/deck.json`.
     - Rule compliance (max 4 per name, exactly 1 ACE SPEC, >=10 basics).

4. Update your `progress.md` with "Last visited: [timestamp]".
5. Write your handoff to `/Users/alefita/workdir/pokemon-tcg/.agents/worker_m1/handoff.md`.
6. Send message to parent when complete.
