## 2026-08-14T11:15:33-03:00
You are a specialized Test Writer for the E2E Testing Track of the Pokémon TCG project.
Your working directory is: `/Users/alefita/workdir/pokemon-tcg/.agents/test_writer_tier3/`
Project workspace root: `/Users/alefita/workdir/pokemon-tcg`

MANDATORY INPUTS:
- Read `/Users/alefita/workdir/pokemon-tcg/.agents/ORIGINAL_REQUEST.md`
- Read `/Users/alefita/workdir/pokemon-tcg/PROJECT.md`
- Read `/Users/alefita/workdir/pokemon-tcg/TEST_INFRA.md`

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

EXCLUSIVE WRITE OWNERSHIP:
You own `tests/e2e/test_tier3_pairwise.py` (and creating `tests/e2e/__init__.py` if needed). Do NOT write to any other test files.

TASK:
Write the complete Tier 3 Cross-Feature Pairwise Interaction test suite in `tests/e2e/test_tier3_pairwise.py`.
Requirements:
1. Cover Cross-Feature Pairwise combinations across the 16 features from `PROJECT.md § Feature Inventory` and `TEST_INFRA.md`:
   - Pair 1: Feature 1 (4D RoPEND PyTorch) + Feature 3 (MoE 4-Expert Topology) — interaction between rotary position embeddings and expert routing in transformer backbone.
   - Pair 2: Feature 3 (MoE Topology) + Feature 4 (MoE Load Balancing Loss) — joint forward and loss computation preventing expert collapse.
   - Pair 3: Feature 3 (MoE Topology) + Feature 6 (Apex Mode Runtime Airgap) — dynamic temperature scaling effect on MoE Top-2 routing sharpness.
   - Pair 4: Feature 1 (4D RoPEND) + Feature 7 (Strict FP32 Precision Contract) — precision contract preservation during Givens rotation and trigonometric embedding.
   - Pair 5: Feature 5 (Vehicle Draft) + Feature 3 (MoE Topology) — vehicle draft embedding injected into MoE sequence encoder.
   - Pair 6: Feature 2 (4D RoPEND MLX) + Feature 7 (Strict FP32 Precision Contract) — MLX rotary transformations under FP32 contract with AdamW/Muon state consistency.
   - Pair 7: Feature 8 (Elite Dataset Compilation) + Feature 9 (Corrected Aux Heads & C++ Oracles) — replay batch generation with aux targets (aux_ko, aux_prize_delta, aux_terminal, aux_return, bc_would_ko).
   - Pair 8: Feature 8 (Elite Dataset) + Feature 10 (SQLite FK Parity) — verifying elite replay match steps match database schema without orphaned foreign keys.
   - Pair 9: Feature 9 (Aux Heads) + Feature 10 (SQLite Database) — auxiliary head targets stored and retrieved from normalized results.db tables.
   - Pair 10: Feature 11 (PageRank-Abelian Monograph) + Feature 12 (Master RFC & Metanoia Index) — monograph theoretical citations aligned with Master RFC and Metanoia 01..06 index.
   - Pair 11: Feature 13 (Wikifita Cross-Project Sync) + Feature 14 (Wikifita Double Audit) — synchronized pages verified by double-pass audit engine (`--fix` then zero-error validation).
   - Pair 12: Feature 11 (Abelian Invariant Elo) + Feature 15 (Tournament Benchmark) — tournament match results updating Bradley-Terry Softmax Abelian Elo ratings in real-time.
   - Pair 13: Feature 1 (4D RoPEND) + Feature 15 (Tournament Benchmark) — tournament inference loop running policy network with 4D RoPEND coordinates.
   - Pair 14: Feature 6 (Apex Mode Airgap) + Feature 16 (Yan #633 Win Rate Target) — evaluation of Yan #633 win rate before vs after Apex Mode activation timestamp.
   - Pair 15: Feature 7 (FP32 Contract) + Feature 15 (Tournament Benchmark) — tournament execution actively checking static feature checksum and FP32 tensor safety during 500 matches.
   - Pair 16: Feature 10 (SQLite DB Parity) + Feature 16 (Yan #633 Win Rate Target) — querying Deck #633 tournament match history and computing MD10 smoothed Bradley-Terry invariant Elo.
2. Total test count: MUST BE AT LEAST 16 test cases covering distinct feature pairs.
3. Use `unittest.TestCase` classes grouped logically.
4. Verify your test file by running `uv run python -m unittest tests/e2e/test_tier3_pairwise.py` using run_command.
5. Write your handoff report to `/Users/alefita/workdir/pokemon-tcg/.agents/test_writer_tier3/handoff.md` and send a message when complete.
