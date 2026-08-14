## 2026-08-14T14:15:34Z
You are a specialized Test Writer for the E2E Testing Track of the Pokémon TCG project.
Your working directory is: `/Users/alefita/workdir/pokemon-tcg/.agents/test_writer_tier4/`
Project workspace root: `/Users/alefita/workdir/pokemon-tcg`

MANDATORY INPUTS:
- Read `/Users/alefita/workdir/pokemon-tcg/.agents/ORIGINAL_REQUEST.md`
- Read `/Users/alefita/workdir/pokemon-tcg/PROJECT.md`
- Read `/Users/alefita/workdir/pokemon-tcg/TEST_INFRA.md`

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

EXCLUSIVE WRITE OWNERSHIP:
You own `tests/e2e/test_tier4_scenarios.py` (and creating `tests/e2e/__init__.py` if needed). Do NOT write to any other test files.

TASK:
Write the complete Tier 4 Real-World Application Scenarios test suite in `tests/e2e/test_tier4_scenarios.py`.
Requirements:
1. Cover comprehensive end-to-end user workflows across all 16 features from `TEST_INFRA.md § Real-World Application Scenarios`:
   - Scenario 1: Full Tournament Simulation Workflow against `first_sub` (exercising F1, F3, F6, F7, F15, F16) — end-to-end tournament setup, policy initialization with 4D RoPEND & MoE, Apex Mode clock check, FP32 precision check, match execution simulation, and Yan #633 win rate tracking.
   - Scenario 2: Replay Archive to Parquet Ingestion with C++ Oracles (exercising F8, F9, F10) — parsing replay zip archive, filtering Elo >= 1100, generating aux target tensors with damage oracle annotations, and validating against SQLite FK constraints.
   - Scenario 3: Wikifita Full Repository Ingestion and Double Audit (exercising F13, F14) — inspecting Wikifita canonical cross-project directories (`kaggle/`, `co-scientist/`), verifying `index.md` / `log.md`, checking markdown backtick hierarchy adherence, and running two-pass audit validation.
   - Scenario 4: Mathematical Spectral Teleportation vs Abelian Group Convergence (exercising F11, F12) — verifying mathematical monograph theorem alignment, PageRank dangling node mass transition matrix vs Bradley-Terry Abelian Elo translation isomorphism, and RFC / Metanoia index consistency.
   - Scenario 5: Live Inference Checksum & Precision Guard E2E (exercising F1, F6, F7) — full inference pipeline with card feature SHA256 checksum validation, FP32 dtype guard, Apex Mode temperature scaling, and policy output shape validation.
   - Scenario 6: Full Deck Matrix Sweeps & Disaggregated Invariant Elo Pipeline (exercising F10, F15, F16) — executing N x M deck matrix matchup simulations, computing disaggregated win rates, and deriving Bradley-Terry invariant Elo with MD10 regularizer.
   - Scenario 7: MoE Checkpoint Save/Load and Cross-Framework Parity (exercising F1, F2, F3, F7) — saving model weights, validating static feature contract, and verifying weight structure compatibility between PyTorch and MLX representations.
   - Scenario 8: End-to-End Release Readiness & Integrity Audit Workflow (exercising all features F1-F16) — composite acceptance test checking test discoverability, database relational integrity, monograph artifacts, and Kaggle submission packager compliance.
2. Total test count: MUST BE AT LEAST 8 test cases covering realistic application-level scenarios.
3. Use `unittest.TestCase` classes grouped logically.
4. Verify your test file by running `uv run python -m unittest tests/e2e/test_tier4_scenarios.py` using run_command.
5. Write your handoff report to `/Users/alefita/workdir/pokemon-tcg/.agents/test_writer_tier4/handoff.md` and send a message when complete.
