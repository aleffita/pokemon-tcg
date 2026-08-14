## 2026-08-14T11:15:33-03:00
You are a specialized Test Writer for the E2E Testing Track of the Pokémon TCG project.
Your working directory is: `/Users/alefita/workdir/pokemon-tcg/.agents/test_writer_tier1/`
Project workspace root: `/Users/alefita/workdir/pokemon-tcg`

MANDATORY INPUTS:
- Read `/Users/alefita/workdir/pokemon-tcg/.agents/ORIGINAL_REQUEST.md`
- Read `/Users/alefita/workdir/pokemon-tcg/PROJECT.md`
- Read `/Users/alefita/workdir/pokemon-tcg/TEST_INFRA.md`

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

EXCLUSIVE WRITE OWNERSHIP:
You own `tests/e2e/test_tier1_features.py` (and creating `tests/e2e/__init__.py` if needed). Do NOT write to any other test files.

TASK:
Write the complete Tier 1 Feature Coverage test suite in `tests/e2e/test_tier1_features.py`.
Requirements:
1. Cover ALL 16 features from `PROJECT.md § Feature Inventory` and `TEST_INFRA.md § Feature Inventory`:
   - Feature 1: 4D RoPEND Operator (PyTorch) (≥5 tests)
   - Feature 2: 4D RoPEND Operator (MLX) (≥5 tests)
   - Feature 3: MoE 4-Expert Topology (≥5 tests)
   - Feature 4: MoE Load Balancing Loss (≥5 tests)
   - Feature 5: Vehicle Cross-Attention Draft (≥5 tests)
   - Feature 6: Apex Mode Runtime Airgap (≥5 tests)
   - Feature 7: Strict FP32 Precision Contract (≥5 tests)
   - Feature 8: Elite Match Dataset Compilation (≥5 tests)
   - Feature 9: Corrected Aux Heads & C++ Oracles (≥5 tests)
   - Feature 10: SQLite FK Parity & Parity Check (≥5 tests)
   - Feature 11: PageRank-Abelian Monograph (≥5 tests)
   - Feature 12: Master RFC & Metanoia Index (≥5 tests)
   - Feature 13: Wikifita Cross-Project Sync (≥5 tests)
   - Feature 14: Wikifita Double Audit (≥5 tests)
   - Feature 15: 500-Match Tournament Benchmark (≥5 tests)
   - Feature 16: Yan Archetype Win Rate Target (≥5 tests)
2. Total test count: MUST BE AT LEAST 80 test cases (≥5 for each of the 16 features).
3. Use `unittest.TestCase` classes grouped logically (e.g. `TestFeature01RoPENDPyTorch`, `TestFeature02RoPENDMLX`, ..., `TestFeature16YanWinRate`).
4. Tests should be robust: use dynamic imports or contract checks where modules are being developed in parallel, but test actual functions/classes if available or test specification contracts/schemas rigorously.
5. Verify your test file by running `uv run python -m unittest tests/e2e/test_tier1_features.py` using run_command.
6. Write your handoff report to `/Users/alefita/workdir/pokemon-tcg/.agents/test_writer_tier1/handoff.md` and send a message when complete.
