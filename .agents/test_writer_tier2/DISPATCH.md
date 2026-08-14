## 2026-08-14T14:15:33Z

You are a specialized Test Writer for the E2E Testing Track of the Pokémon TCG project.
Your working directory is: `/Users/alefita/workdir/pokemon-tcg/.agents/test_writer_tier2/`
Project workspace root: `/Users/alefita/workdir/pokemon-tcg`

MANDATORY INPUTS:
- Read `/Users/alefita/workdir/pokemon-tcg/.agents/ORIGINAL_REQUEST.md`
- Read `/Users/alefita/workdir/pokemon-tcg/PROJECT.md`
- Read `/Users/alefita/workdir/pokemon-tcg/TEST_INFRA.md`

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

EXCLUSIVE WRITE OWNERSHIP:
You own `tests/e2e/test_tier2_boundaries.py` (and creating `tests/e2e/__init__.py` if needed). Do NOT write to any other test files.

TASK:
Write the complete Tier 2 Boundary and Corner Cases test suite in `tests/e2e/test_tier2_boundaries.py`.
Requirements:
1. Cover Boundary & Corner cases for ALL 16 features from `PROJECT.md § Feature Inventory` and `TEST_INFRA.md`:
   - Feature 1: 4D RoPEND PyTorch Boundaries (≥5 tests: e.g. step c1=0 and extreme c1=100000, meta-epoch c2=0.0 and c2=1.0, urgency clock c3=0.0 and extreme urgency, Elo c4=0 and c4=3000 extreme ratings, zero-length / batch=1 tensor dimensions)
   - Feature 2: 4D RoPEND MLX Boundaries (≥5 tests: e.g. empty/minimal batch sizes, extreme coordinate arrays, boundary head dimensions, FP32 numerical stability with large coordinates, NaN/Inf coordinate rejection)
   - Feature 3: MoE Topology Boundaries (≥5 tests: e.g. all router logits equal/tied, extreme logit disparity, single token sequence, empty batch, top-2 selection with 4 experts under extreme gating temperatures)
   - Feature 4: MoE Load Balancing Boundaries (≥5 tests: e.g. zero alpha_balance coefficient, 100% single-expert collapse scenario, uniform expert allocation, negative gradient protection, large expert pool scaling)
   - Feature 5: Vehicle Draft Boundaries (≥5 tests: e.g. non-standard deck sizes <60 or >60 error handling, duplicate card indices, deck with all identical cards, zero-padding sequences, maximum sequence length limits)
   - Feature 6: Apex Mode Airgap Boundaries (≥5 tests: e.g. exact second before switch `2026-08-15T23:59:59Z`, exact switch second `2026-08-16T00:00:00Z`, 1 second after, leap second / timezone boundary handling, extreme temperature clamp tau -> 0)
   - Feature 7: Strict FP32 Precision Boundaries (≥5 tests: e.g. float16 tensor injection rejection, bfloat16 rejection, float64 downcast enforcement/rejection, corrupt SHA256 checksum mismatch, subnormal float handling)
   - Feature 8: Elite Dataset Boundaries (≥5 tests: e.g. exact boundary Elo = 1100 inclusion, Elo = 1099 exclusion, empty replay zip archive handling, corrupted zip header, 0-turn match replay handling)
   - Feature 9: Aux Heads & C++ Oracle Boundaries (≥5 tests: e.g. prize delta = -6 and +6 limits, aux_ko probability 0.0 and 1.0, terminal return bounds, C++ oracle with 0 HP Pokémon, C++ oracle with overkill 990 damage)
   - Feature 10: SQLite FK Parity Boundaries (≥5 tests: e.g. foreign key constraint violation handling, empty tables query behavior, maximum integer match_id boundary, SQL injection resistance, atomic rollback on failed transaction)
   - Feature 11: PageRank-Abelian Monograph Boundaries (≥5 tests: e.g. dangling node mass alpha -> 0 and alpha -> 1 limit cases, single isolated node graph, disconnected subgraph components, zero win rate Elo clipping to w=0.02, 100% win rate Elo clipping to w=0.98)
   - Feature 12: Master RFC & Metanoia Boundaries (≥5 tests: e.g. missing metanoia file detection, empty RFC section detection, duplicate heading validation, RFC versioning boundary, deep link verification)
   - Feature 13: Wikifita Sync Boundaries (≥5 tests: e.g. empty directory handling, circular wikilinks detection, invalid frontmatter YAML handling, deep nested directory paths, unicode/accented filename handling)
   - Feature 14: Wikifita Double Audit Boundaries (≥5 tests: e.g. dry-run mode boundary, audit failure on corrupted markdown backticks (level 1..4 hierarchy), zero-error clean repo idempotence, corrupted symlink detection, audit timeout handling)
   - Feature 15: 500-Match Tournament Boundaries (≥5 tests: e.g. 0 matches requested, 1 match single-step execution, opponent crash simulation, timeout kill handling, disk full/write error during match logging)
   - Feature 16: Yan Archetype Win Rate Boundaries (≥5 tests: e.g. 0-win sample smoothing with N0=10 prior, 100-win sample smoothing, invalid deck id rejection, extreme opponent Elo disparity, threshold margin calculation exactly at 40.0%)
2. Total test count: MUST BE AT LEAST 80 test cases (≥5 for each of the 16 features).
3. Use `unittest.TestCase` classes grouped logically.
4. Verify your test file by running `uv run python -m unittest tests/e2e/test_tier2_boundaries.py` using run_command.
5. Write your handoff report to `/Users/alefita/workdir/pokemon-tcg/.agents/test_writer_tier2/handoff.md` and send a message when complete.
