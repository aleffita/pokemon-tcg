# E2E Test Infra: Magnum Opus Pokémon TCG

## Test Philosophy
- Opaque-box, requirement-driven. Derives test assertions strictly from `ORIGINAL_REQUEST.md` and user specifications.
- Methodology: Category-Partition + Boundary Value Analysis (BVA) + Pairwise Interaction Testing + Workload Testing.

## Feature Inventory & Test Coverage
| # | Feature | Source (Requirement) | Tier 1 (Feature) | Tier 2 (Boundary) | Tier 3 (Pairwise) |
|---|---------|---------------------|:----------------:|:-----------------:|:-----------------:|
| 1 | 4D RoPEND Operator (PyTorch) | ORIGINAL_REQUEST §1 | 5 | 5 | ✓ |
| 2 | 4D RoPEND Operator (MLX) | ORIGINAL_REQUEST §1 | 5 | 5 | ✓ |
| 3 | MoE 4-Expert Topology | ORIGINAL_REQUEST §1 | 5 | 5 | ✓ |
| 4 | MoE Load Balancing Loss | ORIGINAL_REQUEST §1 | 5 | 5 | ✓ |
| 5 | Vehicle Cross-Attention Draft | ORIGINAL_REQUEST §1 | 5 | 5 | ✓ |
| 6 | Apex Mode Runtime Airgap | ORIGINAL_REQUEST §1 | 5 | 5 | ✓ |
| 7 | Strict FP32 Precision Contract | Acceptance Criteria 1 | 5 | 5 | ✓ |
| 8 | Elite Match Dataset Compilation | ORIGINAL_REQUEST §2 | 5 | 5 | ✓ |
| 9 | Corrected Aux Heads & C++ Oracles | ORIGINAL_REQUEST §2 | 5 | 5 | ✓ |
| 10 | SQLite FK Parity & Parity Check | Acceptance Criteria 4 | 5 | 5 | ✓ |
| 11 | PageRank-Abelian Monograph | ORIGINAL_REQUEST §3 | 5 | 5 | ✓ |
| 12 | Master RFC & Metanoia Index | Acceptance Criteria 6 | 5 | 5 | ✓ |
| 13 | Wikifita Cross-Project Sync | ORIGINAL_REQUEST §4 | 5 | 5 | ✓ |
| 14 | Wikifita Double Audit | Acceptance Criteria 5 | 5 | 5 | ✓ |
| 15 | 500-Match Tournament Benchmark | Acceptance Criteria 2 | 5 | 5 | ✓ |
| 16 | Yan #633 Win Rate Target | Acceptance Criteria 3 | 5 | 5 | ✓ |

## Test Architecture
- **Test Runner**: `uv run python -m unittest discover -s tests/e2e -p "test_*.py"`
- **Pass/Fail Semantics**: 100% assertions pass, exit code 0.
- **Directory Layout**:
  - `tests/e2e/test_tier1_features.py`: Feature isolation tests (>=80 tests).
  - `tests/e2e/test_tier2_boundaries.py`: Edge cases, numerical underflow/overflow, boundary coordinates (>=80 tests).
  - `tests/e2e/test_tier3_pairwise.py`: Cross-feature combinations and data flow interactions (>=16 tests).
  - `tests/e2e/test_tier4_scenarios.py`: End-to-end tournament simulations and Wikifita audit integrations (>=8 tests).

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Full Tournament Simulation against `first_sub` | F1, F3, F6, F7, F15, F16 | High |
| 2 | Replay Archive to Parquet Ingestion with C++ Oracles | F8, F9, F10 | High |
| 3 | Wikifita Full Repository Ingestion and Double Audit | F13, F14 | Medium |
| 4 | Mathematical Spectral Teleportation vs Abelian Group Convergence | F11, F12 | Medium |
| 5 | Live Inference Checksum & Precision Guard Test | F1, F6, F7 | Medium |

## Coverage Thresholds
- Tier 1: ≥ 5 per feature (Total ≥ 80 tests)
- Tier 2: ≥ 5 per feature (Total ≥ 80 tests)
- Tier 3: ≥ 16 tests covering major feature pairs
- Tier 4: ≥ 8 application-level scenario tests
- **Total Minimum Target**: ≥ 184 test cases
