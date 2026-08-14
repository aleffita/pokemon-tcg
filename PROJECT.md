# Project: Magnum Opus MoE, Elite Dataset, PageRank-Abelian Monograph & Wikifita Sync

## Architecture
- **Neural Engine**: 4D RoPEND (Rotary Positional Embedding over $c_1$: Step, $c_2$: Meta-Epoch, $c_3$: Urgency, $c_4$: Inferred Elo) with multi-head subspace partitioning ($4 \times 8 = 32$-dim Givens rotations per attention head).
- **MoE Topology**: 4 tactical experts (Agro, Control, Setup, Endgame) with Top-2 router, load-balancing auxiliary loss $\mathcal{L}_{\text{balance}}$, vehicle cross-attention draft (60-card synergy), and runtime Apex Mode airgap activation token at `2026-08-16T00:00:00Z` ($\tau = 0.1$).
- **Precision Contract**: Strict FP32 precision contract across PyTorch inference (`rl/policy_infer_torch.py`) and MLX training (`scripts/bc/bc_train_mlx.py`), with static card feature SHA256 checksum and shape validation.
- **Dataset & ETL Pipeline**: Clean Elite Match Dataset (Elo >= 1100, ~100k matches) compiled from local replay archives (`data/bc_replay_zip/`), 4 corrected auxiliary target heads (`aux_ko`, `aux_prize_delta`, `aux_terminal`, `aux_return`), and C++ native `bc_would_ko` damage annotations.
- **Relational Integrity**: SQLite database (`model/results.db`) with Schema 2.0.0, zero foreign key errors (`PRAGMA foreign_key_check`), and 100.0% physical parity against disk archives.
- **Mathematical Monograph**: Algebraic and spectral isomorphism between PageRank Markov chain stationarity (dangling node mass redistribution) and Bradley-Terry Softmax Abelian Group Elo calibration.
- **Wikifita Ecosystem**: Canonical cross-project sync (`~/Claude/wikifita/kaggle/`, `~/Claude/wikifita/co-scientist/`), double audit validation (`scripts/wikifita_audit.py`), and Master RFC (`docs/technical_handoff_rfc.md`) + Metanoia suite (`docs/metanoia/01..06`) index preservation.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | 4D RoPEND Operator (PyTorch) | 4-axis rotary positional embedding ($c_1, c_2, c_3, c_4$) with 4x8 Givens rotation planes per 32-dim head | M1 | Survey R1 |
| 2 | 4D RoPEND Operator (MLX) | MLX native implementation of 4D RoPEND for Apple Silicon training | M1 | Survey R1 |
| 3 | MoE 4-Expert Topology | 4 specialized FFN experts (Agro, Control, Setup, Endgame) with Top-2 routing | M1 | Survey R1 |
| 4 | MoE Load Balancing Loss | $\mathcal{L}_{\text{balance}} = \alpha_{\text{balance}} E \sum f_e P_e$ to prevent expert starvation | M1 | Survey R1 |
| 5 | Vehicle Cross-Attention Draft | Autoregressive cross-attention over 60-card vehicle deck before step 0 | M1 | Survey R1 |
| 6 | Apex Mode Runtime Airgap | Deterministic exploitation switch on `datetime.now(UTC) >= 2026-08-16` ($\tau = 0.1$) | M1 | Survey R1 |
| 7 | Strict FP32 Precision Contract | PyTorch inference checksum, FP32 static feature validation, and MLX Muon/AdamW FP32 states | M1 | Survey R1 |
| 8 | Elite Match Dataset Compilation | Filter and compile clean dataset (Elo >= 1100, ~100k matches) from replay archives | M2 | Survey R2 |
| 9 | Corrected Aux Heads & C++ Oracles | `aux_ko`, `aux_prize_delta`, `aux_terminal`, `aux_return`, and C++ `bc_would_ko` annotations | M2 | Survey R2 |
| 10 | SQLite FK Parity & Clean-up | Purge 2.94M orphaned rows in `match_steps` & `match_card_usage`; pass `PRAGMA foreign_key_check` | M2 | Survey R2 |
| 11 | PageRank-Abelian Monograph | Formal technical monograph on PageRank stationarity vs Bradley-Terry Abelian Elo isomorphism | M3 | Survey R3 |
| 12 | Master RFC & Metanoia Suite Index | Maintain index in `docs/technical_handoff_rfc.md` and sync `docs/metanoia/01..06` | M3 | Survey R3 |
| 13 | Wikifita Kaggle & Co-Scientist Sync | Synchronize `~/Claude/wikifita/kaggle/` and `~/Claude/wikifita/co-scientist/` with August 2026 architecture | M4 | Survey R4 |
| 14 | Wikifita Double Audit Verification | Run `uv run scripts/wikifita_audit.py --fix` and verify second pass without `--fix` (0 errors) | M4 | Survey R4 |
| 15 | 500-Match Tournament Benchmark | Execute 500 matches against `first_sub_kaggle_2707` without process hangs, NaN, or leaks | M5 | Acceptance |
| 16 | Yan Archetype Win Rate Target | Achieve > 40% win rate against `first_sub` on Deck #633 (Teal Mask Ogerpon ex) | M5 | Acceptance |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | 4D RoPEND & MoE Neural Architecture | Implement 4D RoPEND in PyTorch & MLX, MoE router, experts, load balance loss, vehicle draft, Apex token, and FP32 contract validation | none | IN_PROGRESS (sub_orch_m1: 9a189410-43b1-4cdc-bc2a-7a942180e59c) |
| M2 | Elite Dataset Compilation & DB Parity | Re-compile Elite pool dataset (Elo >= 1100), aux targets, C++ `bc_would_ko` damage oracles, and purge orphaned DB rows | none | IN_PROGRESS (sub_orch_m2: f5143692-4dba-4e8a-aa34-f7465d296f9b) |
| M3 | PageRank-Abelian Graph Invariance Monograph | Author dedicated monograph in `docs/` and project reference on spectral PageRank vs Bradley-Terry Abelian Elo isomorphism; maintain Metanoia suite | none | IN_PROGRESS (sub_orch_m3: 4877bc7d-bfc2-44d3-bc55-1a9dd628ba39) |
| M4 | Canonical Wikifita Cross-Project Sync | Synchronize Wikifita pages (`kaggle/`, `co-scientist/`, `pesquisas/`), verify `index.md`, `log.md`, symlinks, and pass double audit | M3 | PLANNED |
| M5 | E2E Integration, Tournament Validation & Hardening | Execute 500-match benchmark against `first_sub_kaggle_2707`, verify Yan #633 WR > 40%, run E2E test suite (Tiers 1-4) + adversarial hardening (Tier 5) | M1, M2, M4 | PLANNED |

## Interface Contracts
### 4D RoPEND Operator (`rl/ropend/`)
- Function: `apply_ropend_4d(x, c1, c2, c3, c4, freqs_cos, freqs_sin)`
- Input tensor shapes:
  - $x$: `(batch, seq_len, num_heads, head_dim)` where `head_dim = 32`, `num_heads = 4`
  - $c_1$: `(batch, seq_len)` int/float step coordinate
  - $c_2$: `(batch, seq_len)` float meta-epoch coordinate
  - $c_3$: `(batch, seq_len)` float countdown coordinate
  - $c_4$: `(batch, seq_len)` float continuous Elo coordinate
- Output tensor: `(batch, seq_len, num_heads, head_dim)` in float32.

### MoE Router & Experts (`rl/moe/`)
- Function: `Top2MoERouter.forward(x, apex_mode=False)`
- Input: `x` `(batch, seq_len, hidden_dim)`
- Output: `(routed_output, aux_loss, routing_weights)`
- Temperature: $\tau = 1.0$ (standard) / $\tau = 0.1$ (Apex Mode when `datetime.now(UTC) >= 2026-08-16`).

### Database Relational Model (`model/results.db`)
- Normalized Schema 2.0.0. `PRAGMA foreign_key_check` returns 0 rows.
- Invariant Elo derivation via `ResultsDB.get_invariant_deck_elo(deck_id, source="local")`.

## Code Layout
- `rl/ropend/`: `ropend_torch.py`, `ropend_mlx.py`, `__init__.py`
- `rl/moe/`: `router.py`, `experts.py`, `load_balance.py`, `__init__.py`
- `rl/deck/`: `vehicle_draft.py`
- `rl/policy_moe_torch.py`: PyTorch inference engine with MoE and 4D RoPEND
- `rl/policy_moe_mlx.py`: MLX training model with MoE and 4D RoPEND
- `rl/results_db.py`: Database API, Elo calibration, and FK verification
- `docs/pagerank_and_abelian_graph_invariance.md`: Mathematical monograph
- `docs/technical_handoff_rfc.md`: Master RFC
- `docs/metanoia/`: Metanoia suite 01..06
- `~/Claude/wikifita/`: Canonical Wikifita repository
- `tests/e2e/`: E2E test suite (Tiers 1-5)
