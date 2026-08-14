# Scope: Milestone 2 — Elite Match Dataset & Database Relational Parity

## Architecture
- **Relational Integrity**: `model/results.db` Schema 2.0.0. Complete physical parity against disk archives (`data/bc_replay_zip/*.zip`), 0 orphaned rows, and 0 errors under `PRAGMA foreign_key_check`.
- **Elite Dataset Pipeline**: Filter matches with daily Elo >= 1100.0 (~40k - 100k matches, ~6.5M - 12M decision steps) from local replay zips, encoding with Level 2 auxiliary targets (`aux_ko`, `aux_prize_delta`, `aux_terminal`, `aux_return`) and native C++ `bc_would_ko` damage annotations.
- **Precision Contract**: Strict FP32 alignment for decision tensors and static card features.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 8 | Elite Match Dataset Compilation | Filter and compile clean dataset (Elo >= 1100, ~100k matches) from replay archives | M2 | Survey R2 |
| 9 | Corrected Aux Heads & C++ Oracles | `aux_ko`, `aux_prize_delta`, `aux_terminal`, `aux_return`, and C++ `bc_would_ko` annotations | M2 | Survey R2 |
| 10 | SQLite FK Parity & Clean-up | Purge 2.94M orphaned rows in `match_steps` & `match_card_usage`; pass `PRAGMA foreign_key_check` | M2 | Survey R2 |

## Milestones / Work Items
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | DB FK Purge & Parity Verification | Purge orphaned records in `match_steps` (2.48M) and `match_card_usage` (458k); verify `PRAGMA foreign_key_check` returns 0 rows | none | IN_PROGRESS |
| 2 | Elite Pool Filter & Dataset Verification | Verify filtering logic for Elo >= 1100 in DB and `scripts/bc/build_bc_from_zips.py`; verify parquet metadata and row counts | none | IN_PROGRESS |
| 3 | Aux Targets & C++ Would-KO Oracle Verification | Validate 4 aux targets and `annotate_would_ko_with_audit` with native C++ `cg.api` | none | IN_PROGRESS |

## Interface Contracts
### `model/results.db`
- Foreign keys enabled: `PRAGMA foreign_keys = ON;`
- Integrity check: `PRAGMA foreign_key_check;` must return 0 rows.
- Total active matches: ~139,783 (138,023 remote, 1,760 local).
- Daily Elo snapshots: `agent_elo_daily` populated for all 30 days.

### Elite Parquet Dataset (`data/bc_data/`)
- Schema Version: `3.0` (90 columns, FP32 decision tensors, FixedSizeListArray).
- Auxiliary columns: `aux_ko` (float32/int32), `aux_prize_delta` (float32), `aux_terminal` (float32), `aux_return` (float32), `aux_valid` (float32).
- Option attributes: `opt_attr` shape `[max_options, 27]`, with `would_ko` flags populated via native C++ rollouts.
