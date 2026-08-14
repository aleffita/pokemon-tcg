# Scope: Milestone 1 — 4D RoPEND & MoE Neural Architecture

## Architecture
- **4D RoPEND Operator**: Rotary Positional Embedding across 4 axes ($c_1$: Step, $c_2$: Meta-Epoch, $c_3$: Urgency Clock, $c_4$: Inferred Elo). Multi-head subspace partitioning ($4 \times 8 = 32$-dim Givens rotations per attention head, 4 heads = 128 embedding dim). Implemented in PyTorch (`rl/ropend/ropend_torch.py`) and MLX (`rl/ropend/ropend_mlx.py`).
- **MoE 4-Expert Topology**: 4 specialized feedforward experts (Agro, Control, Setup, Endgame) with Top-2 router (`rl/moe/router.py`, `rl/moe/experts.py`).
- **Load Balancing Auxiliary Loss**: $\mathcal{L}_{\text{balance}} = \alpha_{\text{balance}} E \sum_{e=1}^E f_e P_e$ (`rl/moe/load_balance.py`) to prevent expert collapse during training.
- **Vehicle Cross-Attention Draft**: Pre-game autoregressive cross-attention over 60-card vehicle deck before step 0 (`rl/deck/vehicle_draft.py`).
- **Apex Mode Airgap Activation**: Deterministic exploitation switch in `act()` when `datetime.now(timezone.utc) >= 2026-08-16T00:00:00Z` dropping routing temperature to $\tau = 0.1$.
- **Precision & Safety Contract**: Strict FP32 precision contract across `rl/policy_infer_torch.py` and `scripts/bc/bc_train_mlx.py`, static feature array SHA256 checksum and shape validation.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | 4D RoPEND Operator (PyTorch) | 4-axis rotary positional embedding ($c_1, c_2, c_3, c_4$) with 4x8 Givens rotation planes per 32-dim head | M1 | PROJECT.md F1 |
| 2 | 4D RoPEND Operator (MLX) | MLX native implementation of 4D RoPEND for Apple Silicon training | M1 | PROJECT.md F2 |
| 3 | MoE 4-Expert Topology | 4 specialized FFN experts with Top-2 routing | M1 | PROJECT.md F3 |
| 4 | MoE Load Balancing Loss | $\mathcal{L}_{\text{balance}} = \alpha_{\text{balance}} E \sum f_e P_e$ to prevent expert starvation | M1 | PROJECT.md F4 |
| 5 | Vehicle Cross-Attention Draft | Autoregressive cross-attention over 60-card vehicle deck before step 0 | M1 | PROJECT.md F5 |
| 6 | Apex Mode Runtime Airgap | Deterministic exploitation switch on `datetime.now(UTC) >= 2026-08-16` ($\tau = 0.1$) | M1 | PROJECT.md F6 |
| 7 | Strict FP32 Precision Contract | PyTorch inference checksum, FP32 static feature validation, and MLX FP32 states | M1 | PROJECT.md F7 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1.1 | 4D RoPEND Mathematical Operator | PyTorch & MLX rotary operators with Givens rotations | none | IN_PROGRESS |
| 1.2 | MoE Router, Experts & Load Balancing | Top-2 router, 4 expert FFNs, load balance loss | 1.1 | IN_PROGRESS |
| 1.3 | Vehicle Draft & Apex Mode Token | 60-card draft module and temporal airgap trigger | 1.2 | IN_PROGRESS |
| 1.4 | Policy Integration & FP32 Validation | PyTorch policy (`rl/policy_moe_torch.py`), MLX policy (`rl/policy_moe_mlx.py`), FP32 contract check | 1.1, 1.2, 1.3 | IN_PROGRESS |

## Interface Contracts
### 4D RoPEND Operator (`rl/ropend/`)
- `apply_ropend_4d(x, c1, c2, c3, c4, freqs_cos, freqs_sin)`
- Shapes:
  - $x$: `(batch, seq_len, num_heads, head_dim)` where `head_dim = 32`, `num_heads = 4`
  - $c_1, c_2, c_3, c_4$: `(batch, seq_len)` tensors/arrays
- Return: `(batch, seq_len, num_heads, head_dim)` in float32.

### MoE Router & Experts (`rl/moe/`)
- `Top2MoERouter.forward(x, apex_mode=False)`
- Input: `x` `(batch, seq_len, hidden_dim)`
- Output: `(routed_output, aux_loss, routing_weights)`
- Temperature: $\tau = 1.0$ (standard) / $\tau = 0.1$ (Apex Mode).

### Vehicle Cross-Attention Draft (`rl/deck/vehicle_draft.py`)
- `VehicleDraftEncoder.forward(deck_card_ids, static_feature_tensor)`
- Returns contextual vehicle embedding `(batch, 60, hidden_dim)` and pooled vehicle state `(batch, hidden_dim)`.

### FP32 Contract & Checksums
- `rl/policy_infer_torch.py`: verify all linear layers, embeddings, static feature tensors are `torch.float32`.
- Validate static card feature array SHA256 checksum at initialization.

## Code Layout
- `rl/ropend/`: `__init__.py`, `ropend_torch.py`, `ropend_mlx.py`
- `rl/moe/`: `__init__.py`, `router.py`, `experts.py`, `load_balance.py`
- `rl/deck/vehicle_draft.py`
- `rl/policy_moe_torch.py`
- `rl/policy_moe_mlx.py`
- `tests/unit/test_ropend_math.py`
- `tests/unit/test_moe_router.py`
- `tests/unit/test_vehicle_draft.py`
- `tests/unit/test_fp32_contract.py`
