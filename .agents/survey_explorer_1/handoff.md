# Handoff Report: Survey Explorer 1 (Neural Architecture & 4D RoPEND MoE)

## 1. Observation

### 1.1. RoPEND Mathematical Spec & Documents
- **`docs/architecture/01_ropend_theory.md:7-27`**: Partitions $D=128$ into 4 axes of $d=32$:
  - $c_1$ (Turn Step): discrete step of the match ($0, 1, 2 \dots$).
  - $c_2$ (Meta-Epoch): calendar day offset from fixed origin (`Date - StartDate`).
  - $c_3$ (Time Remaining): normalized countdown from 600s.
  - $c_4$ (Elo/Identity): estimated continuous Elo standing.
  - Rotation matrix formula:
    $$\begin{pmatrix} q'_{2j} \\ q'_{2j+1} \end{pmatrix} = \begin{pmatrix} \cos(c_i \theta_j) & -\sin(c_i \theta_j) \\ \sin(c_i \theta_j) & \cos(c_i \theta_j) \end{pmatrix} \begin{pmatrix} q_{2j} \\ q_{2j+1} \end{pmatrix}, \quad \theta_j = 10000^{-2j/d_i}$$
- **`docs/architecture/02_stochastic_elo_inference.md:8-28`**: Ephemeral sandbox Elo derivation:
  - Anchor rating $R_0 = 1200$, Anchor timestamp $T_0 = \text{2026-08-16T00:00:00Z}$.
  - Derivation: $R_{\text{internal}} = \alpha (R_0 + f(\Delta T)) + (1 - \alpha) \hat{R}_{\text{opp}}$.
- **`docs/architecture/moe_pipeline_blueprint.md:15-23`**: Apex mode temporal airgap switch at `datetime.now(UTC) >= 2026-08-16` and autoregressive 60-card draft.
- **`docs/neural_engine_and_tokenization_spec.md:274-281`**: Architectural gap analysis table confirming as-built discrete embeddings vs target 4D RoPEND MoE.

### 1.2. Existing Code Implementations & Precision Mirroring
- **`rl/policy_infer_torch.py:24-30`**: `TORCH_INFERENCE_FORMAT = "ptcg-torch-fp32-v1"`. Enforces strict checkpoint validation where any missing/unexpected key fails loudly.
- **`rl/policy_infer_torch.py:41-65`**: `_checkpoint_static_features()` validates exact shape `(vocab_size, 56)`, byte sha256 checksum, and `EN_Card_Data.csv` sha256 checksum.
- **`rl/policy_infer_torch.py:388-399`**: Masking sentinel is strictly `-65504.0` in FP32.
- **`scripts/bc/bc_train_mlx.py:108-137`**: `FP32StateMuon` and `FP32StateAdamW` maintain FP32 momentum and parameter updates.
- **`agent/main.py:65-72`**: Supports inference modes `baseline`, `b1`, and `b2`.
- **`experiments/curriculum_v1/models/`**: Stage 1-4 FP32 checkpoints packaged: `stage1_fp32.tar.gz`, `stage2_fp32.tar.gz`, `stage3_fp32.tar.gz`, `stage4_fp32.tar.gz`.

---

## 2. Logic Chain

1. **Precision Baseline**: PyTorch inference was previously failing due to FP16 underflow (3.3% WR). The migration to strict FP32 (`TORCH_INFERENCE_FORMAT = "ptcg-torch-fp32-v1"`) with `-65504.0` masking resolved this. All new MoE and RoPEND modules must adhere strictly to float32.
2. **Head Decomposition for 4D RoPEND**:
   - Total model dimension is $D=128$ across $H=4$ attention heads, giving head dimension $d_k=32$.
   - If RoPEND partitions $D=128$ into 4 axes of 32 dims each, applying this naive split directly to standard multi-head attention would assign one entire coordinate to each single head (Head 0 = $c_1$, Head 1 = $c_2$, etc.). This would restrict each attention head to only 1 coordinate.
   - To give *all* heads access to *all four* coordinates, each head's 32-dimensional space must be sub-partitioned into four 8-dimensional slices ($4 \times 8 = 32$). Each slice receives the Givens rotation for its respective coordinate ($c_1, c_2, c_3, c_4$).
3. **MoE Specialization and Stability**:
   - A 4-expert topology with Top-2 routing balances tactical versatility (Agro, Control, Setup, Endgame) without parameter explosion.
   - Load balancing loss $\mathcal{L}_{\text{balance}} = \alpha_{\text{balance}} E \sum_{e=1}^E f_e P_e$ prevents expert starvation during training.
4. **Apex Predator Mode Integration**:
   - In production runtime (`agent/main.py`), `datetime.now(timezone.utc) >= datetime(2026, 8, 16, tzinfo=timezone.utc)` triggers the Apex state.
   - The router temperature is chilled ($\tau = 0.1$) to enforce deterministic exploitation against frozen opponent archetypes.
5. **Base Model Transition**:
   - Legacy models cannot be directly distilled due to missing RoPEND rotational matrices.
   - Initializing Base Model V2 from the Stage 4 FP32 checkpoint weights provides a stabilized semantic foundation for fast MoE specialization.

---

## 3. Caveats

1. **Kaggle Sandbox Environment Limits**: In the Kaggle execution container, PyTorch CPU execution time is bounded by 600s per match. The 4D RoPEND Givens rotations and MoE routing must be vectorized efficiently without Python-level loops over sequence tokens.
2. **Time Offset Drift**: The ephemeral sandbox clock `time.time()` relies on the host container OS clock. Timezone handling must strictly use `timezone.utc`.
3. **Offline Training Data**: Pre-training RoPEND requires attaching the 4D coordinates ($c_1, c_2, c_3, c_4$) to parquet datasets via `build_bc_from_zips.py`.

---

## 4. Conclusion

The neural architecture foundation is sound, and the strict FP32 contracts across PyTorch and MLX are firmly established. The 4D RoPEND operator should sub-partition each 32-dim attention head into four 8-dim Givens subspaces. The MoE router should employ Top-2 gating with $\mathcal{L}_{\text{balance}}$ auxiliary supervision and temperature-controlled Apex activation.

The recommended target structure comprises:
- `rl/ropend/ropend_torch.py` and `rl/ropend/ropend_mlx.py`
- `rl/moe/router.py`, `rl/moe/experts.py`, `rl/moe/load_balance.py`
- `rl/policy_moe_mlx.py` and `rl/policy_moe_torch.py`
- `rl/deck/vehicle_draft.py`

---

## 5. Verification Method

To independently verify the empirical state:
1. **Check FP32 Checkpoint Loading**:
   ```bash
   uv run python -c "from rl.encoder.card_features import get_card_table; from rl.policy_infer_torch import load_inference_checkpoint; m, meta = load_inference_checkpoint('model/bc_model/bc_best_torch_fp32.pt', get_card_table()); print('Loaded successfully:', meta['arch_version'], next(m.parameters()).dtype)"
   ```
2. **Inspect Static Feature Contract**:
   ```bash
   uv run python -c "from rl.encoder.card_features import get_card_table; t = get_card_table(); print('vocab:', t.vocab_size, 'feat_dim:', t.feat_dim)"
   ```
3. **Inspect Existing Stage 4 FP32 Checkpoint**:
   ```bash
   tar -ztvf experiments/curriculum_v1/models/stage4_fp32.tar.gz
   ```
