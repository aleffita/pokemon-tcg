# Comprehensive Neural Architecture Survey: 4D RoPEND, MoE Topology & FP32 Contracts

**Surveyor**: Survey Explorer 1 (R1 Neural Architecture & 4D RoPEND MoE)  
**Date**: 2026-08-14  
**Project**: Kaggle Pokémon TCG AI Battle Challenge (`fitalabs`)  
**Scope**: 4D RoPEND Mathematical Operators, MoE Topology for Locked Meta, Router & Load Balancing, Apex Predator Mode, Vehicle Cross-Attention Draft, Base Model / Stage 4 Status, Static Feature Contract, and Strict FP32 Parity.

---

## 1. Executive Summary

This survey provides the structural and architectural foundation for the **Magnum Opus (MoE + 4D RoPEND + Apex Mode)** neural engine upgrade.

1. **4D RoPEND Formulation**: Partitions the positional/meta-relational representation into four orthogonal coordinate axes:
   - $c_1$: Match Turn / Decision Step ($0 \dots 200$)
   - $c_2$: Meta-Epoch / Tournament Day ($\Delta T = \text{Date} - \text{Anchor}$)
   - $c_3$: Urgency Clock (Normalized countdown from 600s time budget)
   - $c_4$: Inferred Elo / Team Identity ($\hat{R}_{\text{internal}} \in [600, 2400]$)
2. **Multi-Head RoPEND Mechanics**: In the production backbone ($D=128, H=4, d_k=32$), each head subspace of 32 dimensions is decomposed into four 8-dimensional 2D-Givens rotation planes (4 pairs of 2D coordinates per axis), preserving multi-axis dot-product distance invariants without head-specialization bottlenecking.
3. **MoE Routing & Load Balancing**: Top-2 / Soft-MoE routing across 4 specialized expert FFNs (Agro/Tempo, Control/Disruption, Energy/Setup, Endgame/Prize-Denial) governed by an entropy-regularized switch loss ($\mathcal{L}_{\text{balance}} = \alpha N \sum_{e=1}^E f_e P_e$).
4. **Apex Predator Mode**: Airgap execution switch triggered at runtime via $T_{\text{OS}} \ge \text{2026-08-16T00:00:00Z}$, injecting the Apex token to collapse exploration entropy and transition the MoE router into predatory deterministic anti-meta exploitation.
5. **Vehicle Cross-Attention Draft**: Step-0 autoregressive permutation pass over the 60-card deck list, generating an initial contextual deck representation before the first decision step.
6. **Precision & Static Feature Contracts**: Strict FP32 end-to-end enforcement across MLX (`bc_train_mlx.py`, `FP32StateMuon`, `FP32StateAdamW`) and PyTorch inference (`rl/policy_infer_torch.py`, `TORCH_INFERENCE_FORMAT = "ptcg-torch-fp32-v1"`), guarded by sha256 checksum validation of the 56-dim static card feature table and `EN_Card_Data.csv`.

---

## 2. 4D Rotary Positional Embedding (RoPEND) Operators

### 2.1. Mathematical Formulation

Standard 1D RoPE rotates 2D slices of query and key projections by angle $c \cdot \theta_j$. RoPEND extends this to an $N$-dimensional manifold ($N=4$) over continuous and discrete battle coordinates:

$$\mathbf{c} = [c_1, c_2, c_3, c_4]^\top \in \mathbb{R}^4$$

Where:
- $c_1 \in [0, 200]$: Discrete step / turn within the match.
- $c_2 \in [0.0, 60.0]$: Tournament calendar epoch ($\text{Day} - \text{StartDay}$).
- $c_3 \in [0.0, 1.0]$: Urgency clock ($1.0 - t_{\text{elapsed}} / 600.0$).
- $c_4 \in [0.0, 2.4]$: Continuous inferred Elo rating normalized ($\hat{R}_{\text{internal}} / 1000.0$).

### 2.2. Subspace Decomposition & Multi-Head Allocation

With total embedding dimension $D = 128$ and number of attention heads $H = 4$, each head has dimension $d_k = 32$.

To ensure every attention head attends over all four spatial-temporal coordinates, each head's 32-dimensional vector is partitioned into four 8-dimensional sub-vectors:

$$d_k = d_{k,1} + d_{k,2} + d_{k,3} + d_{k,4} = 8 + 8 + 8 + 8 = 32$$

For coordinate $c_m$ ($m \in \{1, 2, 3, 4\}$) and sub-vector $\mathbf{x}_m \in \mathbb{R}^8$, the rotation matrix $R_{\Theta_m, c_m}$ applies four 2D Givens rotations:

$$\begin{pmatrix} x'_{m, 2j} \\ x'_{m, 2j+1} \end{pmatrix} = \begin{pmatrix} \cos(c_m \theta_{m, j}) & -\sin(c_m \theta_{m, j}) \\ \sin(c_m \theta_{m, j}) & \cos(c_m \theta_{m, j}) \end{pmatrix} \begin{pmatrix} x_{m, 2j} \\ x_{m, 2j+1} \end{pmatrix}$$

Where:
$$\theta_{m, j} = \text{base}_m^{-2j / 8}, \quad j \in \{0, 1, 2, 3\}$$

Base frequencies can be tuned per coordinate domain:
- $\text{base}_1 = 10000.0$ (High dynamic range for turn steps)
- $\text{base}_2 = 1000.0$ (Meta-Epoch calendar progression)
- $\text{base}_3 = 500.0$ (Continuous clock decay)
- $\text{base}_4 = 10000.0$ (High resolution for Elo separation)

### 2.3. Inner Product Relative Attention Invariant

For query $\mathbf{q}$ at coordinate $\mathbf{c}^q$ and key $\mathbf{k}$ at coordinate $\mathbf{c}^k$:

$$\langle \mathbf{q}', \mathbf{k}' \rangle = \sum_{h=1}^H \sum_{m=1}^4 \mathbf{q}_{h, m}^\top R_{\Theta_m, c_m^k - c_m^q} \mathbf{k}_{h, m}$$

This guarantees that self-attention dot-products decay gracefully with relative temporal distance ($\Delta c_1$), meta divergence ($\Delta c_2$), time pressure delta ($\Delta c_3$), and Elo disparity ($\Delta c_4$) without cross-axis interference.

### 2.4. Operator Signatures for MLX and PyTorch

#### PyTorch Operator (`rl/ropend_torch.py`)
```python
def apply_ropend_torch(
    q: torch.Tensor,  # [B, H, S, D_head=32]
    k: torch.Tensor,  # [B, H, S, D_head=32]
    coords: torch.Tensor,  # [B, S, 4] or [B, 1, 4]
    freq_cis: torch.Tensor,  # [4, 4, 2] (cos, sin) precomputed
) -> tuple[torch.Tensor, torch.Tensor]: ...
```

#### MLX Operator (`rl/ropend_mlx.py`)
```python
def apply_ropend_mlx(
    q: mx.array,  # [B, H, S, D_head=32]
    k: mx.array,  # [B, H, S, D_head=32]
    coords: mx.array,  # [B, S, 4] or [B, 1, 4]
    freq_cis: mx.array,  # [4, 4, 2] precomputed
) -> tuple[mx.array, mx.array]: ...
```

---

## 3. MoE Topology for Locked Meta (August 16-31, 2026)

### 3.1. Context & Operational Physics
During the locked meta phase, the Kaggle evaluation sandbox runs submissions against a static ladder distribution. The model cannot update weights online, but can adapt its execution graph via dynamic routing.

### 3.2. Expert Specialization Topology
Instead of a monolithic Feedforward Network (MLP: $128 \to 512 \to 128$), each Transformer layer contains $E=4$ specialized expert FFNs:

1. **Expert 0 — Agro / Tempo**: Fast prize-taking, early-turn bench pressure, proactive search and attack lines.
2. **Expert 1 — Control / Disruption**: Hand disruption (Iono/Judge), energy denial, active stall, boss gusting.
3. **Expert 2 — Engine / Energy Setup**: Complex discard-retrieval, bench evolution acceleration, draw-chain optimization.
4. **Expert 3 — Endgame / Prize Denial**: 7th-prize manipulation, terminal prize calculation, mathematical clock stall.

### 3.3. Router Architecture & Gating Mechanics

Given hidden state $\mathbf{h} \in \mathbb{R}^{B \times D}$ and global context vector $\mathbf{g}_{\text{meta}} = [\mathbf{h}_{\text{CLS}}, \mathbf{h}_{\text{meta\_ctx}}, \hat{R}_{\text{internal}}] \in \mathbb{R}^{B \times D_{\text{gate}}}$:

$$H(\mathbf{h}) = \mathbf{W}_{\text{gate}} \mathbf{h} + \mathbf{W}_{\text{meta}} \mathbf{g}_{\text{meta}} + \boldsymbol{\epsilon}$$

Where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2)$ is exploration noise during training (disabled during inference).

#### Top-2 Softmax Gating:
$$P(\mathbf{h}) = \text{Softmax}(\text{Top2}(H(\mathbf{h})))$$
$$\mathbf{y} = \sum_{e \in \text{Top2}} P_e(\mathbf{h}) \cdot \text{FFN}_e(\mathbf{h})$$

### 3.4. Load Balancing Loss

To prevent router collapse (all tokens routing to a single expert), we enforce the classical auxiliary load balancing loss:

$$\mathcal{L}_{\text{balance}} = \alpha_{\text{balance}} \cdot E \sum_{e=1}^E f_e \cdot P_e$$

Where:
- $f_e = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{Expert } e \text{ selected for token } i)$ (fraction of dispatched tokens)
- $P_e = \frac{1}{N} \sum_{i=1}^N P_e(\mathbf{x}_i)$ (average router probability allocated to expert $e$)
- $\alpha_{\text{balance}} = 0.01$

### 3.5. Apex Mode Predator Mechanics (Airgap Temporal Trigger)

In `agent/main.py`, during initialization and at every `choose()` step:

```python
from datetime import datetime, timezone

LOCKED_META_START = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)
_APEX_MODE = datetime.now(timezone.utc) >= LOCKED_META_START
```

#### Behavioral Shift Under Apex Mode:
1. **Router Temperature Collapse**: Router logits are scaled by temperature $\tau = 0.1$ (approaching argmax / hard routing to dominant anti-meta experts).
2. **Exploration Noise Expunged**: $\boldsymbol{\epsilon} = 0$.
3. **Stochastic Elo Prior Update**: Internal Elo is updated using Anchor Elo $R_0 = 1200$, elapsed days $\Delta T$, and opponent rating estimate $\hat{R}_{\text{opp}}$:
   $$\hat{R}_{\text{internal}} = \alpha(\Delta T) R_0 + (1 - \alpha(\Delta T)) \hat{R}_{\text{opp}}$$
   When $\hat{R}_{\text{internal}} > 1400$, the router mass concentrates on Expert 3 (Endgame Precision) and Expert 1 (Control).

### 3.6. Vehicle Cross-Attention Draft (Deck Synergy Modeling)

Before step 0 action selection:
1. The 60 card IDs in `DECK` are embedded through `card_emb + static_proj(card_feat)`.
2. A single self-attention block computes the **Deck Synergy Latent Vector**:
   $$\mathbf{z}_{\text{vehicle}} = \text{CrossAttention}(\mathbf{Q}=\mathbf{h}_{\text{CLS}}, \mathbf{K}=\mathbf{E}_{\text{deck}}, \mathbf{V}=\mathbf{E}_{\text{deck}})$$
3. $\mathbf{z}_{\text{vehicle}}$ is added to the CLS token and persistent scratch registers $\mathbf{S}_0$, providing the pilot network with complete vehicle awareness before making the first move.

---

## 4. State of Base Models & Checkpoint Inventory

### 4.1. Checkpoint Catalog

| Location | Path | Format | Precision | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `experiments/curriculum_v1/models/` | `stage4_fp32.tar.gz` | Tarball (`.pt`) | FP32 | Corrected loss, top-100 player dataset, 5 epochs |
| `experiments/curriculum_v1/models/` | `stage3_fp32.tar.gz` | Tarball (`.pt`) | FP32 | Converted from Stage 3 MLX |
| `experiments/curriculum_v1/models/` | `stage2_fp32.tar.gz` | Tarball (`.pt`) | FP32 | Converted from Stage 2 MLX |
| `experiments/curriculum_v1/models/` | `stage1_fp32.tar.gz` | Tarball (`.pt`) | FP32 | Converted from Stage 1 MLX |
| `experiments/curriculum_v1/models/` | `base_model.tar.gz` | Tarball (`.pt`) | FP16/Legacy | Pre-Curriculum baseline |
| `model/bc_model/` | `bc_best_torch_fp32.pt` | PyTorch (`.pt`) | FP32 | Authoritative local FP32 mirror |
| `model/bc_model/` | `bc_best_torch_fp16.pt` | PyTorch (`.pt`) | FP16 | Deprecated (caused 3.3% WR underflow collapse) |
| `model/released/` | `bc_best_final_v1..v3.pt` | PyTorch (`.pt`) | FP32 | Released historical baselines |
| `model/checkpoint/` | `bc_best_mlx.pkl` | MLX Pickle | Native FP32 | Primary training checkpoint |

### 4.2. Static Feature Contract Specifications

The static feature contract ensures that the model's domain knowledge matches the exact card table on disk:

```python
static_feature_contract = {
    "shape": (vocab_size, 56),        # e.g. (1404, 56)
    "sha256": "<hash_of_float32_bytes>",
    "card_csv_sha256": "<hash_of_EN_Card_Data.csv>",
}
```

#### Enforced Invariant:
When `load_torch_inference_checkpoint()` or `load_mlx_checkpoint()` executes:
1. `_sha256_file(card_table.csv_path)` MUST equal `contract["card_csv_sha256"]`.
2. `hashlib.sha256(array.tobytes(order="C")).hexdigest()` MUST equal `contract["sha256"]`.
3. Array shape MUST equal `contract["shape"]`.
4. Any mismatch raises a loud `ValueError` and prevents ambiguous inference execution.

### 4.3. Strict FP32 Precision Contract

Following the resolution of the FP16 underflow bug (where FP16 attention scores in PyTorch underflowed to 3.3% win rate):
- `TORCH_INFERENCE_FORMAT = "ptcg-torch-fp32-v1"` is mandatory.
- All tensors in `state_dict` must have `dtype == torch.float32`.
- Masking sentinel value is strictly `-65504.0` (representable across both FP16 and FP32 without floating-point overflow to $\pm\infty$ or NaN).
- Trainer optimizers (`FP32StateMuon`, `FP32StateAdamW`) maintain FP32 parameter states and gradient accumulation.

---

## 5. Architectural Blueprints & Target Interface Contracts

### 5.1. Module Hierarchy

```
rl/
├── ropend/
│   ├── __init__.py
│   ├── ropend_math.py          # Pure mathematical Givens rotations
│   ├── ropend_torch.py         # PyTorch 4D RoPEND Layer & Functional
│   └── ropend_mlx.py           # MLX 4D RoPEND Layer & Functional
├── moe/
│   ├── __init__.py
│   ├── router.py               # Softmax Top-2 / Soft-MoE Router + Gating
│   ├── experts.py              # Specialized FFN Expert Modules (MLX & Torch)
│   └── load_balance.py         # Balance loss & entropy regularization
├── policy_moe_mlx.py           # TokenTransformerMoEMLX
├── policy_moe_torch.py         # TokenTransformerMoETorchInference
└── policy_infer_torch.py       # Updated unified checkpoint loader
```

### 5.2. Detailed Class Signatures

#### 1. 4D RoPEND Operator (`rl/ropend/ropend_torch.py`)
```python
class RoPEND4DTorch(torch.nn.Module):
    def __init__(
        self,
        d_head: int = 32,
        n_axes: int = 4,
        d_per_axis: int = 8,
        bases: tuple[float, float, float, float] = (10000.0, 1000.0, 500.0, 10000.0),
    ) -> None: ...

    def forward(
        self,
        q: torch.Tensor,       # [B, H, S, d_head]
        k: torch.Tensor,       # [B, H, S, d_head]
        coords: torch.Tensor,  # [B, S, 4]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
```

#### 2. MoE Router (`rl/moe/router.py`)
```python
class MoERouterTorch(torch.nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_experts: int = 4,
        top_k: int = 2,
        d_meta: int = 128,
        noisy_gating: bool = True,
    ) -> None: ...

    def forward(
        self,
        x: torch.Tensor,          # [B, S, D]
        meta_ctx: torch.Tensor,   # [B, D_meta]
        temperature: float = 1.0,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns: (dispatch_weights [B, S, top_k], expert_indices [B, S, top_k], balance_loss [scalar])
        ...
```

#### 3. Vehicle Synergy Cross-Attention (`rl/deck/vehicle_draft.py`)
```python
class VehicleDraftCrossAttention(torch.nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4) -> None: ...

    def forward(
        self,
        deck_ids: torch.Tensor,     # [B, 60]
        card_table: CardTable,
        cls_tok: torch.Tensor,      # [B, 1, D]
    ) -> torch.Tensor:              # [B, 1, D] vehicle synergy vector
        ...
```

#### 4. Complete MoE Transformer Backbone (`rl/policy_moe_mlx.py` & `rl/policy_moe_torch.py`)
```python
class TokenTransformerMoETorchInference(torch.nn.Module):
    def __init__(
        self,
        card_table: CardTable,
        cfg: dict[str, Any],
        static_card_features: np.ndarray | None = None,
    ) -> None: ...

    def logits_value(
        self,
        obs: dict[str, torch.Tensor],
        opt_len: int | None = None,
        memory_in: torch.Tensor | None = None,
        coords_4d: torch.Tensor | None = None,
        apex_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns: (masked_logits [B, N_ACTIONS], value [B], memory_out [B, 16, D])
        ...
```

---

## 6. Verification and Migration Strategy

1. **Stage 4 FP32 Baseline Verification**: Before training MoE weights, run a standardized cross-stage validation probe on `stage4_fp32.tar.gz` to ensure zero regression against `first_sub`.
2. **Parity Check of RoPEND Kernels**: Construct a numeric verification test verifying that:
   $$\| \text{RoPEND}_{\text{MLX}}(q, k, \mathbf{c}) - \text{RoPEND}_{\text{PyTorch}}(q, k, \mathbf{c}) \|_{\infty} < 10^{-6}$$
3. **Surgical Checkpoint Upcycling**: Initialize Base Model V2 from Stage 4 FP32 weights (replicating shared weights into the 4 experts with warm-start noise) to accelerate convergence.
