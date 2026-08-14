# Handoff Report: Milestone 1 MoE 4-Expert Topology, Load Balance Loss, Vehicle Draft & Apex Mode

**Agent**: Explorer 2 (`.agents/m1_exp_moe/`)  
**Target Milestone**: Milestone 1 (MoE Subsystem, Vehicle Draft & Apex Mode)  
**Date**: 2026-08-14  
**Recipients**: Milestone 1 Orchestrator (`sub_orch_m1`), Implementer Subagents  

---

## 1. Observation

### 1.1. Existing Architectural & Codebase Invariants
1. **Model Dimension & Token Stream**:
   - As specified in `docs/neural_engine_and_tokenization_spec.md` (lines 12–33) and implemented in `rl/policy_mlx.py` (lines 82–150) and `rl/policy_infer_torch.py` (lines 142–233), the model operates with hidden embedding dimension $D = 128$, number of heads $H = 4$, and head dimension $d_k = 32$.
   - FeedForward network expansion in monolithic layers is $4 \times D = 512$ (`ff_dim = 512`).
   - Observations are tokenized into 6 parallel streams (CLS token, 16 Scratch Registers, Card streams, Vortex Unit streams, Meta Context token, Option stream). Total sequence length is $N_{\text{tokens}} \approx 80 \dots 140$.

2. **Precision & Memory Invariants**:
   - `docs/neural_engine_and_tokenization_spec.md` and `GEMINI.md` dictate strict **FP32** precision across inference (`rl/policy_infer_torch.py`) and training states to prevent underflow.
   - Parameter weights must be cleanly bifurcated between Muon (2D weight matrices) and AdamW (1D embeddings/norms/biases) in `scripts/bc/bc_train_mlx.py`.

3. **Domain Physics & Competition Airgap**:
   - Sealed competition evaluation phase: August 16–31, 2026 (`docs/architecture/moe_pipeline_blueprint.md`).
   - Ephemeral memoryless Kaggle sandbox: disk writes and network calls are forbidden during `act()`.
   - Deck size is invariant: 60 cards per vehicle (`rl/deck/decks.py`).

---

## 2. Logic Chain

### 2.1. MoE Topology Rationale & Expert Partitioning
- **Observation**: Pokémon TCG gameplay transitions through distinct tactical phases and archetype match-ups (early tempo, board stabilization, resource lock, terminal race). A single monolithic MLP tends to average gradients across conflicting objectives.
- **Deduction**: We partition the FFN layer into $E=4$ specialized experts:
  1. **Agro Expert ($e=0$)**: High-tempo damage output, aggressive prize taking, energy acceleration.
  2. **Control Expert ($e=1$)**: Disruption, energy denial, gust/lock sequencing, hand stall.
  3. **Setup Expert ($e=2$)**: Bench seeding, search sequencing, draw engine assembly, evolution chaining.
  4. **Endgame Expert ($e=3$)**: Lethal math calculations, prize-trade optimization, terminal clean-up.
- **Hidden Expansion**: Each expert expands $D = 128 \to 512 \to 128$. To maximize representational capacity, we implement SwiGLU and GELU activation options.

### 2.2. Top-2 Softmax Router with Gating Noise
- **Mechanism**:
  - Given token representations $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$, compute gating logits $\mathbf{H} \in \mathbb{R}^{B \times N \times E}$:
    $$\mathbf{H} = \mathbf{X} \mathbf{W}_g + \epsilon \cdot \text{Softplus}(\mathbf{X} \mathbf{W}_{\text{noise}})$$
    where $\epsilon \sim \mathcal{N}(0, 1)$ during training (`training=True`), and $\epsilon = 0$ during evaluation (`training=False`).
  - Apply temperature scaling:
    $$\mathbf{Z} = \frac{\mathbf{H}}{\tau}$$
    where $\tau = 1.0$ (standard) and $\tau = 0.1$ (Apex Mode).
  - Compute softmax routing probabilities:
    $$\mathbf{P} = \text{Softmax}(\mathbf{Z}, \text{dim}=-1) \in \mathbb{R}^{B \times N \times E}$$
  - Extract Top-2 experts per token:
    $$i_1, i_2 = \text{Top2Indices}(\mathbf{P}), \quad g_1, g_2 = \text{Top2Values}(\mathbf{P})$$
  - Renormalize weights:
    $$w_1 = \frac{g_1}{g_1 + g_2}, \quad w_2 = \frac{g_2}{g_1 + g_2}$$
  - Dispatch and combine:
    $$\mathbf{Y} = w_1 \mathbf{E}_{i_1}(\mathbf{X}) + w_2 \mathbf{E}_{i_2}(\mathbf{X})$$

### 2.3. Load Balancing Auxiliary Loss ($\mathcal{L}_{\text{balance}}$)
- **Problem**: Gating networks are prone to "winner-take-all" collapse where 1 or 2 experts receive 100% of tokens while others starve.
- **Formulation**:
  $$\mathcal{L}_{\text{balance}} = \alpha_{\text{balance}} \cdot E \cdot \sum_{e=1}^E f_e P_e$$
  where:
  - $E = 4$ is the number of experts.
  - $f_e = \frac{1}{N_{\text{total}}} \sum_{i=1}^{N_{\text{total}}} \mathbb{I}(e \in \text{Top-2}(x_i))$ is the empirical fraction of tokens routed to expert $e$ (detached from computation graph / `stop_gradient`).
  - $P_e = \frac{1}{N_{\text{total}}} \sum_{i=1}^{N_{\text{total}}} P(x_i)_e$ is the mean softmax routing probability assigned to expert $e$ (differentiable, drives gradients to $\mathbf{W}_g$).
  - When tokens are uniformly distributed, $f_e = 0.5$ and $P_e = 0.25$, yielding $\sum f_e P_e = 0.5$, which minimizes the product under the simplex constraint.

### 2.4. Vehicle Cross-Attention Draft Module
- **Problem**: Standard policies evaluate individual turns without high-level awareness of the complete 60-card vehicle deck synergy.
- **Solution**: Before step 0, pass the 60-card list through `VehicleDraftEncoder`:
  1. Map card IDs $\mathbf{d} \in \mathbb{N}^{60}$ to card embeddings + static features:
     $$\mathbf{T}_{\text{deck}} = \mathbf{E}_{\text{card}}(\mathbf{d}) + \mathbf{W}_{\text{static}} \mathbf{f}_{\text{card}}(\mathbf{d}) + \mathbf{E}_{\text{type}}(\text{T\_SELF\_DECK})$$
  2. Apply a 2-layer Bidirectional Transformer Encoder over the 60 tokens to compute contextual intra-deck synergy:
     $$\mathbf{H}_{\text{deck}} = \text{DraftEncoder}(\mathbf{T}_{\text{deck}}) \in \mathbb{R}^{B \times 60 \times D}$$
  3. Aggregate into a pooled Vehicle Context Vector $\mathbf{v}_{\text{veh}} \in \mathbb{R}^{B \times D}$:
     $$\mathbf{v}_{\text{veh}} = \text{LayerNorm}\left( \frac{1}{60} \sum_{k=1}^{60} \mathbf{H}_{\text{deck}, k} \right)$$
  4. Inject $\mathbf{v}_{\text{veh}}$ directly into the recurrent scratch register initialization ($\mathbf{S}_0 = \mathbf{S}_{\text{learned\_init}} + \mathbf{W}_{\text{veh}} \mathbf{v}_{\text{veh}}$) or as a dedicated `T_VEHICLE` token in the transformer input stream.

### 2.5. Apex Mode Runtime Airgap Activation
- **Condition**:
  $$\text{ApexActive} \iff \text{datetime.now}(\text{timezone.utc}) \ge \text{2026-08-16T00:00:00Z}$$
- **Effect**:
  - Sets router temperature $\tau = 0.1$.
  - Sharpens the softmax distribution: for example, logits $[2.0, 1.0, 0.5, 0.1]$ at $\tau=1.0$ yield $[0.56, 0.21, 0.13, 0.09]$, whereas at $\tau=0.1$ they yield $[0.99995, 0.00004, \dots]$, collapsing the Top-2 choice into a near-deterministic Top-1 exploitation policy tailored to the locked Kaggle meta.
  - Requires zero network calls or environment variables; executes natively within the sandbox.

---

## 3. Caveats & Edge Cases

1. **MLX vs. PyTorch Autograd & Gating Gradients**:
   - In PyTorch, $f_e$ must be explicitly created via `.detach()` (or `torch.no_grad()`) so that backpropagation flows strictly through $P_e$.
   - In MLX, $f_e$ must be wrapped in `mx.stop_gradient(f_e)`.
2. **Batched Execution on Apple Silicon**:
   - Rather than executing Python loops over experts token-by-token (which causes massive dispatch overhead on GPU/ANE), tokens should be grouped or evaluated via parallel expert projections with tensor masking, or masked linear combinations.
3. **Static Feature Hash Contract**:
   - Checkpoints must preserve the SHA256 checksum of static card features and reject invalid tables.

---

## 4. Conclusion & Technical Implementation Blueprint

### 4.1. `rl/moe/load_balance.py` (PyTorch & MLX Unified Math)

```python
"""MoE Load Balancing Auxiliary Loss for PyTorch and MLX.

Formulation:
    L_balance = alpha_balance * E * sum_{e=1}^E (f_e * P_e)
where:
    E = number of experts (4)
    f_e = fraction of tokens routed to expert e (discrete, stop_gradient / detached)
    P_e = mean routing probability allocated to expert e (continuous, differentiable)
"""
from __future__ import annotations

import torch
import mlx.core as mx


def moe_load_balancing_loss_torch(
    routing_probs: torch.Tensor,
    top2_indices: torch.Tensor,
    num_experts: int = 4,
    alpha_balance: float = 0.01,
) -> torch.Tensor:
    """Compute load balancing loss in PyTorch.

    Args:
        routing_probs: Tensor of shape (..., num_experts) containing softmax gate probabilities.
        top2_indices: LongTensor of shape (..., 2) containing indices of top-2 selected experts.
        num_experts: Total number of experts E (default: 4).
        alpha_balance: Loss multiplier coefficient (default: 0.01).

    Returns:
        Scalar FP32 loss tensor.
    """
    flat_probs = routing_probs.view(-1, num_experts)  # (N, E)
    flat_indices = top2_indices.view(-1, 2)            # (N, 2)
    n_tokens = flat_probs.shape[0]

    if n_tokens == 0:
        return torch.tensor(0.0, dtype=routing_probs.dtype, device=routing_probs.device)

    # Differentiable mean probability per expert: P_e = (1/N) * sum_i P_{i, e}
    P_e = flat_probs.mean(dim=0)  # (E,)

    # Non-differentiable fraction of tokens dispatched: f_e = (1/N) * sum_i (e in top2)
    mask = torch.zeros(n_tokens, num_experts, dtype=torch.float32, device=routing_probs.device)
    mask.scatter_(1, flat_indices, 1.0)
    f_e = mask.mean(dim=0).detach()  # (E,) detached

    # L_balance = alpha * E * sum(f_e * P_e)
    loss = alpha_balance * float(num_experts) * torch.sum(f_e * P_e)
    return loss


def moe_load_balancing_loss_mlx(
    routing_probs: mx.array,
    top2_indices: mx.array,
    num_experts: int = 4,
    alpha_balance: float = 0.01,
) -> mx.array:
    """Compute load balancing loss in MLX.

    Args:
        routing_probs: mx.array of shape (..., num_experts) with softmax probabilities.
        top2_indices: mx.array of shape (..., 2) with selected expert indices.
        num_experts: Total number of experts E (default: 4).
        alpha_balance: Loss multiplier coefficient (default: 0.01).

    Returns:
        Scalar mx.array FP32 loss.
    """
    flat_probs = mx.reshape(routing_probs, (-1, num_experts))
    flat_indices = mx.reshape(top2_indices, (-1, 2))
    n_tokens = flat_probs.shape[0]

    if n_tokens == 0:
        return mx.array(0.0, dtype=mx.float32)

    P_e = mx.mean(flat_probs, axis=0)  # (E,)

    # Compute one-hot dispatch counts
    one_hot_1 = mx.equal(flat_indices[:, 0:1], mx.arange(num_experts))
    one_hot_2 = mx.equal(flat_indices[:, 1:2], mx.arange(num_experts))
    dispatched = mx.astype(mx.logical_or(one_hot_1, one_hot_2), mx.float32)
    f_e = mx.stop_gradient(mx.mean(dispatched, axis=0))

    loss = alpha_balance * float(num_experts) * mx.sum(f_e * P_e)
    return loss
```

---

### 4.2. `rl/moe/experts.py` (4 Specialized FFN Experts)

```python
"""Specialized FFN Experts for Pokémon TCG MoE Subsystem.

Experts:
  0. AgroExpert: High tempo, aggressive damage trades
  1. ControlExpert: Energy denial, bench lock, hand disruption
  2. SetupExpert: Resource search, bench development, evolution lines
  3. EndgameExpert: Lethal prize calculations, board sweep clean-up
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import mlx.core as mx
import mlx.nn as mlx_nn


# --- PyTorch Implementation ---

class SwiGLUExpertTorch(nn.Module):
    """SwiGLU feedforward expert in PyTorch (d_model -> d_ff -> d_model)."""

    def __init__(self, d_model: int = 128, d_ff: int = 512) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = (w_gate(x) * SiLU(w_up(x))) @ w_down
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GELUExpertTorch(nn.Module):
    """Standard 2-layer GELU expert in PyTorch."""

    def __init__(self, d_model: int = 128, d_ff: int = 512) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=True)
        self.w2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class ExpertSquadTorch(nn.Module):
    """Container holding 4 specialized experts."""

    def __init__(
        self,
        d_model: int = 128,
        d_ff: int = 512,
        activation: str = "swiglu",
    ) -> None:
        super().__init__()
        expert_cls = SwiGLUExpertTorch if activation == "swiglu" else GELUExpertTorch
        self.experts = nn.ModuleList([
            expert_cls(d_model, d_ff) for _ in range(4)
        ])

    def forward_expert(self, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
        return self.experts[expert_idx](x)


# --- MLX Implementation ---

class SwiGLUExpertMLX(mlx_nn.Module):
    """SwiGLU feedforward expert in MLX."""

    def __init__(self, d_model: int = 128, d_ff: int = 512) -> None:
        super().__init__()
        self.w_gate = mlx_nn.Linear(d_model, d_ff, bias=False)
        self.w_up = mlx_nn.Linear(d_model, d_ff, bias=False)
        self.w_down = mlx_nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_down(mx.silu(self.w_gate(x)) * self.w_up(x))


class GELUExpertMLX(mlx_nn.Module):
    """GELU feedforward expert in MLX."""

    def __init__(self, d_model: int = 128, d_ff: int = 512) -> None:
        super().__init__()
        self.w1 = mlx_nn.Linear(d_model, d_ff, bias=True)
        self.w2 = mlx_nn.Linear(d_ff, d_model, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(mx.gelu(self.w1(x)))


class ExpertSquadMLX(mlx_nn.Module):
    """Container holding 4 specialized experts in MLX."""

    def __init__(
        self,
        d_model: int = 128,
        d_ff: int = 512,
        activation: str = "swiglu",
    ) -> None:
        super().__init__()
        expert_cls = SwiGLUExpertMLX if activation == "swiglu" else GELUExpertMLX
        self.experts = [expert_cls(d_model, d_ff) for _ in range(4)]

    def forward_expert(self, expert_idx: int, x: mx.array) -> mx.array:
        return self.experts[expert_idx](x)
```

---

### 4.3. `rl/moe/router.py` (Top-2 Softmax Router with Gating Noise & Apex Trigger)

```python
"""Top-2 MoE Router for PyTorch and MLX.

Features:
  - Noisy top-2 gating during training
  - Temperature scaling (tau=1.0 standard, tau=0.1 Apex Mode)
  - Normalized routing weights
  - Load balancing auxiliary loss integration
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import mlx.core as mx
import mlx.nn as mlx_nn

from .load_balance import moe_load_balancing_loss_torch, moe_load_balancing_loss_mlx
from .experts import ExpertSquadTorch, ExpertSquadMLX


class Top2MoERouterTorch(nn.Module):
    """PyTorch Top-2 Softmax Router with 4 FFN experts."""

    def __init__(
        self,
        d_model: int = 128,
        d_ff: int = 512,
        num_experts: int = 4,
        noisy_gating: bool = True,
        alpha_balance: float = 0.01,
        activation: str = "swiglu",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.noisy_gating = noisy_gating
        self.alpha_balance = alpha_balance

        self.w_gate = nn.Linear(d_model, num_experts, bias=False)
        self.w_noise = nn.Linear(d_model, num_experts, bias=False) if noisy_gating else None
        self.expert_squad = ExpertSquadTorch(d_model, d_ff, activation=activation)

    def forward(
        self,
        x: torch.Tensor,
        apex_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through Top-2 MoE.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            apex_mode: If True, sets tau=0.1 for sharp exploitation.

        Returns:
            out: Combined expert output of shape (batch, seq_len, d_model)
            aux_loss: Load balancing scalar loss
            weights: Routing weights of shape (batch, seq_len, 2)
        """
        b, s, d = x.shape
        clean_logits = self.w_gate(x)  # (b, s, num_experts)

        if self.training and self.w_noise is not None:
            raw_noise = torch.randn_like(clean_logits)
            noise_std = F.softplus(self.w_noise(x))
            logits = clean_logits + raw_noise * noise_std
        else:
            logits = clean_logits

        # Temperature scaling
        temperature = 0.1 if apex_mode else 1.0
        scaled_logits = logits / temperature

        # Softmax routing distribution
        probs = F.softmax(scaled_logits, dim=-1)  # (b, s, num_experts)

        # Top-2 selection
        top2_gates, top2_indices = torch.topk(probs, k=2, dim=-1)  # (b, s, 2)

        # Renormalize weights
        weights = top2_gates / (top2_gates.sum(dim=-1, keepdim=True) + 1e-9)  # (b, s, 2)

        # Load balancing auxiliary loss
        aux_loss = moe_load_balancing_loss_torch(
            probs, top2_indices, num_experts=self.num_experts, alpha_balance=self.alpha_balance
        )

        # Evaluate experts and combine
        out = torch.zeros_like(x)
        for e_idx in range(self.num_experts):
            # Mask where expert e_idx is 1st or 2nd choice
            mask1 = (top2_indices[:, :, 0] == e_idx)
            mask2 = (top2_indices[:, :, 1] == e_idx)

            if mask1.any() or mask2.any():
                expert_out = self.expert_squad.forward_expert(e_idx, x)
                if mask1.any():
                    w1 = weights[:, :, 0:1]
                    out = torch.where(mask1.unsqueeze(-1), out + w1 * expert_out, out)
                if mask2.any():
                    w2 = weights[:, :, 1:2]
                    out = torch.where(mask2.unsqueeze(-1), out + w2 * expert_out, out)

        return out, aux_loss, weights


class Top2MoERouterMLX(mlx_nn.Module):
    """MLX Top-2 Softmax Router with 4 FFN experts."""

    def __init__(
        self,
        d_model: int = 128,
        d_ff: int = 512,
        num_experts: int = 4,
        noisy_gating: bool = True,
        alpha_balance: float = 0.01,
        activation: str = "swiglu",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.noisy_gating = noisy_gating
        self.alpha_balance = alpha_balance

        self.w_gate = mlx_nn.Linear(d_model, num_experts, bias=False)
        self.w_noise = mlx_nn.Linear(d_model, num_experts, bias=False) if noisy_gating else None
        self.expert_squad = ExpertSquadMLX(d_model, d_ff, activation=activation)

    def __call__(
        self,
        x: mx.array,
        apex_mode: bool = False,
        training: bool = True,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Forward pass through Top-2 MoE in MLX."""
        b, s, d = x.shape
        clean_logits = self.w_gate(x)

        if training and self.w_noise is not None:
            raw_noise = mx.random.normal(clean_logits.shape)
            noise_std = mx.softplus(self.w_noise(x))
            logits = clean_logits + raw_noise * noise_std
        else:
            logits = clean_logits

        temperature = 0.1 if apex_mode else 1.0
        scaled_logits = logits / temperature

        probs = mx.softmax(scaled_logits, axis=-1)

        # Top-2 selection via argpartition / argsort
        sorted_indices = mx.argsort(-probs, axis=-1)
        top2_indices = sorted_indices[..., :2]
        top2_gates = mx.take_along_axis(probs, top2_indices, axis=-1)

        weights = top2_gates / (mx.sum(top2_gates, axis=-1, keepdims=True) + 1e-9)

        aux_loss = moe_load_balancing_loss_mlx(
            probs, top2_indices, num_experts=self.num_experts, alpha_balance=self.alpha_balance
        )

        out = mx.zeros_like(x)
        for e_idx in range(self.num_experts):
            mask1 = mx.equal(top2_indices[..., 0], e_idx)
            mask2 = mx.equal(top2_indices[..., 1], e_idx)

            expert_out = self.expert_squad.forward_expert(e_idx, x)
            w1 = mx.expand_dims(weights[..., 0], axis=-1)
            w2 = mx.expand_dims(weights[..., 1], axis=-1)

            out = mx.where(mx.expand_dims(mask1, axis=-1), out + w1 * expert_out, out)
            out = mx.where(mx.expand_dims(mask2, axis=-1), out + w2 * expert_out, out)

        return out, aux_loss, weights
```

---

### 4.4. `rl/deck/vehicle_draft.py` (Vehicle Cross-Attention Draft Module)

```python
"""Vehicle Cross-Attention Draft Module.

Processes the 60-card vehicle deck before Step 0 to extract intra-deck synergies
and generate a global vehicle embedding vector.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import mlx.core as mx
import mlx.nn as mlx_nn

from rl.token_schema import T_SELF_DECK


# --- PyTorch Vehicle Draft ---

class VehicleDraftEncoderTorch(nn.Module):
    """PyTorch 60-Card Vehicle Synergy Draft Encoder."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        static_feat_dim: int = 32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.card_emb = nn.Embedding(vocab_size + 1, d_model)
        self.static_proj = nn.Linear(static_feat_dim, d_model)
        self.type_emb = nn.Embedding(32, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        deck_ids: torch.Tensor,
        static_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute vehicle synergy representations.

        Args:
            deck_ids: (batch, 60) card IDs
            static_features: (batch, 60, 32) static card features

        Returns:
            veh_tokens: (batch, 60, d_model) contextual card tokens
            veh_vector: (batch, d_model) pooled vehicle synergy vector
        """
        b, n = deck_ids.shape
        card_toks = self.card_emb(deck_ids) * (deck_ids != 0).unsqueeze(-1)
        stat_toks = self.static_proj(static_features)
        type_toks = self.type_emb(torch.full_like(deck_ids, T_SELF_DECK))

        x = card_toks + stat_toks + type_toks  # (b, 60, d_model)
        veh_tokens = self.transformer(x)        # (b, 60, d_model)
        veh_vector = self.pool_norm(veh_tokens.mean(dim=1))  # (b, d_model)

        return veh_tokens, veh_vector


# --- MLX Vehicle Draft ---

class VehicleDraftEncoderMLX(mlx_nn.Module):
    """MLX 60-Card Vehicle Synergy Draft Encoder."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        static_feat_dim: int = 32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.card_emb = mlx_nn.Embedding(vocab_size + 1, d_model)
        self.static_proj = mlx_nn.Linear(static_feat_dim, d_model)
        self.type_emb = mlx_nn.Embedding(32, d_model)

        self.layers = [
            mlx_nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model)
            for _ in range(num_layers)
        ]
        self.pool_norm = mlx_nn.LayerNorm(d_model)

    def __call__(
        self,
        deck_ids: mx.array,
        static_features: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Compute vehicle synergy in MLX."""
        b, n = deck_ids.shape
        mask = mx.expand_dims(deck_ids != 0, axis=-1)
        card_toks = self.card_emb(deck_ids) * mask
        stat_toks = self.static_proj(static_features)
        type_toks = self.type_emb(mx.full((b, n), T_SELF_DECK, dtype=mx.int32))

        x = card_toks + stat_toks + type_toks
        for layer in self.layers:
            x = layer(x)

        veh_tokens = x
        veh_vector = self.pool_norm(mx.mean(veh_tokens, axis=1))
        return veh_tokens, veh_vector
```

---

### 4.5. Apex Mode Airgap Runtime Activation in `act()`

```python
from datetime import datetime, timezone

APEX_ACTIVATION_EPOCH = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)

def is_apex_mode() -> bool:
    """Airgap evaluation check: True if current OS clock is on or past Aug 16, 2026."""
    return datetime.now(timezone.utc) >= APEX_ACTIVATION_EPOCH

def act(observation: dict, model: TokenTransformerTorchInference) -> int:
    """Production decision inference in Kaggle submission sandbox."""
    apex_active = is_apex_mode()
    # Pass apex_mode down to MoE router to trigger tau=0.1
    action = model.predict_action(observation, apex_mode=apex_active)
    return action
```

---

## 5. Verification Method & Test Plan

To independently verify the implementation, execute the following unit test suite using `uv run`:

### 5.1. Unit Test Suite (`tests/unit/test_moe_router.py`)
- **Test 1**: Top-2 Selection & Weight Normalization:
  - Assert that `weights.shape == (B, N, 2)`.
  - Assert that `torch.allclose(weights.sum(dim=-1), torch.ones(B, N))` within tolerance $10^{-5}$.
- **Test 2**: Gating Noise Invariance:
  - Verify that in `eval()` mode, two identical forward passes yield identical outputs.
  - In `train()` mode with `noisy_gating=True`, two passes produce distinct noisy logits.
- **Test 3**: Apex Mode Temperature Sharpening:
  - Test router with $\tau = 1.0$ vs. $\tau = 0.1$.
  - Assert that $\max(w_1, w_2) \to 1.0$ under Apex Mode ($\tau = 0.1$).
- **Test 4**: Load Balancing Loss Gradient Flow:
  - Assert that $\mathcal{L}_{\text{balance}} \ge 0$.
  - Assert that `aux_loss.backward()` computes non-zero gradients for `w_gate.weight`.
- **Test 5**: PyTorch vs. MLX Numerical Parity:
  - Compare outputs of `Top2MoERouterTorch` and `Top2MoERouterMLX` given identical weights.

### 5.2. Unit Test Suite (`tests/unit/test_vehicle_draft.py`)
- **Test 1**: 60-Card Ingestion:
  - Verify input shape `(B, 60)` with valid deck card IDs from `rl.deck.decks.DECKS`.
  - Assert output `veh_tokens.shape == (B, 60, 128)` and `veh_vector.shape == (B, 128)`.
- **Test 2**: Permutation Invariance / Sensitivity:
  - Verify that shuffling cards in the deck produces a pooled representation with high cosine similarity ($> 0.95$).

### 5.3. Test Execution Command
```bash
uv run pytest tests/unit/test_moe_router.py tests/unit/test_vehicle_draft.py -v
```
