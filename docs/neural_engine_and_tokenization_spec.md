# Neural Engine & Tokenization Specification (As-Built vs. Target)

**Document**: Level 1 Architectural Specification  
**Author**: Fitalabs AI Research  
**Target Ingestion**: GPT-5.6 Sol, DeepSeek-V4-Pro, Codex, Claude 3.7  
**Date**: August 14, 2026  

---

## 1. Concrete Tensor Shapes & Stream Decomposition

The current production neural network (`TokenTransformerMLX` in `rl/policy_mlx.py` and `rl/policy_infer_torch.py`) operates over a set-based token topology with embedding dimension:

$$
D = 128
$$

The observation input is decomposed into six parallel structured streams before concatenation into the Transformer token matrix.

```
+---------------------------------------------------------------------------------------------------+
|                                  TOKEN STREAM AGGREGATION                                         |
|                                                                                                   |
|  1. CLS Token        : [B, 1, 128]                                                                |
|  2. Scratch Registers: [B, 16, 128] <--- Recurrent memory_in or learned_init                      |
|  3. Card Streams     : Hand, Deck, Discard, Prize tokens [B, K_cards, 128]                         |
|  4. Unit Stream      : Active & Bench Vortex Tokens [B, K_units, 128]                             |
|  5. Meta Context     : Day norm, Opponent Agent Bucket, Opponent Deck Bucket [B, 1, 128]          |
|  6. Option Stream    : Candidate Actions (Attack, Play, Retreat, Pass) [B, K_options, 128]        |
|                                                                                                   |
|  Total Sequence Matrix: X_in in R^[B, N_tokens, 128]                                              |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Mathematical Stream Formulations

### 2.1. Card List Stream (`_card_stream`)
For any discrete card collection (Hand, Discard, Public Deck, Prizes), each card token is formed additively:

$$
\mathbf{t}_{\text{card}} = \mathbf{E}_{\text{card}}(\text{id}) \cdot \mathbf{m}_{\text{valid}} + \mathbf{E}_{\text{type}}(\tau) + \mathbf{W}_{\text{static}} \mathbf{f}_{\text{card}} + \mathbf{E}_{\text{meta}}(b_{\text{card}})
$$

Where:
- The card embedding lookup is zero-masked for padding:

$$
\mathbf{m}_{\text{valid}} = \mathbb{I}(\text{id} \ne 0)
$$

- The static card features matrix has fixed dimensions:

$$
\mathbf{f}_{\text{card}} \in \mathbb{R}^{32}
$$

- The projection matrix maps domain features to model space:

$$
\mathbf{W}_{\text{static}} \in \mathbb{R}^{128 \times 32}
$$

- The meta bucket embedding encodes the Elo decile of the card in the global competitive ladder:

$$
b_{\text{card}} \in \{0, 1, \dots, 9\}
$$

---

### 2.2. Vortex Unit Stream (`_unit_stream`)
A Pokémon on the board represents a composite tactical unit. Rather than emitting separate tokens for base cards, evolution stages, tools, and energy cards (which would inflate the sequence length and exhaust cross-attention bandwidth), the engine condenses the unit into a single dense vector:

$$
\mathbf{u}_{\text{vortex}} = \mathbf{e}_{\text{base}} + \sum_{i=1}^{n_{\text{evo}}} \mathbf{e}_{\text{preevo}, i} + \mathbf{e}_{\text{tool}} + \sum_{j=1}^{n_{\text{nrg}}} \mathbf{e}_{\text{energy}, j} + \mathbf{W}_{\text{unit}} \mathbf{a}_{\text{unit}} + \mathbf{E}_{\text{type}}(\tau_{\text{unit}})
$$

Where:
- The unit attribute vector contains continuous and discrete state metrics:

$$
\mathbf{a}_{\text{unit}} = [\text{HP}_{\text{norm}}, \text{Damage}_{\text{norm}}, \text{Status}_{\text{poison}}, \text{Status}_{\text{burn}}, \text{Status}_{\text{sleep}}, \text{Status}_{\text{paralyze}}, \text{Status}_{\text{confuse}}] \in \mathbb{R}^{7}
$$

- The linear projection maps attributes to embedding space:

$$
\mathbf{W}_{\text{unit}} \in \mathbb{R}^{128 \times 7}
$$

- The spatial type index distinguishes active and bench slots:

$$
\tau_{\text{unit}} \in \{\text{T\_SELF\_ACTIVE}, \text{T\_SELF\_BENCH}, \text{T\_OPP\_ACTIVE}, \text{T\_OPP\_BENCH}\}
$$

---

### 2.3. Option Stream (`_opt_stream`)
For each valid candidate decision emitted by the environment:

$$
\mathbf{o}_{\text{action}} = \mathbf{E}_{\text{verb}}(v) + \mathbf{W}_{\text{src}} \mathbf{t}_{\text{src}} + \mathbf{W}_{\text{tgt}} \mathbf{t}_{\text{tgt}} + \mathbf{W}_{\text{attr}} \mathbf{a}_{\text{opt}} + \mathbf{E}_{\text{atk}}(a_{\text{idx}})
$$

Where:
- Action verb type:

$$
v \in \{0, 1, \dots, 11\}
$$

- Attack index:

$$
a_{\text{idx}} \in \{0, 1, 2, 3\}
$$

- Structured option attributes:

$$
\mathbf{a}_{\text{opt}} \in \mathbb{R}^{24}
$$

---

## 3. Transformer Decoder Architecture & Attention Math

The backbone consists of stacked Transformer Encoder layers in standard Post-LayerNorm configuration (`norm_first=False`):

```
x_0 = X_in
h_l = Attention(x_{l-1}, x_{l-1}, x_{l-1})
x'_l = LayerNorm(x_{l-1} + h_l)
f_l = MLP(x'_l)
x_l = LayerNorm(x'_l + f_l)
```

### 3.1. Scaled Multi-Head Self-Attention
For number of heads $H = 4$ and head dimension $d_k = \frac{D}{H} = 32$:

$$
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V
$$

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{32}} + \mathbf{M}_{\text{mask}} \right) \mathbf{V}
$$

### 3.2. Feedforward Network (MLP)
The feedforward projection expands by a factor of 4:

$$
\text{MLP}(\mathbf{x}) = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
$$

Where:

$$
\mathbf{W}_1 \in \mathbb{R}^{512 \times 128}, \quad \mathbf{W}_2 \in \mathbb{R}^{128 \times 512}
$$

---

## 4. Scratch Registers & Truncated Backpropagation Through Time (TBPTT)

To endow the agent with working memory across decision steps without quadratic context growth, the architecture allocates 16 learnable scratch tokens:

$$
\mathbf{S} \in \mathbb{R}^{16 \times 128}
$$

### 4.1. Recurrence Dynamics
1. **Cold Step ($t=0$ or episode boundary)**:
   The scratch register inputs are seeded from a learned parameter:

$$
\mathbf{S}_0 = \mathbf{S}_{\text{learned\_init}} \in \mathbb{R}^{16 \times 128}
$$

2. **Step Progression ($t \to t+1$)**:
   The Transformer encoder processes the full token set and outputs updated scratch token states:

$$
\mathbf{S}_{t+1} = \mathbf{X}_{\text{out}}[\text{slice}(\text{scratch\_start}, \text{scratch\_end})]
$$

3. **TBPTT Truncation**:
   Memory is propagated across sequential decision steps within a chunk of size $K=32$. Gradients are truncated at chunk boundaries:

$$
\mathbf{S}_{\text{chunk\_next}} = \text{stop\_gradient}(\mathbf{S}_{\text{chunk\_last}})
$$

---

## 5. Muon + AdamW Split Optimization Engine

The training pipeline implements a hybrid optimizer that decouples 2D matrix weights from 1D embeddings and normalization layers.

### 5.1. Parameter Routing Rules
- **Muon (Matrix Update with Orthogonalization)**: Applied strictly to 2D weight matrices ($\text{ndim} == 2$) of internal projections and Transformer linear layers (`encoder.layers.*.attn.*.weight`, `encoder.layers.*.ff.*.weight`, `unit_attr_proj.weight`, `opt_src_proj.weight`).
- **Structured AdamW**: Applied to verb-conditioned head parameters (`type_query.weight`, `type_bias.weight`) with high weight decay ($\lambda = 0.1$) to force rare verbs to regularize toward the shared fallback policy.
- **Default AdamW**: Applied to embeddings (`card_emb`, `type_emb`, `meta_bucket_emb`), normalization scales/biases, and linear bias vectors.

### 5.2. Muon Newton-Schulz Matrix Orthogonalization
Given gradient $\mathbf{G} \in \mathbb{R}^{M \times N}$, Muon computes an approximate polar decomposition update via iterative matrix multiplications:

$$
\mathbf{X}_0 = \frac{\mathbf{G}}{\|\mathbf{G}\|_F + \epsilon}
$$

Iterate for $k = 1, \dots, 5$:

$$
\mathbf{X}_{k+1} = a \mathbf{X}_k + b (\mathbf{X}_k \mathbf{X}_k^\top) \mathbf{X}_k + c (\mathbf{X}_k \mathbf{X}_k^\top)^2 \mathbf{X}_k
$$

With quintic coefficients:

$$
a = 3.4445, \quad b = -4.7750, \quad c = 2.0315
$$

The updated orthogonal matrix $\mathbf{X}_5$ replaces the raw gradient in momentum accumulation, ensuring uniform spectral updates across all attention heads.

---

## 6. Multi-Task Loss Formulation

The objective function optimizes the behavioral cloning decision policy jointly with four auxiliary predictive heads:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + \lambda_{\text{ko}} \mathcal{L}_{\text{ko}} + \lambda_{\text{prize}} \mathcal{L}_{\text{prize}} + \lambda_{\text{term}} \mathcal{L}_{\text{terminal}} + \lambda_{\text{return}} \mathcal{L}_{\text{return}}
$$

### 6.1. Policy Loss (Cross-Entropy with Option Masking)

$$
\mathcal{L}_{\text{policy}} = -\sum_{i=1}^{B} \log \left( \frac{\exp(\mathbf{z}_{i, y_i})}{\sum_{j \in \mathcal{A}_{\text{valid}}(i)} \exp(\mathbf{z}_{i, j})} \right)
$$

### 6.2. Auxiliary Predictive Losses
1. **Knockout Head (Binary Cross-Entropy)**:

$$
\mathcal{L}_{\text{ko}} = -\frac{1}{N_{\text{valid}}} \sum_{i \in \text{valid}} \left[ y_i^{\text{ko}} \log \sigma(\hat{z}_i^{\text{ko}}) + (1 - y_i^{\text{ko}}) \log (1 - \sigma(\hat{z}_i^{\text{ko}})) \right]
$$

2. **Prize Delta Head (Mean Squared Error)**:

$$
\mathcal{L}_{\text{prize}} = \frac{1}{N_{\text{valid}}} \sum_{i \in \text{valid}} (\hat{p}_i - y_i^{\text{prize}})^2
$$

3. **Terminal Outcome Head (Binary Cross-Entropy)**:

$$
\mathcal{L}_{\text{terminal}} = -\frac{1}{N_{\text{valid}}} \sum_{i \in \text{valid}} \left[ y_i^{\text{term}} \log \sigma(\hat{z}_i^{\text{term}}) + (1 - y_i^{\text{term}}) \log (1 - \sigma(\hat{z}_i^{\text{term}})) \right]
$$

4. **Return Head (Mean Squared Error)**:

$$
\mathcal{L}_{\text{return}} = \frac{1}{N_{\text{valid}}} \sum_{i \in \text{valid}} (\hat{r}_i - y_i^{\text{return}})^2
$$

---

## 7. Current Architectural Gap Analysis (As-Built vs. Target MoE)

| Subsystem | As-Built State (`develop`) | Target State (`Magnum Opus MoE`) | Architectural Delta |
| :--- | :--- | :--- | :--- |
| **Positional Encoding** | Discrete Type Embeddings only (No RoPE) | 4D RoPEND ($c_1$: Step, $c_2$: Day, $c_3$: Clock, $c_4$: Elo) | Requires 4D rotation operators in self-attention |
| **Policy Routing** | Single Monolithic Transformer Decoder | Mixture of Experts (Softmax Gating + 4 Sub-Networks) | Requires routing loss + load balancing entropy |
| **Elo Awareness** | Static meta bucket embeddings | Stochastic in-game Elo derivation from opponent model | Real-time Bayesian prior integration |
| **Pre-Game Reasoning** | Direct step-0 evaluation | Autoregressive 60-Card Draft Prediction | Pre-match cross-attention on vehicle synergy |
| **Runtime Mode** | Static weights | Airgap Apex Mode activated when $T_{\text{OS}} \ge \text{Aug 16}$ | Dynamic conditional execution graph |
