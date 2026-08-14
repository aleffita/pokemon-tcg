# Empirical Ablation Monograph & Game-Theoretic Meta Analysis

**Document**: Level 3 Empirical Research & Game Theory Monograph  
**Author**: Fitalabs AI Research  
**Target Ingestion**: GPT-5.6 Sol, DeepSeek-V4-Pro, Codex, Claude 3.7  
**Date**: August 14, 2026  

---

## 1. The Cross-Stage Ablation Matrix (420 Matches per Stage)

To evaluate the progression of Behavioral Cloning Curriculum V1, a cross-stage tournament matrix was executed. Each checkpoint competed in a heavy asymmetric matrix against the top 5 decks of the anchor teacher model (`first_sub_kaggle_2707`).

```
+---------------------------------------------------------------------------------------------------+
|                               CURRICULUM V1 ABLATION PROGRESSION                                  |
|                                                                                                   |
|  Stage 1 (Raw BC All-Elo)          ───> 14.3% Overall WR  (Peak Deck 633: 22.9%)                  |
|  Stage 2 (Filtered Elo >= 600)     ───> 15.2% Overall WR  (Peak Deck 633: 26.4%)                  |
|  Stage 3 (Loss-Corrupted Top-100)  ───> 13.8% Overall WR  (Peak Deck 633: 22.9%)                  |
|  Stage 4 (Loss-Corrected Top-100)  ───> 17.1% Overall WR  (Peak Deck 633: 27.9%)                  |
|                                                                                                   |
|  Baseline Anchor (first_sub 2707)  ───> 67.16% Public Kaggle Win Rate (Teacher Ceiling)           |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Decoupling of Validation Accuracy & Game-Theoretic Win Rate

A critical finding in the Curriculum V1 experiments is the empirical independence between Cross-Entropy validation accuracy ($\text{Acc}_{\text{val}}$) and tournament win rate ($\text{WR}_{\text{tourn}}$).

### 2.1. Mathematical Formulation of the Decoupling

Let $\mathcal{D}_{\text{human}}$ be the empirical distribution of actions in the Kaggle replay dataset, and let $\pi^*$ be the minimax-optimal policy in the Pokémon TCG POMDP.

The Behavioral Cloning objective minimizes:

$$
\mathcal{L}_{\text{BC}}(\theta) = D_{\text{KL}}(\pi_{\text{human}} \,\|\, \pi_{\theta})
$$

As training progresses:

$$
\lim_{N_{\text{epochs}} \to \infty} \pi_{\theta} = \pi_{\text{human}} \implies \text{Acc}_{\text{val}} \to \text{Acc}_{\max} \approx 78.4\%
$$

However, the expected win rate against an optimal opponent $\pi^*$ is governed by the state value function:

$$
\text{WR}(\pi_{\theta}, \pi^*) = \mathbb{E}_{\tau \sim (\pi_{\theta}, \pi^*)} \left[ \mathbb{I}(R_T = +1) \right]
$$

Because $\pi_{\text{human}}$ contains suboptimal play, bluffs, hesitation, and tactical blunders:

$$
\text{WR}(\pi_{\text{human}}, \pi^*) \ll 1.0 \implies \text{WR}(\pi_{\theta}, \pi^*) \le 17.1\%
$$

Maximizing prediction accuracy over a mediocre or average human dataset forces the neural network to replicate human mistakes. A high validation accuracy merely indicates that the model has successfully memorized the suboptimal human action distribution, bounding its tournament win rate to the ceiling of human mediocrity.

---

## 3. The "Pilot vs. Vehicle" Thesis (Deck Saliency Analysis)

In the ablation tournament for Stage 4 FP32, the exact same neural weights exhibited radically different win rates depending exclusively on the chosen deck list:

| Deck Identifier | Archetype | Composition | Overall Win Rate | Peak vs. Opponent |
| :--- | :--- | :--- | :--- | :--- |
| **Deck #633** | Yan Archetype | Fast energy acceleration, low-cost attackers | **27.9% (39 W / 101 L)** | 15.0% vs Top Decks |
| **Deck #21** | Oshbocker Archetype | High HP basic tanks, slow setup | **20.0% (28 W / 112 L)** | 40.0% vs specific variant |
| **Deck #251** | Submission Default | Balanced starter / generic energy curve | **12.9% (18 W / 122 L)** | 5.0% vs Top Decks |

```
                 PILOT VS. VEHICLE PERFORMANCE DIVERGENCE
  30% ┼─────────────────────────────────────────────── Deck #633 (27.9%)
      │                                                Yan Fast Acceleration
  20% ┼─────────────────────── Deck #21 (20.0%)
      │                        Oshbocker Setup Tank
  10% ┼────── Deck #251 (12.9%)
      │       Default Starter
   0% ┴───────────────────────────────────────────────
             (All evaluated under identical Stage 4 neural weights)
```

### Strategic Deduction
The neural policy (the Pilot) is fundamentally constrained by the structural capability of the 60-card list (the Vehicle). Even an optimal neural decision engine will suffer catastrophic failure if the underlying deck cannot maintain energy curve pacing. 

This justifies the **Autoregressive Draft Module** in the Magnum Opus architecture: the agent must encode and attend over its own 60-card vehicle list during the pre-match step.

---

## 4. The FP16 Precision Collapse: Numerical Root Cause Analysis

During the migration from native MLX execution to standalone PyTorch CPU evaluation, the agent experienced a severe performance drop from 45.0% WR to 3.3% WR.

### 4.1. Underflow in Scaled Dot-Product Attention
In FP16 representation, the smallest positive normal float is:

$$
\epsilon_{\text{FP16}} \approx 6.10 \times 10^{-5}
$$

For embedding dimension $D=128$, the pre-Softmax dot-product scores between query and key vectors are computed as:

$$
S_{i, j} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{32}}
$$

When attention layers undergo LayerNorm, small variance vectors in early layers produced logits where:

$$
S_{i, j} - \max_k(S_{i, k}) < -10.0 \implies \exp(S_{i, j}) \to 0.0 \quad \text{in FP16}
$$

Under PyTorch's native CPU FP16 kernels, this caused the Softmax denominator to collapse to zero, producing `NaN` or completely uniform probability distributions over action options.

### 4.2. Resolution via Strict FP32 Boundaries
By enforcing `float32` across all inference tensors and embedding tables:

$$
\epsilon_{\text{FP32}} \approx 1.19 \times 10^{-7}
$$

The dynamic range is expanded from $[-6.5 \times 10^4, 6.5 \times 10^4]$ to $[-3.4 \times 10^{38}, 3.4 \times 10^{38}]$, eliminating Softmax underflow and restoring full decision entropy.

---

## 5. Kaggle Sampling Bias & The Abelian Scale Isomorphism

Empirical auditing of the Kaggle replay dataset revealed that only **1.05% to 5.57%** of daily competition matches are exported in the public ZIP archives. Out of 6,791 unique registered teams, only 1,059 teams appear in replay logs.

### 5.1. Impact on Raw Elo Calculation
Because top-ranked teams play thousands of hidden ladder matches, calculating raw Elo on the sparse 5% sample severely underestimates their true rating:

$$
N_{\text{observed}} \ll N_{\text{true}} \implies R_{\text{raw}} \approx 600.0 \quad \text{for elite agents}
$$

### 5.2. Mathematical Correction via Softmax Abelian Translation
By computing the translation offset $\Delta R_{\text{Abeliano}}$ over the overlapping subset $\mathcal{C}$ with the live leaderboard:

$$
\Delta R_{\text{Abeliano}} = \sum_{k \in \mathcal{C}} \frac{\exp(N_k / 20.0)}{\sum_{j \in \mathcal{C}} \exp(N_j / 20.0)} \left( R_k^{\text{remote}} - \hat{R}_{k, \infty}^{\text{local}} \right)
$$

The system maps local tournament ratings onto the official leaderboard scale, ensuring that local ablation metrics are calibrated directly against the competitive standard.
