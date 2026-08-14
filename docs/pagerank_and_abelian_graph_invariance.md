# PageRank & Abelian Graph Invariance: Mathematical Monograph

**Document**: Research Monograph — Graph Theory & Algebraic Invariance  
**Author**: Research Director & Alefita (Fitalabs AI Research)  
**Classification**: Applied Spectral Graph Theory & Bradley-Terry Scaling  
**Target Ingestion**: GPT-5.6 Sol, DeepSeek-V4-Pro, Codex, Claude 3.7  
**Date**: August 14, 2026  

---

## 1. The Dual Incomplete Graph Problem

Both knowledge retrieval systems (the Wikifita Atlas) and competitive multi-agent game environments (Pokémon TCG AI Challenge) confront the identical mathematical challenge: **inferring an invariant latent measure over an incomplete, directed, stochastic interaction graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$.**

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                                  THE DUAL GRAPH ISOMORPHISM                                       |
|                                                                                                   |
|  [Wikifita Knowledge Graph]                              [Pokémon TCG Tournament Graph]           |
|  - Nodes: Markdown Pages (u, v)                         - Nodes: Decks / Agents (d_i, d_j)        |
|  - Edges: [[wikilinks]]                                 - Edges: Match Outcomess (Win/Loss)       |
|  - Anomaly: Dangling Nodes (Outdegree = 0)              - Anomaly: Low Sample Volatility (N < 10) |
|  - Solution: Teleportation & Dangling Redistribution     - Solution: MD10 Shrinkage & Abelian Shift|
+───────────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 2. Spectral PageRank Formulation in Wikifita (`lib/wiki.ts`)

In the Wikifita Atlas, the importance vector $\mathbf{r} \in \mathbb{R}^N$ represents the stationary distribution of a random walk with damping factor $d = 0.85$ and convergence tolerance $\epsilon = 10^{-10}$:

$$
r(i)^{(t+1)} = \frac{1 - d}{N} + d \cdot \left( \sum_{j \in \text{inlinks}(i)} \frac{r(j)^{(t)}}{\text{outdegree}(j)} + \frac{\text{danglingMass}^{(t)}}{N} \right)
$$

Where the dangling mass regularizer captures probability leakage from leaf pages:

$$
\text{danglingMass}^{(t)} = \sum_{j \,:\, \text{outdegree}(j) = 0} r(j)^{(t)}
$$

---

## 3. Bradley-Terry Abelian Invariance Formulation (`rl/results_db.py`)

In the Pokémon TCG AI Challenge, the true latent skill of a deck $R_{\text{invariante}}(N)$ is estimated from an empirical win rate $w = \frac{W}{N}$ across non-uniform, sparse match counts:

### 3.1. Asymptotic Logistic Inversion
$$
\hat{R}_{\infty} = 600.0 + 400.0 \cdot \log_{10}\left( \frac{w}{1 - w} \right)
$$

### 3.2. MD10 Placement Regularization ($N_0 = 10$)
$$
R_{\text{smoothed}} = \left(\frac{N}{N + 10}\right) \cdot \hat{R}_{\infty} + \left(\frac{10}{N + 10}\right) \cdot 600.0
$$

### 3.3. Softmax Abelian Translation ($\Delta R_{\text{Abeliano}}$)
$$
\alpha_k = \frac{\exp(N_k / 20.0)}{\sum_{j \in \mathcal{C}} \exp(N_j / 20.0)}
$$

$$
\Delta R_{\text{Abeliano}} = \sum_{k \in \mathcal{C}} \alpha_k \cdot \left( R_k^{\text{remote}} - \hat{R}_{k,\infty}^{\text{local}} \right)
$$

### 3.4. Final Invariant Measure
$$
R_{\text{invariante}}(N) = R_{\text{smoothed}} + \Delta R_{\text{Abeliano}}
$$

---

## 4. Mathematical Comparison Matrix

| Property | Wikifita PageRank | Sample-Size Invariant Elo |
| :--- | :--- | :--- |
| **Domain** | Knowledge Graph / Information Retrieval | Competitive Policy Evaluation / Game Theory |
| **Graph Topology** | Directed citation network $\mathcal{V}_{\text{pages}}$ | Stochastic bipartite match pairings $\mathcal{V}_{\text{decks}}$ |
| **Primary Operator** | Markov transition matrix with damping | Bradley-Terry logistic link function |
| **Boundary Shrinkage** | Uniform teleportation $(1-d)/N$ | MD10 Bayesian prior shrinkage $\frac{10}{N+10} \cdot 600$ |
| **Missing Mass Handling** | Dangling mass redistribution | Softmax Abelian group translation $\Delta R_{\text{Abeliano}}$ |
| **Convergence Guarantee** | Perron-Frobenius theorem ($L_1 < 10^{-10}$) | Convex MLE optimization with Abelian shift invariance |

---

## 5. Architectural Implications for Multi-Agent Swarms

1. **PageRank-Weighted Retrieval**: Subagents ingesting Wikifita prioritize documents according to their stationary PageRank, preventing low-density peripheral notes from overwhelming the context window.
2. **Abelian-Calibrated Tournament Scheduling**: The tournament orchestrator (`scripts/tournament.py`) selects opponent decks based on their invariant rating $R_{\text{invariante}}$, maximizing the Fisher Information of each local battle.
