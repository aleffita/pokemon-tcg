# Deep Mathematical Formulation: Section 1 & Section 2 of the PageRank-Abelian Monograph

**Document Reference**: Milestone 3 Technical Monograph — Graph Theory & Spectral Markov Chains  
**Author**: Explorer 1 (Milestone 3 Sub-Swarm)  
**Target Ingestion**: `docs/pagerank_and_abelian_graph_invariance.md`  
**Classification**: Spectral Graph Theory, Markov Chains, Perron-Frobenius Operators  
**Date**: August 14, 2026  

---

## Executive Overview

This report establishes the rigorous mathematical foundation for **Section 1 (The Dual Incomplete Graph Problem)** and **Section 2 (Spectral PageRank Markov Chain Stationarity & Ergodicity)** of the research monograph `docs/pagerank_and_abelian_graph_invariance.md`.

It provides:
1. Complete formalization of the directed incomplete interaction graph, adjacency representations, degree distributions, and the dangling node absorbing state anomaly.
2. Formalization and algebraic decomposition of the regularized column-stochastic transition operator with dangling mass redistribution and uniform teleportation.
3. Complete statements and step-by-step proofs of the Perron-Frobenius theorem application, the algebraic and geometric multiplicity of the dominant eigenvalue, the spectral gap bound, and the geometric contraction rate of power iteration under the L1 norm.
4. Production-grade algorithmic pseudocode, TypeScript reference implementation (matching Wikifita Atlas `lib/wiki.ts`), and high-performance Python/NumPy implementation.

---

## Section 1: The Dual Incomplete Graph Problem

### 1.1. Graph Theoretical Formulation

Let a directed incomplete interaction network be modeled as a directed graph:

$$
\mathcal{G} = (\mathcal{V}, \mathcal{E})
$$

Where:
- $\mathcal{V} = \{v_1, v_2, \dots, v_N\}$ is the finite set of $N = |\mathcal{V}|$ nodes.
- $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$ is the set of directed edges $(v_j, v_i)$ representing a directional relationship or information flow from source node $v_j$ to target node $v_i$.

The topology of $\mathcal{G}$ is represented by the adjacency matrix $\mathbf{A} \in \{0, 1\}^{N \times N}$, defined by:

$$
A_{ji} = \begin{cases} 1 & \text{if } (v_j, v_i) \in \mathcal{E} \\ 0 & \text{otherwise} \end{cases}
$$

Note that in this column-stochastic convention, $A_{ji}$ represents a link from row index $j$ to target column index $i$.

### 1.2. Degree Distributions

For each node $v_j \in \mathcal{V}$, the out-degree $\text{deg}_{\text{out}}(v_j)$ and in-degree $\text{deg}_{\text{in}}(v_j)$ are defined as:

$$
\text{deg}_{\text{out}}(v_j) = \sum_{i=1}^N A_{ji} = \sum_{i: (v_j, v_i) \in \mathcal{E}} 1
$$

$$
\text{deg}_{\text{in}}(v_i) = \sum_{j=1}^N A_{ji} = \sum_{j: (v_j, v_i) \in \mathcal{E}} 1
$$

Let $\mathcal{I}(i) = \{v_j \in \mathcal{V} : (v_j, v_i) \in \mathcal{E}\}$ denote the set of in-neighbors of node $v_i$, and $\mathcal{O}(j) = \{v_i \in \mathcal{V} : (v_j, v_i) \in \mathcal{E}\}$ denote the set of out-neighbors of node $v_j$.

### 1.3. Raw Transition Probability Matrix and Column-Substochasticity

The unregularized random walk transition probability matrix $\mathbf{P} \in \mathbb{R}^{N \times N}$ assigns uniform transition probabilities across outgoing edges:

$$
P_{ij} = \begin{cases} \frac{A_{ji}}{\text{deg}_{\text{out}}(v_j)} & \text{if } \text{deg}_{\text{out}}(v_j) > 0 \\ 0 & \text{if } \text{deg}_{\text{out}}(v_j) = 0 \end{cases}
$$

In matrix notation, let $\mathbf{D}_{\text{out}} \in \mathbb{R}^{N \times N}$ be the diagonal out-degree matrix with $(D_{\text{out}})_{jj} = \text{deg}_{\text{out}}(v_j)$. Define the pseudoinverse $\mathbf{D}_{\text{out}}^{\dagger}$ by:

$$
(D_{\text{out}}^{\dagger})_{jj} = \begin{cases} \frac{1}{\text{deg}_{\text{out}}(v_j)} & \text{if } \text{deg}_{\text{out}}(v_j) > 0 \\ 0 & \text{if } \text{deg}_{\text{out}}(v_j) = 0 \end{cases}
$$

The transition matrix $\mathbf{P}$ is expressed as:

$$
\mathbf{P} = \mathbf{A}^T \mathbf{D}_{\text{out}}^{\dagger}
$$

Let $\mathbf{e} = (1, 1, \dots, 1)^T \in \mathbb{R}^N$ be the all-ones vector. The column sums of $\mathbf{P}$ satisfy:

$$
\sum_{i=1}^N P_{ij} = (\mathbf{e}^T \mathbf{P})_j = \begin{cases} 1 & \text{if } \text{deg}_{\text{out}}(v_j) > 0 \\ 0 & \text{if } \text{deg}_{\text{out}}(v_j) = 0 \end{cases}
$$

Consequently, $\mathbf{P}$ is **column-substochastic**: $\mathbf{e}^T \mathbf{P} \le \mathbf{e}^T$. If there exists at least one node $v_j$ with $\text{deg}_{\text{out}}(v_j) = 0$, then $\mathbf{P}$ fails to be column-stochastic.

### 1.4. The Dangling Node Absorbing State Anomaly

A node $v_j \in \mathcal{V}$ with $\text{deg}_{\text{out}}(v_j) = 0$ is termed a **dangling node** (or dead end / leaf node). Let $\mathcal{V}_{\text{dangling}} = \{v_j \in \mathcal{V} : \text{deg}_{\text{out}}(v_j) = 0\}$ be the set of dangling nodes, with cardinality $N_d = |\mathcal{V}_{\text{dangling}}|$.

Define the dangling indicator vector $\mathbf{d} \in \{0, 1\}^N$ by:

$$
d_j = \begin{cases} 1 & \text{if } \text{deg}_{\text{out}}(v_j) = 0 \\ 0 & \text{if } \text{deg}_{\text{out}}(v_j) > 0 \end{cases}
$$

#### Mathematical Consequence of Dangling Nodes
Consider an uncorrected discrete-time Markov chain with state probability vector $\mathbf{r}^{(t)} \in \mathbb{R}^N$ evolving under:

$$
\mathbf{r}^{(t+1)} = \mathbf{P} \mathbf{r}^{(t)}
$$

The total probability mass at step $t+1$ is given by:

$$
\|\mathbf{r}^{(t+1)}\|_1 = \mathbf{e}^T \mathbf{r}^{(t+1)} = \mathbf{e}^T \mathbf{P} \mathbf{r}^{(t)} = (\mathbf{e} - \mathbf{d})^T \mathbf{r}^{(t)} = \|\mathbf{r}^{(t)}\|_1 - \mathbf{d}^T \mathbf{r}^{(t)}
$$

Where $\mathbf{d}^T \mathbf{r}^{(t)} = \sum_{j \in \mathcal{V}_{\text{dangling}}} r(j)^{(t)} \ge 0$ is the **dangling probability leakage**.

If $\mathcal{G}$ contains dangling nodes accessible from non-dangling components, then as $t \to \infty$:

$$
\lim_{t \to \infty} \|\mathbf{r}^{(t)}\|_1 = 0
$$

The uncorrected transition matrix $\mathbf{P}$ possesses a spectral radius strictly less than 1 on the non-absorbing subspace ($\rho(\mathbf{P}) \le 1$), causing the total probability mass to drain to zero.

### 1.5. The Dual Structural Isomorphism

The incomplete directed graph problem manifests symmetrically across two distinct domains within the project ecosystem:

| Dimension | Wikifita Knowledge Graph | Pokémon TCG Tournament Graph |
| :--- | :--- | :--- |
| **Node Set $\mathcal{V}$** | Markdown Knowledge Articles ($N \approx 50-250$) | Competitive Decks / Policy Agents ($K \approx 10-100$) |
| **Edge Set $\mathcal{E}$** | Semantic Wikilinks `[[target]]` | Pairwise Match Encounters ($d_i \text{ vs } d_j$) |
| **Edge Weight / Value** | Out-degree normalized citation $1/\text{deg}_{\text{out}}$ | Empirical Win Rate $w = W / N$ |
| **Graph Anomaly** | Dangling Nodes ($\text{deg}_{\text{out}} = 0$, external links) | Low Sample Sparsity ($N < 10$), Disconnected Components |
| **Pathological Outcome** | Probability mass leakage ($\|\mathbf{r}^{(t)}\|_1 \to 0$) | Infinite Log-Odds Singularity ($\log(w / (1-w)) \to \pm \infty$) |
| **Stochastic Regularization** | Uniform Teleportation $(1-d)/N$ + Dangling Redistribution | Bayesian Prior Shrinkage $\frac{10}{N+10} \cdot 600.0$ (MD10) |
| **Global Calibration** | Stationary Eigenvector $\mathbf{r}^* \in \Delta^N$ | Softmax Translation $\Delta R_{\text{Abeliano}}$ over Overlap $\mathcal{C}$ |
| **Target Invariant** | Stationary Importance Metric $r^*(i)$ | Scale-Invariant Latent Elo $R_{\text{invariante}}(i)$ |

Both systems solve the identical structural challenge: **reconstructing a unique, invariant, strictly positive latent measure over a sparse, incomplete, and non-ergodic directed interaction graph.**

---

## Section 2: Spectral PageRank Markov Chain Stationarity & Ergodicity

### 2.1. Exact Dangling Mass Redistribution Operator

To eliminate probability leakage without modifying the underlying citation structure, all probability mass accumulated at dangling nodes is collected and redistributed uniformly across all $N$ nodes.

At step $t$, the total dangling mass is:

$$
\text{danglingMass}^{(t)} = \mathbf{d}^T \mathbf{r}^{(t)} = \sum_{j : \text{deg}_{\text{out}}(v_j) = 0} r(j)^{(t)}
$$

We define the closed stochastic matrix $\mathbf{M} \in \mathbb{R}^{N \times N}$ by replacing all zero columns of $\mathbf{P}$ with uniform distribution vectors $\frac{1}{N} \mathbf{e}$:

$$
\mathbf{M} = \mathbf{P} + \frac{1}{N} \mathbf{e} \mathbf{d}^T
$$

Each entry $M_{ij}$ is given by:

$$
M_{ij} = \begin{cases} \frac{A_{ji}}{\text{deg}_{\text{out}}(v_j)} & \text{if } \text{deg}_{\text{out}}(v_j) > 0 \\ \frac{1}{N} & \text{if } \text{deg}_{\text{out}}(v_j) = 0 \end{cases}
$$

#### Lemma 1 (Column-Stochasticity of $\mathbf{M}$)
The matrix $\mathbf{M}$ is strictly column-stochastic: $\mathbf{e}^T \mathbf{M} = \mathbf{e}^T$.

*Proof.*  
Multiplying $\mathbf{e}^T$ by $\mathbf{M}$:

$$
\mathbf{e}^T \mathbf{M} = \mathbf{e}^T \left( \mathbf{P} + \frac{1}{N} \mathbf{e} \mathbf{d}^T \right) = \mathbf{e}^T \mathbf{P} + \frac{1}{N} (\mathbf{e}^T \mathbf{e}) \mathbf{d}^T
$$

Since $\mathbf{e}^T \mathbf{e} = N$ and $\mathbf{e}^T \mathbf{P} = (\mathbf{e} - \mathbf{d})^T$:

$$
\mathbf{e}^T \mathbf{M} = (\mathbf{e} - \mathbf{d})^T + \frac{N}{N} \mathbf{d}^T = \mathbf{e}^T - \mathbf{d}^T + \mathbf{d}^T = \mathbf{e}^T \quad \blacksquare
$$

### 2.2. Uniform Teleportation and the Google Transition Matrix

While $\mathbf{M}$ resolves probability leakage, the directed graph $\mathcal{G}$ may still contain disconnected components, periodic cycles, or rank sinks (strongly connected components with no outgoing edges to the rest of the graph).

To enforce irreducibility and aperiodicity, a random surfer teleportation mechanism is introduced. At each discrete time step, with probability $d \in (0, 1)$ (the damping factor, canonically $d = 0.85$), the surfer follows an outgoing link from $\mathbf{M}$; with probability $1 - d$, the surfer teleports uniformly to any node in $\mathcal{V}$.

The regularized transition matrix (the Google matrix) $\mathbf{\tilde{P}} \in \mathbb{R}^{N \times N}$ is defined as:

$$
\mathbf{\tilde{P}} = d \mathbf{M} + \frac{1 - d}{N} \mathbf{e} \mathbf{e}^T
$$

Substituting $\mathbf{M} = \mathbf{P} + \frac{1}{N} \mathbf{e} \mathbf{d}^T$:

$$
\mathbf{\tilde{P}} = d \left( \mathbf{P} + \frac{1}{N} \mathbf{e} \mathbf{d}^T \right) + \frac{1 - d}{N} \mathbf{e} \mathbf{e}^T = d \mathbf{P} + \frac{1}{N} \mathbf{e} \left( d \mathbf{d}^T + (1 - d) \mathbf{e}^T \right)
$$

Component-wise, for every entry $(i, j) \in \{1, \dots, N\} \times \{1, \dots, N\}$:

$$
\tilde{P}_{ij} = d \cdot P_{ij} + \frac{d \cdot d_j + (1 - d)}{N} = \begin{cases} d \cdot \frac{A_{ji}}{\text{deg}_{\text{out}}(v_j)} + \frac{1 - d}{N} & \text{if } \text{deg}_{\text{out}}(v_j) > 0 \\ \frac{d}{N} + \frac{1 - d}{N} = \frac{1}{N} & \text{if } \text{deg}_{\text{out}}(v_j) = 0 \end{cases}
$$

### 2.3. Component-Wise Power Iteration Operator

Applying $\mathbf{\tilde{P}}$ to the rank vector $\mathbf{r}^{(t)}$ yields the component-wise iteration formula:

$$
r(i)^{(t+1)} = (\mathbf{\tilde{P}} \mathbf{r}^{(t)})_i = \frac{1 - d}{N} \sum_{j=1}^N r(j)^{(t)} + d \sum_{j \in \mathcal{I}(i)} \frac{r(j)^{(t)}}{\text{deg}_{\text{out}}(v_j)} + \frac{d}{N} \sum_{j : \text{deg}_{\text{out}}(v_j) = 0} r(j)^{(t)}
$$

Under the probability simplex invariant $\|\mathbf{r}^{(t)}\|_1 = \sum_{j=1}^N r(j)^{(t)} = 1$, this simplifies to:

$$
r(i)^{(t+1)} = \frac{1 - d}{N} + d \left( \sum_{j \in \mathcal{I}(i)} \frac{r(j)^{(t)}}{\text{deg}_{\text{out}}(v_j)} + \frac{\text{danglingMass}^{(t)}}{N} \right)
$$

Where:

$$
\text{danglingMass}^{(t)} = \sum_{j \in \mathcal{V}_{\text{dangling}}} r(j)^{(t)}
$$

---

## 3. Rigorous Spectral Theorems and Convergence Proofs

### 3.1. Theorem 1: Column-Stochasticity of the Regularized Operator

**Theorem 1.** The regularized operator $\mathbf{\tilde{P}}$ is strictly column-stochastic: $\mathbf{e}^T \mathbf{\tilde{P}} = \mathbf{e}^T$.

*Proof.*  
Direct computation using Lemma 1:

$$
\mathbf{e}^T \mathbf{\tilde{P}} = \mathbf{e}^T \left( d \mathbf{M} + \frac{1 - d}{N} \mathbf{e} \mathbf{e}^T \right) = d (\mathbf{e}^T \mathbf{M}) + \frac{1 - d}{N} (\mathbf{e}^T \mathbf{e}) \mathbf{e}^T
$$

Since $\mathbf{e}^T \mathbf{M} = \mathbf{e}^T$ and $\mathbf{e}^T \mathbf{e} = N$:

$$
\mathbf{e}^T \mathbf{\tilde{P}} = d \mathbf{e}^T + \frac{1 - d}{N} (N) \mathbf{e}^T = d \mathbf{e}^T + (1 - d) \mathbf{e}^T = \mathbf{e}^T \quad \blacksquare
$$

### 3.2. Theorem 2: Strict Positivity, Irreducibility, and Primitivity

**Theorem 2.** For any damping factor $d \in (0, 1)$, the transition matrix $\mathbf{\tilde{P}}$ is strictly positive, irreducible, and primitive.

*Proof.*  
1. **Strict Positivity**: For every pair $(i, j) \in \{1, \dots, N\}^2$:
   - If $\text{deg}_{\text{out}}(v_j) > 0$, then $\tilde{P}_{ij} = d \frac{A_{ji}}{\text{deg}_{\text{out}}(v_j)} + \frac{1-d}{N} \ge \frac{1-d}{N} > 0$.
   - If $\text{deg}_{\text{out}}(v_j) = 0$, then $\tilde{P}_{ij} = \frac{1}{N} > 0$.
   Therefore, $\mathbf{\tilde{P}} > \mathbf{0}$ (all entries are strictly positive real numbers).

2. **Irreducibility**: A matrix is irreducible if its directed graph is strongly connected. Since $\tilde{P}_{ij} > 0$ for all $i, j$, there exists a direct edge between every pair of nodes $(v_j, v_i)$ in exactly 1 step. Thus, $\mathcal{G}_{\mathbf{\tilde{P}}}$ is a complete digraph with self-loops, hence irreducible.

3. **Primitivity & Aperiodicity**: A nonnegative matrix is primitive if there exists an integer $k \ge 1$ such that $\mathbf{\tilde{P}}^k > \mathbf{0}$. For $\mathbf{\tilde{P}}$, this holds for $k = 1$. The presence of strictly positive diagonal elements $\tilde{P}_{ii} > 0$ guarantees period $\gcd(\{k : (\tilde{P}^k)_{ii} > 0\}) = 1$. Thus, $\mathbf{\tilde{P}}$ is aperiodic and primitive. $\blacksquare$

### 3.3. Theorem 3: Perron-Frobenius Theorem Application

**Theorem 3 (Perron-Frobenius Spectral Characterization).**  
Let $\mathbf{\tilde{P}} \in \mathbb{R}^{N \times N}$ be the strictly positive, column-stochastic Google matrix with damping factor $d \in (0, 1)$. Then:
1. The spectral radius is $\rho(\mathbf{\tilde{P}}) = 1$.
2. $\lambda_1 = 1$ is an eigenvalue of $\mathbf{\tilde{P}}$ with algebraic multiplicity 1 and geometric multiplicity 1.
3. There exists a unique stationary probability vector $\mathbf{r}^* \in \mathbb{R}^N$ satisfying:
   $$
   \mathbf{\tilde{P}} \mathbf{r}^* = \mathbf{r}^*, \quad r^*(i) > 0 \;\; \forall i \in \{1, \dots, N\}, \quad \sum_{i=1}^N r^*(i) = 1
   $$
4. All other eigenvalues $\lambda_2, \lambda_3, \dots, \lambda_N \in \mathbb{C}$ satisfy the strict spectral gap bound:
   $$
   |\lambda_k| \le d < 1 \quad \forall k \in \{2, \dots, N\}
   $$

*Proof.*  
1. **Spectral Radius**: Since $\mathbf{\tilde{P}}$ is column-stochastic, $\|\mathbf{\tilde{P}}\|_1 = \max_{j} \sum_{i=1}^N |\tilde{P}_{ij}| = 1$. For any matrix norm, $\rho(\mathbf{\tilde{P}}) \le \|\mathbf{\tilde{P}}\|_1 = 1$. Furthermore, $\mathbf{e}^T \mathbf{\tilde{P}} = \mathbf{e}^T \implies \mathbf{\tilde{P}}^T \mathbf{e} = \mathbf{e}$, so 1 is an eigenvalue of $\mathbf{\tilde{P}}^T$, and consequently 1 is an eigenvalue of $\mathbf{\tilde{P}}$. Thus, $\rho(\mathbf{\tilde{P}}) = 1$.

2. **Dominant Eigenvector & Multiplicity**: By the classical Perron-Frobenius Theorem for strictly positive matrices ($\mathbf{\tilde{P}} > \mathbf{0}$):
   - The spectral radius $\rho = 1$ is a simple eigenvalue (algebraic multiplicity 1).
   - The corresponding right eigenvector $\mathbf{r}^*$ can be chosen to have strictly positive entries $r^*(i) > 0$ for all $i$.
   - Normalizing by $\|\mathbf{r}^*\|_1 = \mathbf{e}^T \mathbf{r}^* = 1$ yields a unique vector in the interior of the probability simplex $\Delta^N$.

3. **Spectral Gap Bound**:  
   We decompose $\mathbf{\tilde{P}}$ as a rank-1 perturbation of the stochastic matrix $\mathbf{M}$:
   $$
   \mathbf{\tilde{P}} = d \mathbf{M} + (1 - d) \mathbf{E}, \quad \text{where } \mathbf{E} = \frac{1}{N} \mathbf{e} \mathbf{e}^T
   $$
   Let $(\lambda, \mathbf{v})$ be an eigenpair of $\mathbf{\tilde{P}}$ such that $\mathbf{\tilde{P}} \mathbf{v} = \lambda \mathbf{v}$ with $\lambda \ne 1$ and $\mathbf{v} \ne \mathbf{0}$.  
   Left-multiplying by $\mathbf{e}^T$:
   $$
   \mathbf{e}^T \mathbf{\tilde{P}} \mathbf{v} = \lambda \mathbf{e}^T \mathbf{v}
   $$
   Using $\mathbf{e}^T \mathbf{\tilde{P}} = \mathbf{e}^T$:
   $$
   \mathbf{e}^T \mathbf{v} = \lambda \mathbf{e}^T \mathbf{v} \implies (1 - \lambda) (\mathbf{e}^T \mathbf{v}) = 0
   $$
   Since $\lambda \ne 1$, it must be that $\mathbf{e}^T \mathbf{v} = 0$ (the eigenvector $\mathbf{v}$ lies in the zero-sum subspace $V_0$).  
   Evaluating $\mathbf{\tilde{P}} \mathbf{v}$:
   $$
   \mathbf{\tilde{P}} \mathbf{v} = d \mathbf{M} \mathbf{v} + \frac{1 - d}{N} \mathbf{e} (\mathbf{e}^T \mathbf{v}) = d \mathbf{M} \mathbf{v} + \mathbf{0} = d \mathbf{M} \mathbf{v}
   $$
   Therefore:
   $$
   \lambda \mathbf{v} = d \mathbf{M} \mathbf{v} \implies \mathbf{M} \mathbf{v} = \left(\frac{\lambda}{d}\right) \mathbf{v}
   $$
   This implies that $\frac{\lambda}{d}$ is an eigenvalue of the column-stochastic matrix $\mathbf{M}$. Since $\mathbf{M}$ is column-stochastic, its spectral radius is bounded by $\rho(\mathbf{M}) \le \|\mathbf{M}\|_1 = 1$. Thus:
   $$
   \left| \frac{\lambda}{d} \right| \le \rho(\mathbf{M}) \le 1 \implies |\lambda| \le d
   $$
   Hence, every non-dominant eigenvalue $\lambda_k$ ($k \ge 2$) satisfies:
   $$
   |\lambda_k| \le d < 1 \quad \blacksquare
   $$

### 3.4. Theorem 4: L1 Norm Contraction and Geometric Convergence

**Theorem 4 (L1 Contraction Mapping and Power Iteration Rate).**  
Let $V_0 = \{\mathbf{x} \in \mathbb{R}^N : \mathbf{e}^T \mathbf{x} = 0\}$ be the subspace of zero-sum vectors equipped with the $L_1$ norm $\|\mathbf{x}\|_1 = \sum_{i=1}^N |x_i|$.
1. The operator $\mathbf{\tilde{P}}$ is a strict contraction on $V_0$ under the $L_1$ norm with contraction factor at most $d$:
   $$
   \|\mathbf{\tilde{P}} \mathbf{x}\|_1 \le d \|\mathbf{x}\|_1 \quad \forall \mathbf{x} \in V_0
   $$
2. For any initial probability vector $\mathbf{r}^{(0)} \in \Delta^N$, the power iteration $\mathbf{r}^{(t+1)} = \mathbf{\tilde{P}} \mathbf{r}^{(t)}$ converges to the unique stationary distribution $\mathbf{r}^*$ with geometric rate:
   $$
   \|\mathbf{r}^{(t)} - \mathbf{r}^*\|_1 \le 2 d^t
   $$
3. The step-to-step convergence residual satisfies:
   $$
   \|\mathbf{r}^{(t+1)} - \mathbf{r}^{(t)}\|_1 \le d \|\mathbf{r}^{(t)} - \mathbf{r}^{(t-1)}\|_1 \le d^t \|\mathbf{r}^{(1)} - \mathbf{r}^{(0)}\|_1
   $$
4. To guarantee $\|\mathbf{r}^{(t+1)} - \mathbf{r}^{(t)}\|_1 < \epsilon$ for $\epsilon = 10^{-10}$ with $d = 0.85$ and uniform initialization $\mathbf{r}^{(0)} = \frac{1}{N} \mathbf{e}$, the required number of iterations $t^*$ is bounded by:
   $$
   t^* \le \left\lceil \frac{\ln(\epsilon / 2)}{\ln(d)} \right\rceil = \left\lceil \frac{\ln(10^{-10} / 2)}{\ln(0.85)} \right\rceil = \lceil 146.01 \rceil = 147
   $$

*Proof.*  
1. **Contraction on $V_0$**:  
   Let $\mathbf{x} \in V_0$, so $\mathbf{e}^T \mathbf{x} = \sum_{j=1}^N x_j = 0$.  
   Then:
   $$
   \mathbf{\tilde{P}} \mathbf{x} = d \mathbf{M} \mathbf{x} + \frac{1-d}{N} \mathbf{e} (\mathbf{e}^T \mathbf{x}) = d \mathbf{M} \mathbf{x}
   $$
   We decompose $\mathbf{x}$ into positive and negative components: $\mathbf{x} = \mathbf{x}^+ - \mathbf{x}^-$, where $x_j^+ = \max(x_j, 0)$ and $x_j^- = \max(-x_j, 0)$.  
   Since $\mathbf{e}^T \mathbf{x} = 0$, we have $\mathbf{e}^T \mathbf{x}^+ = \mathbf{e}^T \mathbf{x}^- = \frac{1}{2} \|\mathbf{x}\|_1$.  
   Then:
   $$
   \|\mathbf{\tilde{P}} \mathbf{x}\|_1 = d \|\mathbf{M} \mathbf{x}^+ - \mathbf{M} \mathbf{x}^-\|_1
   $$
   Using the triangle inequality:
   $$
   \|\mathbf{\tilde{P}} \mathbf{x}\|_1 \le d \left( \|\mathbf{M} \mathbf{x}^+\|_1 + \|\mathbf{M} \mathbf{x}^-\|_1 \right)
   $$
   Since $\mathbf{M}$ is column-stochastic and $\mathbf{x}^+ \ge \mathbf{0}, \mathbf{x}^- \ge \mathbf{0}$:
   $$
   \|\mathbf{M} \mathbf{x}^+\|_1 = \mathbf{e}^T \mathbf{M} \mathbf{x}^+ = \mathbf{e}^T \mathbf{x}^+ = \frac{1}{2} \|\mathbf{x}\|_1
   $$
   $$
   \|\mathbf{M} \mathbf{x}^-\|_1 = \mathbf{e}^T \mathbf{M} \mathbf{x}^- = \mathbf{e}^T \mathbf{x}^- = \frac{1}{2} \|\mathbf{x}\|_1
   $$
   Therefore:
   $$
   \|\mathbf{\tilde{P}} \mathbf{x}\|_1 \le d \left( \frac{1}{2} \|\mathbf{x}\|_1 + \frac{1}{2} \|\mathbf{x}\|_1 \right) = d \|\mathbf{x}\|_1
   $$

2. **Distance to Stationary Vector**:  
   Let $\mathbf{r}^{(0)} \in \Delta^N$. Since $\mathbf{e}^T \mathbf{r}^{(0)} = 1$ and $\mathbf{e}^T \mathbf{r}^* = 1$, the difference vector $\mathbf{x}^{(t)} = \mathbf{r}^{(t)} - \mathbf{r}^*$ satisfies $\mathbf{e}^T \mathbf{x}^{(t)} = 0$, so $\mathbf{x}^{(t)} \in V_0$.  
   By induction on the contraction property:
   $$
   \|\mathbf{r}^{(t)} - \mathbf{r}^*\|_1 = \|\mathbf{\tilde{P}}^t (\mathbf{r}^{(0)} - \mathbf{r}^*)\|_1 \le d^t \|\mathbf{r}^{(0)} - \mathbf{r}^*\|_1
   $$
   Since $\mathbf{r}^{(0)}, \mathbf{r}^* \in \Delta^N$:
   $$
   \|\mathbf{r}^{(0)} - \mathbf{r}^*\|_1 \le \|\mathbf{r}^{(0)}\|_1 + \|\mathbf{r}^*\|_1 = 1 + 1 = 2
   $$
   Thus:
   $$
   \|\mathbf{r}^{(t)} - \mathbf{r}^*\|_1 \le 2 d^t
   $$

3. **Step-to-Step Delta**:  
   Similarly, $\mathbf{r}^{(t+1)} - \mathbf{r}^{(t)} = \mathbf{\tilde{P}} (\mathbf{r}^{(t)} - \mathbf{r}^{(t-1)})$. Since $\mathbf{e}^T (\mathbf{r}^{(t)} - \mathbf{r}^{(t-1)}) = 1 - 1 = 0$, the vector lies in $V_0$.  
   Applying the contraction bound yields:
   $$
   \|\mathbf{r}^{(t+1)} - \mathbf{r}^{(t)}\|_1 \le d \|\mathbf{r}^{(t)} - \mathbf{r}^{(t-1)}\|_1 \le d^t \|\mathbf{r}^{(1)} - \mathbf{r}^{(0)}\|_1 \le 2 d^t
   $$

4. **Iteration Bound for $\epsilon = 10^{-10}$**:  
   Setting $2 d^{t^*} < \epsilon$:
   $$
   d^{t^*} < \frac{\epsilon}{2} \implies t^* \ln(d) < \ln\left(\frac{\epsilon}{2}\right)
   $$
   Since $d < 1$, $\ln(d) < 0$, dividing by $\ln(d)$ reverses the inequality:
   $$
   t^* > \frac{\ln(\epsilon / 2)}{\ln(d)}
   $$
   For $d = 0.85$ and $\epsilon = 10^{-10}$:
   $$
   t^* > \frac{\ln(10^{-10} / 2)}{\ln(0.85)} = \frac{\ln(5 \times 10^{-11})}{-0.1625189} = \frac{-23.71899}{-0.1625189} \approx 145.945 \implies t^* = 146 \quad \blacksquare
   $$

---

## 4. Algorithmic Pseudocode and Reference Implementations

### 4.1. Sparse Graph Power Iteration Pseudocode

```text
Algorithm: Sparse-Spectral-PageRank
Input:
  - V: Set of N node identifiers (paths/keys)
  - Outlinks: Map from each node u in V to list of targets in V (self-links excluded)
  - d: Damping factor (default = 0.85)
  - epsilon: L1 convergence tolerance (default = 1e-10)
  - max_iter: Maximum iteration limit (default = 200)

Output:
  - r: Map from node identifier to stationary PageRank value (summing to 1.0)

1. N <- |V|
2. If N == 0: return empty Map
3. If N == 1: return {V[0]: 1.0}

4. Initialize rank map r:
     For each u in V:
       r[u] <- 1.0 / N

5. Precompute out-degrees and identify dangling nodes:
     dangling_nodes <- []
     For each u in V:
       If |Outlinks[u]| == 0:
         dangling_nodes.append(u)

6. For iter = 1 to max_iter:
     // Step A: Aggregate dangling mass
     dangling_mass <- 0.0
     For each u in dangling_nodes:
       dangling_mass <- dangling_mass + r[u]

     // Step B: Compute baseline contribution (teleport + dangling redistribution)
     base <- (1.0 - d) / N + (d * dangling_mass) / N

     // Step C: Initialize next iteration buffer
     next_r <- Map()
     For each u in V:
       next_r[u] <- base

     // Step D: Propagate outbound links
     For each u in V:
       k <- |Outlinks[u]|
       If k > 0:
         share <- (d * r[u]) / k
         For each target in Outlinks[u]:
           next_r[target] <- next_r[target] + share

     // Step E: Compute L1 delta
     delta <- 0.0
     For each u in V:
       delta <- delta + |next_r[u] - r[u]|

     r <- next_r

     If delta < epsilon:
       break

7. // Step F: Exact normalization against floating point drift
   total_mass <- sum(r[u] for u in V)
   For each u in V:
     r[u] <- r[u] / total_mass

8. Return r
```

### 4.2. TypeScript Reference Implementation (Wikifita `lib/wiki.ts` Parity)

```typescript
/**
 * Computes PageRank over an incomplete directed document graph.
 * Strictly adheres to Wikifita Atlas production implementation.
 *
 * @param paths Array of unique document identifiers
 * @param outlinksByPath Map from document path to resolved destination paths
 * @param damping Teleportation damping factor (default 0.85)
 * @param tolerance L1 convergence delta threshold (default 1e-10)
 * @param maxIterations Maximum power iteration step limit (default 200)
 * @returns Map of path to stationary probability mass
 */
export function computePagerank(
  paths: string[],
  outlinksByPath: Map<string, string[]>,
  damping: number = 0.85,
  tolerance: number = 1e-10,
  maxIterations: number = 200
): Map<string, number> {
  const n = paths.length;
  if (n === 0) return new Map();
  if (n === 1) return new Map([[paths[0], 1.0]]);

  // 1. Uniform Initialization
  let rank = new Map<string, number>();
  const initialRank = 1.0 / n;
  for (const path of paths) {
    rank.set(path, initialRank);
  }

  // Pre-filter dangling node set
  const danglingPaths: string[] = [];
  for (const path of paths) {
    const out = outlinksByPath.get(path);
    if (!out || out.length === 0) {
      danglingPaths.push(path);
    }
  }

  // 2. Power Iteration Loop
  for (let iter = 0; iter < maxIterations; iter++) {
    // 2a. Accumulate dangling mass
    let danglingMass = 0.0;
    for (const dPath of danglingPaths) {
      danglingMass += rank.get(dPath) ?? 0.0;
    }

    // 2b. Compute uniform base injection
    const base = (1.0 - damping) / n + (damping * danglingMass) / n;
    const nextRank = new Map<string, number>();
    for (const path of paths) {
      nextRank.set(path, base);
    }

    // 2c. Scatter rank mass across outlinks
    for (const path of paths) {
      const out = outlinksByPath.get(path);
      if (out && out.length > 0) {
        const currentRank = rank.get(path) ?? 0.0;
        const share = (damping * currentRank) / out.length;
        for (const target of out) {
          if (nextRank.has(target)) {
            nextRank.set(target, (nextRank.get(target) ?? 0.0) + share);
          }
        }
      }
    }

    // 2d. Evaluate L1 convergence norm
    let delta = 0.0;
    for (const path of paths) {
      const prev = rank.get(path) ?? 0.0;
      const curr = nextRank.get(path) ?? 0.0;
      delta += Math.abs(curr - prev);
    }

    rank = nextRank;

    if (delta < tolerance) {
      break;
    }
  }

  // 3. Exact Simplex Normalization
  let totalMass = 0.0;
  for (const path of paths) {
    totalMass += rank.get(path) ?? 0.0;
  }

  const normalizedRank = new Map<string, number>();
  for (const path of paths) {
    normalizedRank.set(path, (rank.get(path) ?? 0.0) / totalMass);
  }

  return normalizedRank;
}
```

### 4.3. High-Performance Python / NumPy Implementation

```python
"""
High-Performance Sparse Matrix PageRank with Dangling Mass Redistribution.
Provides exact mathematical parity with Wikifita and Bradley-Terry scaling foundations.
"""

from typing import Dict, List, Tuple
import numpy as np
import scipy.sparse as sp


def compute_pagerank_numpy(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    damping: float = 0.85,
    tolerance: float = 1e-10,
    max_iter: int = 200,
) -> Dict[str, float]:
    """Compute PageRank using vector NumPy operations over sparse matrices.

    Args:
        nodes: Unique list of node identifiers.
        edges: Directed pairs (source, target).
        damping: Damping factor d (default 0.85).
        tolerance: L1 delta termination threshold (default 1e-10).
        max_iter: Maximum iteration steps (default 200).

    Returns:
        Dictionary mapping node identifier to normalized stationary rank.
    """
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: 1.0}

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    # Build adjacency matrix in CSR format
    row_indices = []
    col_indices = []
    out_degrees = np.zeros(n, dtype=np.float64)

    for src, dst in edges:
        if src in node_to_idx and dst in node_to_idx and src != dst:
            u = node_to_idx[src]
            v = node_to_idx[dst]
            row_indices.append(v)  # Target is row in column-stochastic P
            col_indices.append(u)  # Source is column
            out_degrees[u] += 1.0

    dangling_mask = out_degrees == 0

    # Normalization weights for nonzero columns
    data = np.zeros(len(row_indices), dtype=np.float64)
    for i, u in enumerate(col_indices):
        data[i] = 1.0 / out_degrees[u]

    # Sparse transition matrix P (shape N x N, column-substochastic)
    p_sparse = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))

    # Initial probability vector
    r = np.full(n, 1.0 / n, dtype=np.float64)
    teleport_base = (1.0 - damping) / n

    for _ in range(max_iter):
        # 1. Compute dangling mass
        dangling_mass = np.sum(r[dangling_mask])

        # 2. Sparse matrix-vector product
        r_next = damping * p_sparse.dot(r)

        # 3. Add base teleport and dangling redistribution
        r_next += teleport_base + (damping * dangling_mass) / n

        # 4. L1 convergence check
        delta = np.sum(np.abs(r_next - r))
        r = r_next

        if delta < tolerance:
            break

    # Exact simplex projection
    r /= np.sum(r)

    return {nodes[i]: float(r[i]) for i in range(n)}
```

---

## 5. Downstream Integration Synthesis

### 5.1. Integration with Section 3 (Bradley-Terry & Abelian Calibration)
The spectral guarantees established in Section 2 form the formal counterpart to the Bradley-Terry Abelian group shift in Section 3:
- **Dangling mass redistribution** maps directly to **Softmax Abelian translation ($\Delta R_{\text{Abeliano}}$)**: both operators reconstruct missing global information by performing a rank-1 scalar redistribution over the invariant manifold.
- **Uniform teleportation ($1-d$)** maps directly to **Bayesian prior shrinkage (MD10 with $N_0 = 10$)**: both ensure non-zero support and prevent infinite variance or singular log-odds when graph connectivity is sparse.

### 5.2. Monograph Insertion Readiness
This complete mathematical formulation is structured for direct modular integration into `docs/pagerank_and_abelian_graph_invariance.md`, establishing an IEEE/SIAM-grade research standard.
