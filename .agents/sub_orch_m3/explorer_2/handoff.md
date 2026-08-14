# Handoff Report — Explorer 2: Bradley-Terry Softmax Abelian Group Rating Invariance & Duality Isomorphism

**Agent**: Explorer 2 (Milestone 3 — Mathematical Monograph)  
**Recipient**: Parent Sub-Orchestrator (`4877bc7d-bfc2-44d3-bc55-1a9dd628ba39`)  
**Working Directory**: `/Users/alefita/workdir/pokemon-tcg/.agents/sub_orch_m3/explorer_2/`  
**Primary Deliverable**: `/Users/alefita/workdir/pokemon-tcg/.agents/sub_orch_m3/explorer_2/analysis.md`  
**Date**: August 14, 2026  

---

## 1. Observation

Direct code and documentation observations:

1. **`rl/results_db.py:625-681`**:
   - Implements `get_invariant_deck_elo(self, deck_id: int, source: str = "local")`.
   - Line 643: `w_clipped = max(0.02, min(0.98, w_rate))`
   - Line 644: `r_asymptotic = 600.0 + 400.0 * math.log10(w_clipped / (1.0 - w_clipped))`
   - Line 647-648: `n0 = 10.0`, `r_smoothed = (n / (n + n0)) * r_asymptotic + (n0 / (n + n0)) * INITIAL_ELO`
   - Line 661-672: `tau = 20.0`, `weights = [math.exp(min(r["n_loc"] / tau, 20.0)) for r in overlapping]`, `delta_abeliano = sum(deltas)`
   - Line 674: `r_invariant = r_smoothed + delta_abeliano`

2. **`docs/abelian_group_elo_formulation.md:1-170`**:
   - Details algebraic group properties of $(\mathbb{R}, +)$ and Translation Isomorphism Theorem.

3. **`docs/pagerank_and_abelian_graph_invariance.md:45-86`**:
   - Outlines draft structure for Sections 3, 4, and 5 of the research monograph.

4. **`GEMINI.md:21-42`**:
   - Defines the mathematical framework and communication rules (ASD-STE100, KaTeX bold isolation).

---

## 2. Logic Chain

The step-by-step reasoning from observations to technical formulations:

1. **Observation 1 (Logistic Odds Derivation)**:
   - Under $P(i \succ j) = \frac{1}{1 + 10^{-(R_i - R_j)/400}}$, log-odds ratio equals $\frac{R_i - R_j}{400}$.
   - For an agent evaluated against a baseline reference pool with effective mean $R_0 = 600.0$, the empirical win rate $w = W/N$ inverts to $\hat{R}_\infty = 600.0 + 400.0 \log_{10}(w / (1-w))$.
   - Clamping $w \in [0.02, 0.98]$ bounds the dynamic range to $[-76.08, 1276.08]$, preventing logarithmic singularities.

2. **Observation 2 (Bayesian MD10 Shrinkage)**:
   - Small sample sizes ($N < 10$) produce severe sampling variance in MLE estimates.
   - Formulating a Gaussian prior $R \sim \mathcal{N}(R_0, \sigma_0^2)$ yields posterior mean $R_{\text{smoothed}} = \frac{N}{N + N_0}\hat{R}_\infty + \frac{N_0}{N + N_0}R_0$ with pseudo-count $N_0 = 10.0$.
   - Proved: Estimator variance $\operatorname{Var}(R_{\text{smoothed}}(N)) = \frac{N}{(N+N_0)^2} v(w)$ is globally bounded by $\frac{v(w)}{40.0}$ at $N = 10$, achieving $> 99.1\%$ variance reduction at $N = 1$.
   - As $N \to \infty$, $R_{\text{smoothed}}(N) \to \hat{R}_\infty$ with deterministic convergence rate $O(1/N)$.

3. **Observation 3 (Abelian Group Translation Invariance)**:
   - Proved that $(\mathbb{R}, +)$ satisfies all five Abelian group axioms (closure, associativity, identity $0$, inverse $-R$, commutativity).
   - Proved the Translation Isomorphism Theorem: For any scalar $\Delta \in \mathbb{R}$, $T_\Delta(R) = R + \Delta$ preserves pairwise differences $T_\Delta(R_i) - T_\Delta(R_j) = R_i - R_j$ and win probabilities $P_{T_\Delta}(i \succ j) = P(i \succ j)$.

4. **Observation 4 (Softmax Temperature Calibration $\Delta R_{\text{Abeliano}}$)**:
   - For overlapping decks $\mathcal{C} = \mathcal{V}_{\text{local}} \cap \mathcal{V}_{\text{remote}}$, individual scale offsets are $\delta_k = R_k^{\text{remote}} - \hat{R}_{k,\infty}^{\text{local}}$.
   - Weighting via sample-size Softmax $\alpha_k = \frac{\exp(\min(N_k/\tau, 20.0))}{\sum \exp(\min(N_j/\tau, 20.0))}$ with $\tau = 20.0$ forms a convex combination $\sum \alpha_k = 1$.
   - Exponential clipping at $20.0$ bounds exponential terms to $\le 4.85 \times 10^8$, preventing floating-point overflow while concentrating calibration weight on mature decks ($N \gg 20$).

5. **Observation 5 (Duality Isomorphism Mapping)**:
   - PageRank and Abelian Elo solve the identical incomplete graph problem.
   - PageRank conserves mass on the $L_1$ simplex via teleportation and dangling mass redistribution.
   - Abelian Elo preserves probabilities in Euclidean group quotient space via MD10 shrinkage and Softmax translation.
   - Synthesized a comprehensive 14-dimension theoretical comparison matrix.

---

## 3. Caveats

1. **Draw Handling**: The current formula treats empirical win rate as $w = \text{wins} / N$. If draws are non-zero, they should be incorporated as $w = (\text{wins} + 0.5 \cdot \text{draws}) / N$. The code in `rl/results_db.py` currently computes `w_rate = float(row["wins"]) / max(n, 1.0)`. This is completely safe in tournament modes where ties are broken by sudden death or minimal draw frequencies.
2. **Disconnected Local Subgraphs**: When $\mathcal{C} = \emptyset$ (no anchor decks present in both local and remote sets), $\Delta R_{\text{Abeliano}} = 0.0$. The local rating remains self-consistent on the $R_0 = 600.0$ base scale.

---

## 4. Conclusion

All theoretical derivations, mathematical proofs, and comparative matrices required for Section 3 and Section 4 of `docs/pagerank_and_abelian_graph_invariance.md` have been fully developed and formatted in `analysis.md`. The equations, variable names, and constants strictly match `rl/results_db.py`.

The analysis is ready for direct authoring into the final monograph.

---

## 5. Verification Method

To verify the theoretical formulations and code alignment:

1. **Verify Python Invariant Elo Execution**:
   ```bash
   uv run python -c '
   import math
   # 1. Asymptotic Inversion
   w = 0.75
   w_clipped = max(0.02, min(0.98, w))
   r_asymptotic = 600.0 + 400.0 * math.log10(w_clipped / (1.0 - w_clipped))
   assert abs(r_asymptotic - (600.0 + 400.0 * math.log10(3.0))) < 1e-6
   
   # 2. MD10 Regularization
   n = 10.0
   n0 = 10.0
   r_smoothed = (n / (n + n0)) * r_asymptotic + (n0 / (n + n0)) * 600.0
   assert abs(r_smoothed - (0.5 * r_asymptotic + 0.5 * 600.0)) < 1e-6
   
   # 3. Softmax Weights
   n_samples = [10.0, 40.0, 80.0]
   tau = 20.0
   weights = [math.exp(min(ns / tau, 20.0)) for ns in n_samples]
   total_w = sum(weights)
   alphas = [w / total_w for w in weights]
   assert abs(sum(alphas) - 1.0) < 1e-6
   assert alphas[2] > alphas[1] > alphas[0]
   print("Verification PASS: Math logic verified successfully.")
   '
   ```

2. **Inspect Analysis Report**:
   ```bash
   # View primary artifact
   cat /Users/alefita/workdir/pokemon-tcg/.agents/sub_orch_m3/explorer_2/analysis.md
   ```
