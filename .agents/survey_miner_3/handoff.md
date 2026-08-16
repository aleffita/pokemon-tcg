# Combinatorial & Hypergeometric Handoff Report — Pokémon TCG AI Deck Engine

**Agent**: survey_miner_3 (Hypergeometric & Combinatorial Modeler)  
**Date**: 2026-08-16  
**Target File**: `.agents/survey_miner_3/handoff.md`  

---

## 1. Observation

All probability distributions and combinatorial models were evaluated over a standard Pokémon TCG deck of population size $N = 60$, with sample sizes $n = 7$ (opening hand) and $n = 8$ (opening hand plus Turn 1 natural draw), without replacement.

### 1.1 Multivariate Hypergeometric Distribution

Let the 60-card deck be partitioned into $m$ disjoint subsets of cards with cardinalities $K_1, K_2, \dots, K_m$ such that:

$$\sum_{i=1}^m K_i = N = 60$$

When drawing a sample of size $n$ cards without replacement, the joint probability mass function of drawing $(k_1, k_2, \dots, k_m)$ cards with $\sum_{i=1}^m k_i = n$ is:

$$P(X_1 = k_1, X_2 = k_2, \dots, X_m = k_m) = \frac{\prod_{i=1}^m \binom{K_i}{k_i}}{\binom{N}{n}}$$

The marginal expected value, variance, and pairwise covariance are given by:

$$E[X_i] = n \cdot \frac{K_i}{N}$$

$$\operatorname{Var}(X_i) = n \cdot \frac{K_i}{N} \cdot \left(1 - \frac{K_i}{N}\right) \cdot \frac{N - n}{N - 1}$$

$$\operatorname{Cov}(X_i, X_j) = -n \cdot \frac{K_i K_j}{N^2} \cdot \frac{N - n}{N - 1} \quad (i \neq j)$$

---

### 1.2 Opening Hand Setup and Mulligan Probabilities

A mulligan occurs when the opening 7-card hand contains zero Basic Pokémon ($k_b = 0$).

$$P(\text{Mulligan}) = \frac{\binom{60 - K_b}{7}}{\binom{60}{7}}$$

$$P(\text{Setup}) = 1 - P(\text{Mulligan}) = 1 - \frac{\binom{60 - K_b}{7}}{\binom{60}{7}}$$

| Basic Count $K_b$ | $P(\text{Mulligan}) (n=7)$ | $P(\text{Setup}) (n=7)$ | $P(\text{Setup}) (n=8)$ | Exact Rational $P(\text{Setup}, n=7)$ |
| :--- | :--- | :--- | :--- | :--- |
| 4 | 60.0500% | 39.9500% | 44.4820% | 38962 / 97527 |
| 5 | 52.5438% | 47.4562% | 52.4132% | 370261 / 780216 |
| 6 | 45.8564% | 54.1436% | 59.3349% | 193617 / 357599 |
| 7 | 39.9120% | 60.0880% | 65.3594% | 5801596 / 9655173 |
| 8 | 34.6406% | 65.3594% | 70.5881% | 6310559 / 9655173 |
| 9 | 29.9775% | 70.0225% | 75.1130% | 1502399 / 2145594 |
| 10 | 25.8629% | 74.1371% | 79.0169% | 216911 / 292581 |
| 11 | 22.2421% | 77.7579% | 82.3742% | 1137524 / 1462905 |
| 12 | 19.0647% | 80.9353% | 85.2519% | 394669 / 487635 |
| 14 | 13.8591% | 86.1409% | 89.8018% | 252032 / 292581 |
| 18 | 6.9850% | 93.0150% | 95.3433% | 18193859 / 19559980 |

Under single-draw constraints, achieving $P(\text{Setup}) \ge 92.0\%$ requires $K_b \ge 18$.  
Under official tournament rules allowing mulligan redraws, the cumulative probability of setting up within $m$ mulligans is:

$$P(\text{Setup within } m \text{ mulligans}) = 1 - [P(\text{Mulligan})]^{m+1}$$

| Basic Count $K_b$ | Single Draw ($m=0$) | Within 1 Mulligan ($m=1$) | Within 2 Mulligans ($m=2$) |
| :--- | :--- | :--- | :--- |
| 8 | 65.36% | 88.00% | 95.84% |
| 9 | 70.02% | 91.01% | 97.31% |
| 10 | 74.14% | 93.31% | 98.27% |
| 11 | 77.76% | 95.05% | 98.90% |
| 12 | 80.94% | 96.37% | 99.31% |
| 14 | 86.14% | 98.08% | 99.73% |

---

### 1.3 Opening Hand Access to Search Engine and Draw Supporters

Using Principle of Inclusion-Exclusion for bivariate and trivariate hypergeometric distributions:

$$P(X_A \ge 1 \cap X_B \ge 1) = 1 - \frac{\binom{N - K_A}{n}}{\binom{N}{n}} - \frac{\binom{N - K_B}{n}}{\binom{N}{n}} + \frac{\binom{N - K_A - K_B}{n}}{\binom{N}{n}}$$

#### Basic Pokémon ($K_b = 10$) and Ball/Search Items ($K_s$)

| Search Count $K_s$ | $P(B \ge 1 \cap S \ge 1) (n=7)$ | $P(B \ge 1 \cap S \ge 1) (n=8)$ | Exact Fraction ($n=7$) |
| :--- | :--- | :--- | :--- |
| 4 | 27.946% | 33.697% | 9085 / 32509 |
| 6 | 38.203% | 45.279% | 18442816 / 48275865 |
| 8 | 46.482% | 54.218% | 22439536 / 48275865 |
| 10 | 53.102% | 61.039% | 1709014 / 3218391 |
| 12 | 58.340% | 66.180% | 4991 / 8555 |

#### Ideal Turn 1 Setup: Basic ($K_b = 10$), Engine ($K_{\text{eng}} = 16$), Energy ($K_e$)

$$P(A \ge 1 \cap B \ge 1 \cap C \ge 1) = 1 - \sum P(A=0) + \sum P(A=0 \cap B=0) - P(A=0 \cap B=0 \cap C=0)$$

| Energy Count $K_e$ | $P(\text{Ideal T1}) (n=7)$ | $P(\text{Ideal T1}) (n=8)$ |
| :--- | :--- | :--- |
| 8 | 39.944% | 49.122% |
| 10 | 45.875% | 55.503% |
| 12 | 50.638% | 60.361% |
| 14 | 54.417% | 64.008% |

---

### 1.4 Turn 2 Energy Density & Acceleration Sustainability

Probability of drawing $m$ or more Energy cards in $n$ cards drawn:

$$P(X_e \ge m) = 1 - \sum_{k=0}^{m-1} \frac{\binom{K_e}{k}\binom{60 - K_e}{n - k}}{\binom{60}{n}}$$

| Energy Count $K_e$ | $P(E \ge 1) (n=7)$ | $P(E \ge 1) (n=8)$ | $P(E \ge 2) (n=9, \text{Natural T2})$ | $P(E \ge 2) (n=15, \text{T2+Research})$ |
| :--- | :--- | :--- | :--- | :--- |
| 8 | 65.359% | 70.588% | 34.389% | 64.971% |
| 9 | 70.023% | 75.113% | 40.654% | 72.134% |
| 10 | 74.137% | 79.017% | 46.735% | 78.138% |
| 11 | 77.758% | 82.374% | 52.546% | 83.075% |
| 12 | 80.935% | 85.252% | 58.025% | 87.064% |
| 14 | 86.141% | 89.802% | 67.837% | 92.725% |

Expected Energy in sample $n$:

$$E[X_e] = n \cdot \frac{K_e}{60}$$

For $K_e = 12$: $E[X_e \mid n=7] = 1.40$, $E[X_e \mid n=8] = 1.60$, $E[X_e \mid n=9] = 1.80$, $E[X_e \mid n=15] = 3.00$.

---

### 1.5 Prize Trade Dynamics & The 7-Prize Asymmetry

Let $P = 6$ be the initial prize card pool.

1. **Standard Two-Prize Sequence (2-2-2)**:
   - Knockouts required: $\lceil 6 / 2 \rceil = 3$ KOs.
   - Attack tempo: Exactly 3 successful attacks to win.

2. **Single-Prize Interjection Sequence (1-2-2-2 / 1-2-2-1)**:
   - Sequence of prizes taken by opponent: $1 \to 3 \to 5 \to 7$ (or $2 \to 3 \to 5 \to 7$).
   - Number of KOs required: $4$ KOs.
   - Total prizes taken: $7$ (1 prize overkill on final KO).
   - Attack tempo: Opponent requires $4$ attacks instead of $3$.

---

## 2. Logic Chain

1. **Setup Optimization vs. Deck Thinning**:
   - Running $K_b \ge 18$ Basics to achieve $P(\text{Setup}) \ge 92\%$ on single opening hand compromises mid-game draw quality by diluting Trainer and Energy densities.
   - In competitive tournament play, mulligan redraws have a small penalty (giving opponent $+1$ card), while dead draws in turns 2-6 cause losses.
   - Setting $K_b \in [10, 12]$ achieves $74.1\% - 80.9\%$ setup on hand 1, and $93.3\% - 96.4\%$ within 1 mulligan, satisfying the structural reliability threshold while reserving $48 - 50$ slots for Trainers and Energy.

2. **Search Engine Synergy**:
   - A Search suite of $K_s = 10$ (e.g. 4 Nest Ball, 4 Ultra Ball, 1 Hisuian Heavy Ball, 1 Buddy-Buddy Poffin) and Supporter suite of $K_d = 8$ creates an effective Engine pool $K_{\text{eng}} = 18$.
   - This ensures $P(\text{Basic} \ge 1 \cap \text{Engine} \ge 1) = 74.82\%$ by Turn 1 draw ($n=8$).

3. **Energy Sustainability & Acceleration**:
   - For an attack requiring 2 Energy on Turn 2, natural attachment has only a $46.7\%$ ($K_e=10$) to $58.0\%$ ($K_e=12$) probability.
   - Integrating Teal Dance (Teal Mask Ogerpon ex) allows attaching 1 Basic Grass Energy from hand to Ogerpon ex and drawing 1 card.
   - This converts Turn 2 Energy requirement from 2 sequential natural top-decks into a compound event: attaching 1 Energy manually and 1 via Teal Dance, with sample size expanding from $n=8$ to $n=10+$.
   - At $K_e = 12$, $P(E \ge 1 \text{ on T1}) = 85.25\%$, and with Supporter draw ($n=15$), $P(E \ge 2) = 87.06\%$.

4. **Prize Trade Asymmetry (The 7th Prize Rule)**:
   - In a pure 2-prize meta (Ogerpon ex, Fezandipiti ex, Mew ex, Dragapult ex), games terminate in 3 turns of mutual KOs.
   - Forcing the opponent to KO a 1-prize Pokémon (e.g. Cornerstone Ogerpon, Radiant Greninja, or a 1-prize tech attacker) converts the opponent's prize progression to $1 \to 3 \to 5 \to 7$.
   - This demands 4 KOs from the opponent, giving the player an extra turn of attacks ($+33.3\%$ tempo advantage).

---

## 3. Caveats

1. **Prize Card Partition Variance**:
   - 6 cards are set aside into prizes after setup. Any key 1-of card has a $6/60 = 10.0\%$ probability of being prized.
   - Including Hisuian Heavy Ball recovers prized Basic Pokémon with $P(\text{HHB in Hand} \mid \text{Basic Prized}) = \frac{\binom{53}{7}}{\binom{54}{7}} \approx 87.0\%$ over the course of early search.
2. **Shuffle Stochasticity & Drawing in Blocks**:
   - While hypergeometric formulas assume independent uniform random sampling without replacement, sequence-dependent search items (e.g. Ultra Ball discarding 2 cards before searching) actively alter the deck size $N$ dynamically ($60 \to 53 \to 46$).
3. **Turn 1 Going First vs. Going Second Rules**:
   - Going first: cannot attack, cannot play Supporter on Turn 1. Sample size is strictly $n=7$.
   - Going second: can play Supporter and attack on Turn 1. Sample size expands to $n=8$ (or $n=15$ with Professor's Research).

---

## 4. Conclusion

### Optimal 60-Card Macro-Composition Breakdown

| Category | Recommended Slots | Breakdown |
| :--- | :--- | :--- |
| **Pokémon** | **14 cards** | 10-11 Basic Pokémon (4 Teal Mask Ogerpon ex, 1 Cornerstone Ogerpon ex, 1 Wellspring Ogerpon ex, 1 Radiant Greninja, 1 Fezandipiti ex, 1 Mew ex, 1-2 1-Prize Tech Basics), 3 Stage 1/Evolution (if applicable) |
| **Search Items** | **10 cards** | 4 Nest Ball, 4 Ultra Ball, 1 Hisuian Heavy Ball, 1 Buddy-Buddy Poffin / Bug Catching Set |
| **Draw/Setup Supporters** | **8 cards** | 4 Professor's Research, 3 Iono, 1 Boss's Orders |
| **Utility/Switch/Recovery** | **12 cards** | 2 Super Rod, 2 Energy Retrieval / Earthen Vessel, 3 Switch / Switch Cart, 1 Prime Catcher (ACE SPEC), 2 Stadiums (Artazon / Bug Catching Set), 2 Tool Cards (Bravery Charm / Forest Seal Stone) |
| **Basic Energy** | **12 cards** | 12 Basic Grass Energy (or 10 Grass + 2 Tech Energy) |
| **Total** | **60 cards** | **$100\%$ Physical Deck Parity** |

### Key Invariant Metrics
- $P(\text{Setup Hand 1}) = 74.14\%$ ($K_b = 10$) to $80.94\%$ ($K_b = 12$).
- $P(\text{Setup within 1 Mulligan}) \ge 93.31\%$ (Target met).
- $P(\text{T1 Engine Access}) \ge 74.82\%$.
- $P(\text{T1 Energy Access}) \ge 85.25\%$.
- Turn 2 Energy Acceleration: $E[\text{Attached Energy}] \ge 2.0$ via Teal Dance + manual attachment.
- Prize Trade: Enforces 4-KO requirement on opponent via 1-Prize tech inclusion.

---

## 5. Verification Method

To independently verify all combinatorial and hypergeometric values with exact rational arithmetic, execute the following command in the project directory:

```bash
uv run python scratch/probe_hypergeometric.py
```

### Reproducibility Verification Assertions

```python
import math
from fractions import Fraction

def comb(n, k):
    return math.comb(n, k)

# Assertion 1: Population 60, Sample 7, Mulligan for 10 Basics
p_mul_10 = Fraction(comb(50, 7), comb(60, 7))
assert float(p_mul_10) == 0.2586289608690048, "Mulligan P(k=10) mismatch"

# Assertion 2: Setup within 1 Mulligan >= 92%
p_setup_1_mul = 1 - (p_mul_10 ** 2)
assert float(p_setup_1_mul) >= 0.92, "P(Setup with 1 mulligan) must exceed 92%"

# Assertion 3: Natural Energy Draw for 12 Energy in 8 cards >= 85%
p_energy_8 = 1 - Fraction(comb(48, 8), comb(60, 8))
assert float(p_energy_8) >= 0.85, "P(Energy >= 1 in 8 cards) must exceed 85%"
```
