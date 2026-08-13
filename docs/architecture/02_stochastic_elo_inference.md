# Stochastic Elo Inference in Ephemeral Sandboxes

## 1. Abstract
The Kaggle execution sandbox is ephemeral (memoryless). The agent cannot persist data to disk between matches. To maintain situational awareness of its Global Rank, the agent must derive its current Elo stochastically at the start of each match using Bayesian Priors and the Bradley-Terry model.

## 2. Variables & Constants
- **$R_0$:** Anchor Elo (Hardcoded base rating from the August 16th submission).
- **$T_0$:** Anchor Timestamp (August 16th, 00:00 UTC).
- **$T_{now}$:** Current OS Timestamp (`time.time()`).
- **$\Delta T$:** Days elapsed since anchor ($T_{now} - T_0$).
- **$V$:** Expected match volume per day (Derived from historical data, approx. 15-20 games/day per agent).
- **$N_k$:** Expected number of games played so far ($N_k = \Delta T \times V$).
- **$\hat{R}_{opp}$:** Opponent's Elo estimated in-game via Aux Heads (Opponent Modeling).

## 3. Stochastic Derivation
The agent cannot know the exact outcome of the past $N_k$ matches. However, assuming the agent plays optimally against the environment distribution, its Expected Win Rate ($E_{win}$) over the $N_k$ matches can be modeled against the average ladder Elo $\mu_{ladder}$:

$$ E_{win} = \frac{1}{1 + 10^{(\mu_{ladder} - R_0) / 400}} $$

The expected Elo drift ($\Delta R$) over $N_k$ matches, given an average K-factor ($K=20$), is approximated by:
$$ \Delta R \approx N_k \times K \times (W_{actual} - E_{win}) $$

Since $W_{actual}$ is unknown, the agent relies on the **Opponent Sampling Theorem**. The Kaggle matchmaking algorithm pairs agents with similar Elo. Therefore, the estimated Elo of the current opponent ($\hat{R}_{opp}$) serves as a highly correlated proxy for the agent's current Elo:
$$ R_{current} \sim \mathcal{N}(\hat{R}_{opp}, \sigma^2) $$

## 4. Synthesis for MoE Routing
The router concatenates the prior mathematically:
$$ R_{internal} = \alpha (R_0 + f(\Delta T)) + (1 - \alpha) \hat{R}_{opp} $$
Where $\alpha$ decays as $\Delta T$ increases (the older the anchor, the more weight is given to the current opponent's Elo as a proxy for our own standing).

This $R_{internal}$ is fed into the MoE Router. If $R_{internal}$ indicates severe rank decay, the softmax router shifts probability mass towards High-Variance Experts.
