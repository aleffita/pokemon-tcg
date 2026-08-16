# Mathematical Thesis — Admissible Fibers, Recurrent State, Relative Credit, and RoPE-ND

This file extracts the mathematical objects that are directly useful for the final Pokémon TCG experiment.

It is not a requirement to prove the full long-horizon thesis before implementation.

The purpose is to give the research agent the derivation rather than asking it to guess what the project means by fibers, chaining, phase, and group-relative selection.

---

# 1. State, recurrence, and admissibility

Let the observable game state at logical decision \(t\) be

\[
X_t.
\]

Let the persistent recurrent workspace be

\[
J_t\in\mathbb R^{m\times d}.
\]

The Stage 4 family implements some recurrent update of the form

\[
J_{t+1}
=
F_\theta(J_t,X_t,a_t),
\]

with exact arguments determined by current code.

The policy is not merely

\[
\pi_\theta(a_t|X_t),
\]

but

\[
\boxed{
\pi_\theta(a_t|X_t,J_t).
}
\]

At state \((X_t,J_t)\), legality defines an admissibility predicate

\[
A(X_t,a)\in\{0,1\}.
\]

The legal-action fiber is

\[
\boxed{
\mathcal F_t
=
\{a:A(X_t,a)=1\}.
}
\]

The masked policy is therefore

\[
\pi_\theta(a|X_t,J_t)
=
\frac{
\mathbf 1[a\in\mathcal F_t]\exp z_\theta(a;X_t,J_t)
}{
\sum_{b\in\mathcal F_t}
\exp z_\theta(b;X_t,J_t)
}.
\]

This is the concrete finite version of the admissibility-fiber idea.

---

# 2. Chaining

An accepted action changes not only the state but the next legal fiber:

\[
(X_t,J_t,\mathcal F_t)
\xrightarrow{a_t}
(X_{t+1},J_{t+1},\mathcal F_{t+1}).
\]

Therefore local decision problems are chained.

The next choice set depends on the trajectory that produced the next base state.

Abstractly, if

\[
B_t
\]

is the committed/current base condition and

\[
F_{B_t}
\]

is a fiber of structurally valid continuations, one can write an admissible subset

\[
\mathcal A_t
=
\{u\in F_{B_t}:E_t(u)\le \lambda_t\},
\]

choose

\[
u_t\in\mathcal A_t,
\]

and evolve

\[
B_{t+1}
=
\Phi(B_t,u_t).
\]

For Pokémon, the legal-action predicate is explicit and therefore provides a much cleaner finite laboratory than the abstract construction.

---

# 3. The recurrent workspace result

The prior project evidence supports this careful statement:

\[
\boxed{
J_t
\text{ acquired future-relevant latent information.}
}
\]

The empirical clue was that auxiliary future heads learned to predict quantities such as future KO/prize/terminal/return structure from persistent scratch.

Let a future target be \(Y_{t+h}\).

An auxiliary head has form

\[
\widehat Y_{t+h}
=
f_h(J_t).
\]

The stronger information test is

\[
\operatorname{Predict}(Y_{t+h}|X_t,J_t)
>
\operatorname{Predict}(Y_{t+h}|X_t)
\]

under comparable capacity/evaluation.

The project does **not** need to prove this hierarchy during the competition run. The important operational conclusion is:

> the recurrent state already contains useful learned temporal content, so self-play RL should be allowed to exploit it rather than resetting it.

---

# 4. Gradient horizon is not memory horizon

TBPTT imposes a gradient truncation.

It does not logically imply that recurrent state must reset at the same boundary.

The key separation is

\[
\boxed{
\text{gradient horizon}
\neq
\text{working-memory horizon}.
}
\]

If the recurrent state crosses chunk boundaries,

\[
J_{t+k}
\]

can carry information originating before the current gradient graph.

For self-play GRPO this produces two legitimate update approximations.

## Detached-boundary update

Use stored recurrent boundary state:

\[
\bar J_t
=
\operatorname{stopgrad}(J_t),
\]

and optimize

\[
\pi_\theta(a_t|X_t,\bar J_t).
\]

This is cheap and was close to the prior PPO probe.

## Recurrent-recompute update

Given a chunk starting at \(t_0\),

\[
J_{t_0}
\]

is the boundary state, then recompute

\[
J_{t+1}=F_\theta(J_t,X_t,a_t)
\]

inside the chunk and allow gradient flow through that chunk.

This better aligns the learner's policy with its own current recurrent dynamics.

Autoresearch should compare them rather than treating either as dogma.

---

# 5. Why the historical win-rate drop did not falsify recurrence

The prior project history contains two distinct causal paths.

Representation path:

\[
\text{persistent scratch + recurrent training}
\rightarrow
\text{future-relevant }J_t
\rightarrow
\text{aux heads decode future structure}.
\]

Policy-credit path:

\[
\text{rare-event normalization defect}
\rightarrow
\text{distorted gradient weighting}
\rightarrow
\text{policy degradation}.
\]

Therefore it is possible that

\[
\text{aux predictive quality}\uparrow
\]

while

\[
\text{win rate}\downarrow.
\]

There is no contradiction because they measure different objects.

The corrected research goal is to give the learned representation a direct self-play objective tied to game outcome.

---

# 6. Rare events and strategic information density

Suppose routine decisions dominate the empirical action distribution.

BC optimizes

\[
L_{\mathrm{BC}}
=
-\mathbb E_{(s,a)\sim D}
\log\pi_\theta(a|s).
\]

If strategically decisive events are rare under \(D\), they contribute little to the expectation even when they have enormous effect on terminal outcome.

Thus:

\[
\boxed{
P_D(\text{event})
\neq
\text{strategic importance of event}.
}
\]

The historical bug accidentally amplified gradients near would-KO, prize and terminal transitions.

The scientific interpretation is not “bugs are good.”

It is:

> event frequency is a poor proxy for game-theoretic value.

Self-play terminal return solves this at the objective level because winning is directly optimized.

Potential shaping may later accelerate credit assignment without replacing the terminal objective.

---

# 7. Group-relative selection as a quotient-like geometry

Consider \(G\) sibling candidate trajectories from a common base condition.

Returns:

\[
r
=
(r_1,\ldots,r_G)^\top.
\]

Define the centering projector

\[
\boxed{
P_G
=
I-\frac1G\mathbf 1\mathbf 1^\top.
}
\]

Then

\[
P_Gr
=
r-\bar r\mathbf 1.
\]

Normalize:

\[
\widehat r
=
\frac{P_Gr}
{\|P_Gr\|+\epsilon}.
\]

The relative signal is insensitive to a shared additive offset:

\[
P_G(r+b\mathbf 1)=P_Gr.
\]

After normalization it is also insensitive to positive global scaling:

\[
r
\sim
ar+b\mathbf 1,
\qquad
a>0.
\]

This motivates the projective/quotient-like interpretation:

> the learning signal is the relative ordering/direction among siblings, not an absolute reward origin.

This is an analytical lens for GRPO. It is not a claim that GRPO literally equals projective geometry.

---

# 8. Why the natural GRPO group is a fiber

At base state \(b\), legal actions form

\[
\mathcal F_b.
\]

Choose sibling actions

\[
a_1,\ldots,a_G
\in
\mathcal F_b.
\]

Each produces a continuation

\[
\tau_i
=
\operatorname{Rollout}(b,a_i).
\]

These continuations are naturally comparable because they share the same base.

Hence:

\[
\boxed{
\text{fiber}
\rightarrow
\text{sibling candidate family}
\rightarrow
\text{relative return}
\rightarrow
\text{group-relative credit}.
}
\]

This is the exact mathematical motivation for sibling-fiber GRPO.

---

# 9. Behavior ratios

Let the behavior snapshot be

\[
\pi_{\text{old}}.
\]

For a sampled logical action \(a_t\),

\[
\ell_t^{\text{old}}
=
\log\pi_{\text{old}}(a_t|X_t,J_t),
\]

and the current learner gives

\[
\ell_t^\theta
=
\log\pi_\theta(a_t|X_t,J_t).
\]

Importance ratio:

\[
\boxed{
\rho_t(\theta)
=
\exp(
\ell_t^\theta-\ell_t^{\text{old}}
).
}
\]

A clipped group-relative objective is

\[
L(\theta)
=
-
\mathbb E
\left[
\min(
\rho_tA,
\operatorname{clip}(\rho_t,1-\epsilon_c,1+\epsilon_c)A
)
\right].
\]

The exact estimator, rollout granularity, entropy term, KL term and number of epochs are experimental variables.

---

# 10. Composite actions

If one game decision is represented by conditional choices

\[
a=(a_1,\ldots,a_m),
\]

then

\[
\pi(a|s,J)
=
\prod_{j=1}^m
\pi(a_j|s,J,a_{<j}),
\]

so

\[
\boxed{
\log\pi(a|s,J)
=
\sum_j
\log\pi(a_j|s,J,a_{<j}).
}
\]

This is required for a mathematically meaningful behavior ratio.

If the existing environment encodes multi-select decisions as several rows, they must be regrouped into one logical action for the probability contract, while respecting the recurrent-state semantics implemented by the policy.

---

# 11. RoPE as relational phase

The project explicitly separates recurrent content from phase.

\[
J_t
\rightarrow
\text{content / learned sufficient statistics},
\]

\[
\phi_t
\rightarrow
\text{relative position / phase / relation}.
\]

For one scalar coordinate \(p\), define a block rotation representation

\[
\rho(p)
=
R_{\omega_1p}
\oplus
\cdots
\oplus
R_{\omega_mp},
\]

where

\[
R_\alpha
=
\begin{bmatrix}
\cos\alpha & -\sin\alpha\\
\sin\alpha & \cos\alpha
\end{bmatrix}.
\]

Then

\[
\boxed{
\rho(p)^\top\rho(q)
=
\rho(q-p).
}
\]

Thus dot products after rotating \(Q,K\) depend on relative displacement.

---

# 12. RoPE-ND

Let the multidimensional coordinate be

\[
p_t
=
(p_t^{(1)},\ldots,p_t^{(d)}).
\]

Candidate Pokémon coordinates include:

\[
p_t
=
(
\text{turn},
\text{decision},
\text{substep},
\text{side},
\text{strategic phase},
\ldots
).
\]

Allocate rotary pairs to coordinate axes.

For axis \(j\) and frequency \(r\),

\[
R_{j,r}(p)
=
R_{\omega_{j,r}p^{(j)}}.
\]

Then

\[
\rho(p)
=
\bigoplus_{j,r}
R_{j,r}(p).
\]

Rotated queries and keys:

\[
q'_t=\rho(p_t)q_t,
\qquad
k'_u=\rho(p_u)k_u.
\]

Their relative operator is

\[
\rho(p_t)^\top\rho(p_u)
=
\bigoplus_{j,r}
R_{\omega_{j,r}(p_u^{(j)}-p_t^{(j)})}.
\]

Hence each selected coordinate contributes an explicit relative phase.

This is what RoPE-ND is testing.

It is not being introduced as a memory mechanism.

---

# 13. State-dependent phase as an optional later branch

The broader thesis allows phase itself to depend on recurrent/task state:

\[
\phi_t
=
P_\psi(J_t,E_t,\mathbf H_t,X_t).
\]

For the competition implementation, a simple explicit coordinate vector should come first.

If useful evidence appears, learned scale/gating can be added:

\[
\widetilde p_t^{(j)}
=
\alpha_j p_t^{(j)}
\]

with trainable \(\alpha_j\).

This allows the model to learn how strongly each axis affects rotation.

---

# 14. 2-adic / toroidal geometry — context, not P0 implementation

The broader thesis considers binary congruence through the 2-adic metric

\[
d_2(x,y)
=
2^{-v_2(x-y)}
\]

and modular coordinates through phase maps

\[
\chi_j(x_j)
=
e^{2\pi i x_j/2^{w_j}},
\]

whose product lies on

\[
\mathbb T^k=(S^1)^k.
\]

The conceptual pairing is

\[
\boxed{
\text{2-adic depth}
+
\text{phase/orientation}.
}
\]

For the present competition run this is **research context**, not a requirement to implement a p-adic architecture.

RoPE-ND is the concrete near-term phase experiment.

---

# 15. Global trajectory topology — later explanatory layer

The broader research has candidate invariants such as:

- persistent homology;
- topological entropy;
- symbolic dynamics / subshifts;
- Conley index;
- winding numbers / homology classes;
- coupled 2-adic + phase filtrations;
- observational-equivalence classes.

These are not required for the competition implementation.

The immediate finite bridge is simpler:

\[
\text{admissible fiber}
+
\text{recurrent chain}
+
\text{relative sibling credit}
+
\text{multidimensional phase}.
\]

That is already mathematically concrete and executable.

---

# 16. The exact thesis being tested now

The competition experiment is testing the following composite hypothesis:

1. Stage 4 already contains a useful recurrent policy representation.
2. Persistent \(J_t\) carries future-relevant information.
3. True recurrent self-play exposes the policy to its own endogenous state distribution.
4. GRPO replaces uniform imitation frequency with direct relative game-outcome credit.
5. Sibling-fiber grouping may reduce irrelevant variance by comparing admissible continuations from a common base.
6. RoPE-ND may expose relative coordinates that the recurrent representation currently has to infer implicitly.
7. The combination may improve competitive policy strength faster than another BC pass.

Symbolically:

\[
\boxed{
(X_t,J_t,\mathcal F_t,\phi_t)
\xrightarrow{\text{self-play}}
\{\tau_i\}_{i=1}^G
\xrightarrow{\text{relative return}}
A_i
\xrightarrow{\text{GRPO}}
\theta'
}
\]

with

\[
\theta'
\]

evaluated by actual gameplay.

That is the core experimental thesis.
