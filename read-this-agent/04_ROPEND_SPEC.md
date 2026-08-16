# RoPE-ND Integration Specification

## 1. Scope

RoPE-ND is the only new positional/phase architecture requested in this continuation.

Do not introduce HOPE or another architecture from transcription noise.

RoPE-ND must be **added to the current Stage 4 attention representation**, not used as a justification to delete the learned content encodings.

The core project distinction is:

\[
\boxed{
J_t=\text{recurrent content}
\qquad
\phi_t=\text{relational phase}.
}
\]

---

# 2. Standard rotary block

For angle \(\theta\):

\[
R(\theta)
=
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{bmatrix}.
\]

For scalar coordinate \(p\):

\[
\rho(p)
=
\bigoplus_{r=1}^{m}
R(\omega_r p).
\]

Then:

\[
\rho(p)^\top\rho(q)
=
\rho(q-p).
\]

Apply to attention:

\[
q'_p=\rho(p)q_p,
\]

\[
k'_q=\rho(q)k_q.
\]

Therefore:

\[
(q'_p)^\top k'_q
=
q_p^\top
\rho(q-p)
k_q.
\]

This exposes relative coordinate displacement to attention.

---

# 3. Multidimensional construction

Let:

\[
p
=
(p^{(1)},\ldots,p^{(D)}).
\]

Partition rotary pairs across axes.

For axis \(j\), pair \(r\):

\[
R_{j,r}(p)
=
R(\omega_{j,r}p^{(j)}).
\]

Then:

\[
\boxed{
\rho_D(p)
=
\bigoplus_{j=1}^{D}
\bigoplus_{r=1}^{m_j}
R(\omega_{j,r}p^{(j)}).
}
\]

with:

\[
\sum_j 2m_j
\le
d_{\text{head}}.
\]

Unused head dimensions may remain unrotated.

Relative operator:

\[
\rho_D(p)^\top\rho_D(q)
=
\bigoplus_{j,r}
R(
\omega_{j,r}[q^{(j)}-p^{(j)}]
).
\]

---

# 4. Candidate Pokémon axes

Start from coordinates already available in the live inference path.

Candidate axes:

### Axis 1 — turn

\[
p^{(1)}=\text{turn index}.
\]

### Axis 2 — logical decision

\[
p^{(2)}=\text{decision index within episode}.
\]

### Axis 3 — substep

\[
p^{(3)}=\text{autoregressive/multi-select substep}.
\]

### Axis 4 — side / perspective

Prior thesis material included side as a coordinate.

Because side is binary/categorical, test it rather than assuming it deserves a large rotary allocation.

### Axis 5 — strategic phase

Construct only from an already-defined numeric game signal.

Examples that may be derived cheaply from current state:

- prize progression;
- terminal proximity;
- phase bucket already encoded by the model;
- other existing strategic scalar.

Do not invent an expensive new feature pipeline.

---

# 5. First implementation should be small

Do not rotate the whole model indiscriminately.

Possible initial allocation for each attention head:

```text
some rotary pairs → turn
some → decision
some → substep
remaining dims → unchanged
```

Then autoresearch:

- add side;
- add strategic phase;
- change pair allocation;
- change frequencies;
- change scale.

The exact allocation must match `d_head`.

---

# 6. Learnable axis scales

A useful low-cost extension is:

\[
\theta_{j,r}
=
\alpha_j\omega_{j,r}p^{(j)}
\]

with trainable \(\alpha_j\).

This makes each axis suppressible/amplifiable by learning.

Possible initialization:

\[
\alpha_j=0
\]

gives an identity rotation at initialization.

Possible small nonzero initialization gives immediate relational bias.

Do not treat identity initialization as mandatory.

Autoresearch can compare:

```text
identity-start RoPE-ND
vs
standard-scale RoPE-ND
```

---

# 7. Weight loading

Stage 4 is an initialization.

Load all compatible Stage 4 weights into the expanded model.

Initialize only the new RoPE-specific parameters if any.

If RoPE uses no trainable parameters besides existing Q/K projections, architecture serialization still needs a config/version flag describing axes and rotary allocation.

Do not require training from scratch.

---

# 8. Attention integration

Find the current Stage 4 attention implementation.

RoPE-ND belongs after Q/K projection and before QK score computation:

```text
hidden
↓
Q,K,V projections
↓
RoPE-ND(Q,K, coordinates)
↓
masked attention
↓
output
```

The legal-action mask is a policy-level admissibility object and is independent of RoPE-ND.

Do not mix the legal mask with positional phase.

---

# 9. Scratch tokens / registers

If scratch registers participate in the same attention sequence, explicitly decide their coordinate treatment.

Candidate options:

### shared current coordinate

Scratch tokens at decision \(t\) receive current episode coordinate \(p_t\).

### dedicated register axis/type

Scratch tokens receive an additional register identity through existing embeddings, while RoPE encodes only temporal/relational axes.

Prefer not to turn scratch-register ID into a continuous temporal axis unless there is a concrete reason.

Remember:

\[
\text{scratch register identity}
\neq
\text{time coordinate}.
\]

---

# 10. Multi-select semantics

For a logical decision with substeps:

\[
p_{t,j}
=
(\text{turn}_t,\text{decision}_t,j,\ldots).
\]

This allows attention to distinguish sub-actions while maintaining common higher-level coordinates.

If current recurrent memory intentionally remains fixed across substeps, RoPE-ND can still distinguish the substep relation without pretending a new recurrent step occurred.

---

# 11. Minimal tests

Before a long run:

1. shape test;
2. finite-value test;
3. mask compatibility;
4. serialization/load test;
5. action-selection smoke;
6. short gradient test.

If identity-initialized:

verify outputs approximately match Stage 4 before training.

If not identity-initialized:

do not demand output equality; instead verify the perturbation is finite and trainable.

Then train.

---

# 12. Required ablation

At minimum compare:

\[
\text{best GRPO configuration without RoPE-ND}
\]

against

\[
\text{same GRPO configuration with RoPE-ND}.
\]

If RoPE-ND wins:

increase its compute allocation and tune axes.

If it loses:

try one meaningful axis/allocation variant before abandoning it, unless the failure is catastrophic.

---

# 13. Relation to the thesis

The larger project hypothesis is not:

> RoPE creates memory.

It is:

> persistent scratch already transports learned temporal content; RoPE-ND may expose explicit relative geometry for how that content is situated along the chained decision process.

Symbolically:

\[
J_t
=
\text{what survived},
\]

\[
p_t
=
\text{where/when/under-which-relation},
\]

\[
\rho_D(p_t)
=
\text{group action used to expose relative relation}.
\]

This is the architecture experiment to test.
