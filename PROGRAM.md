# PROGRAM.md — Stage 4 → RoPE-ND → Self-Play GRPO Autoresearch

## Mission

Continue the existing Pokémon TCG autoresearch run from the current repository state.

The current Stage 4 code and learned weights are the starting condition, not an immutable architecture and not a preservation target.

The research program is:

\[
\boxed{
\text{Stage 4}
\rightarrow
\text{RoPE-ND}
\rightarrow
\text{true recurrent self-play}
\rightarrow
\text{GRPO}
\rightarrow
\text{tournament + validation}
\rightarrow
\text{iterate}
}
\]

The principal engineering goal is to maximize the number and quality of meaningful self-play → update → tournament cycles that can be executed on the dedicated M3 Pro with 24 GB unified memory.

Resource conservation is not the objective. The machine is dedicated to this run. Use the available memory and compute when doing so increases research throughput.

---

# 1. Resume, do not restart

Recover the current repository truth:

- current branch and HEAD;
- the 18-commit autoresearch progression or whatever the repository now contains;
- current working tree;
- `experiments/autoresearch/experiment_ledger.jsonl`;
- latest State Capsules;
- AR-015 and AR-017 artifacts;
- current Stage 4 loading path;
- current trajectory collector;
- current PPO/update implementation;
- current tournament harness;
- current candidate-loading path.

Do not spend a new research phase auditing old ETL.

The previous run already demonstrated enough to begin the central experiment.

---

# 2. Remove stale restrictions

Where an older `PROGRAM.md`, State Capsule, reviewer report, or final report says any of the following, treat it as historical context rather than a current restriction:

- Stage 4 architecture must remain unchanged;
- RoPE-ND is unjustified and should not be opened;
- stop PPO/RL exploration because AR-017 regressed;
- do not modify model/loss/root architecture;
- no RL until packed BC backend provenance is perfect;
- package/finalize merely because a previous file says `FINAL`;
- do not continue because an old deadline assumption said the exploration window closed.

The current research direction explicitly requires architecture and RL changes.

Git provides history. Use it.

---

# 3. Ignore the historical BC data path as a research objective

No more ETL research.

No more Parquet optimization research.

No more packed historical BC backend hardening unless it directly blocks one concrete command needed by the new RL loop.

Stage 4 already contains the BC knowledge.

The hot training stream should come directly from self-play.

Preferred conceptual loop:

```text
policy snapshot
    ↓
batched recurrent self-play
    ↓
in-memory / compact rollout groups
    ↓
GRPO update
    ↓
candidate
    ↓
tournament + cheap validation
    ↓
next policy snapshot
```

Do not round-trip fresh self-play through the historical Parquet pipeline.

---

# 4. True recurrent self-play is P0

The previous `mirror_no_memory` probe is insufficient.

Implement actual recurrent self-play in which both players have independent recurrent states:

\[
J_t^{(0)}, \qquad J_t^{(1)}.
\]

For each episode and side, the correct recurrent lane must persist according to the current agent's real inference semantics.

Do not reset the opponent's memory every decision.

Do not assume when memory updates occur; inspect the current Stage 4 inference contract and preserve the correct logical-decision/substep semantics.

The collector must support:

- current vs current;
- current vs lagged/current-champion snapshot;
- optionally current vs fixed useful opponents;
- side alternation;
- controlled seeds / determinization when available;
- full terminal outcome;
- complete behavior log probability;
- legal masks;
- recurrent state boundaries.

Implement the simplest correct version first, then batch it.

---

# 5. Composite action probability is P0

If a logical action is autoregressive:

\[
a_t =
(a_{t,1},\ldots,a_{t,m}),
\]

then the policy probability of the complete action is

\[
\pi_\theta(a_t\mid s_t,J_t)
=
\prod_{j=1}^m
\pi_\theta(
a_{t,j}
\mid
s_t,J_t,a_{t,<j}
),
\]

and therefore

\[
\log \pi_\theta(a_t\mid s_t,J_t)
=
\sum_{j=1}^m
\log
\pi_\theta(
a_{t,j}
\mid
s_t,J_t,a_{t,<j}
).
\]

Use the exact legal mask at every conditional step.

Do not compute GRPO ratios from only the final sub-action.

Add a focused test for this contract, then move on.

---

# 6. Build both requested GRPO variants

The autoresearch search space starts with two concrete algorithms.

## Variant A — trajectory-group GRPO

Create groups of comparable complete trajectories under controlled conditions.

For group \(G\) of size \(K\),

\[
R=(R_1,\ldots,R_K).
\]

Center:

\[
P_K
=
I-\frac1K\mathbf 1\mathbf 1^\top,
\]

\[
\widetilde R
=
P_KR
=
R-\bar R\mathbf 1.
\]

Normalize when variance is nonzero:

\[
A_i
=
\frac{R_i-\bar R}
{\sigma_R+\epsilon}.
\]

Assign the group-relative signal to the actions of trajectory \(i\), initially using the simplest defensible return assignment.

For behavior snapshot \(\pi_{\mathrm{old}}\),

\[
\rho_{i,t}(\theta)
=
\exp[
\log\pi_\theta(a_{i,t}|s_{i,t},J_{i,t})
-
\log\pi_{\mathrm{old}}(a_{i,t}|s_{i,t},J_{i,t})
].
\]

Clipped objective:

\[
L_A(\theta)
=
-
\mathbb E_{i,t}
\left[
\min
\left(
\rho_{i,t}A_i,\,
\operatorname{clip}
(\rho_{i,t},1-\epsilon_c,1+\epsilon_c)A_i
\right)
\right].
\]

Add entropy, KL, or other regularizers only as tunable experimental variables, not mandatory constraints.

## Variant B — sibling-fiber GRPO

At a strategically meaningful base state

\[
b=(s_t,J_t,\xi),
\]

let the legal-action fiber be

\[
\mathcal F_b
=
\{a:A(s_t,a)=1\}.
\]

Select \(K\) sibling actions

\[
a_1,\ldots,a_K\in\mathcal F_b.
\]

Branch the simulator from the same base condition and, where possible, the same determinization/random stream \(\xi\):

\[
\tau_i
=
\operatorname{Rollout}(b,a_i,\pi).
\]

Compute returns

\[
R_i=R(\tau_i).
\]

Center and normalize inside the sibling family:

\[
A_i
=
\frac{R_i-\bar R}{\sigma_R+\epsilon}.
\]

The simplest first update applies relative credit to the branching action:

\[
L_B^{\text{root}}
=
-
\mathbb E_i
\left[
\min(
\rho_i A_i,
\operatorname{clip}(\rho_i,1-\epsilon_c,1+\epsilon_c)A_i
)
\right].
\]

Then test a trajectory version that propagates the branch advantage through downstream actions.

This variant directly instantiates the project's fiber thesis: relative credit is defined among admissible siblings sharing the same base condition.

---

# 7. Compare the two GRPO forms experimentally

Do not ask the human which one is correct.

That is what autoresearch is for.

Minimum experimental progression:

```text
A0: trajectory-group GRPO smoke
B0: sibling-fiber GRPO smoke

A1: short train → tournament
B1: short train → tournament

winner / complementary signal
    ↓
hyperparameter refinement
    ↓
larger self-play budget
    ↓
tournament
```

If both work differently, test a hybrid:

- trajectory-group updates for broad policy improvement;
- sibling-fiber updates concentrated on high-entropy/high-impact decisions.

---

# 8. RoPE-ND must be added, not used as a replacement story

Implement a multidimensional rotary relational geometry in the current Stage 4 attention path.

Existing learned embeddings and structural encodings remain available because they carry content.

RoPE-ND adds relation.

The intended distinction is:

\[
J_t
\rightarrow
\text{recurrent learned content},
\]

\[
\phi_t
\rightarrow
\text{explicit relative phase / relation}.
\]

Implement the smallest compatible RoPE-ND and test it.

Do not spend the entire run debating whether RoPE-ND is theoretically necessary.

The program says to add it and measure it.

Required comparison:

\[
\text{GRPO without RoPE-ND}
\quad\text{vs}\quad
\text{GRPO with RoPE-ND}.
\]

See `04_ROPEND_SPEC.md`.

---

# 9. Recurrent gradients are an experimental variable, not a prerequisite sermon

The prior PPO candidate used detached recurrent inputs.

That was a useful bounded probe, not the final architecture.

Implement a path that can recompute recurrent state over contiguous chunks under the learner when practical:

\[
J_{t+1}
=
F_\theta(J_t,X_t,a_t).
\]

Compare at least:

### detached-boundary update

Use stored \(J_t\) as fixed inputs inside an update.

### recurrent-recompute update

Replay a short contiguous chunk and allow gradient flow through the chunk.

TBPTT may still truncate the graph. That is fine.

The distinction is:

\[
\text{gradient horizon}
\neq
\text{memory horizon}.
\]

Do not redesign TBPTT for its own sake. Only change chunking/reset semantics when it improves the self-play GRPO implementation or measured result.

---

# 10. Reward

Primary reward is actual game outcome:

\[
R=
\begin{cases}
+1 & \text{win}\\
0 & \text{draw if applicable}\\
-1 & \text{loss}.
\end{cases}
\]

The prospective heads (`would_ko`, prize delta, terminal, return) are useful learned signals and diagnostics.

Do not freeze them by default.

Do not delete them by default.

Do not let an accidental loss-scale pathology dominate the new objective.

Possible autoresearch branches after terminal-only GRPO works:

- prospective heads trained jointly with correctly normalized losses;
- prospective outputs used for value/potential shaping;
- rare-event-balanced auxiliary updates;
- gradient balancing;
- terminal-only control.

Win rate decides.

---

# 11. Historical rare-event discovery

The earlier training path demonstrated a useful asymmetry:

\[
\text{event frequency}
\not\propto
\text{strategic importance}.
\]

Rare would-KO, prize, and terminal transitions can have much larger game-theoretic consequence than routine actions.

The old bug must not be recreated accidentally, but the principle may be exploited intentionally.

A potential-based shaping branch is:

\[
r'_t
=
r_t
+
\eta[
\gamma\Phi(s_{t+1})-\Phi(s_t)
].
\]

A frozen or jointly learned prospective signal may be tested as \(\Phi\).

Do this only after a terminal-only self-play GRPO loop exists.

---

# 12. Optimize for experiments/hour

The machine is an M3 Pro with 24 GB unified memory and is dedicated to this work.

Primary performance quantity:

\[
\text{useful collect→update→evaluate cycles per hour}.
\]

Measure:

- self-play decisions/s;
- games/s;
- model-forward batch size;
- update steps/s;
- unified-memory pressure;
- tournament games/s;
- complete cycle wall time.

Use the memory.

Keep multiple policy snapshots resident if that is faster.

Keep rollout tensors in memory.

Batch independent game decisions.

Preallocate buffers.

Avoid serialization in the hot path.

Use the fastest viable training/inference framework already present in the repo. If MLX↔PyTorch conversion is the bottleneck, collapse the RL loop into one framework rather than preserving a historical boundary.

A small direct benchmark is enough to choose.

Do not start another historical data-engineering project.

---

# 13. Tournament and validation are part of every research cycle

A candidate is not selected by loss alone.

After each meaningful policy change:

1. run a cheap sanity evaluation;
2. run a tournament screen;
3. if promising, increase games;
4. compare against the current best candidate;
5. keep exploring the winning direction.

Track:

- total W/L/D;
- per-opponent W/L/D;
- candidate vs current champion;
- first/second-side splits;
- paired seed results where possible;
- validation accuracy on the already-existing BC validation data when cheap;
- action/attack/KO validation diagnostics already available;
- entropy/KL/gradient stats.

The competition leaderboard is dynamic, so do not optimize only candidate-vs-root.

Maintain an opponent set that includes useful public/fixed agents currently available in the repository.

---

# 14. Validation accuracy is diagnostic, not sovereign

Behavior-cloning validation accuracy remains useful because it can detect sudden destruction of previously learned competence.

Record it when the current code can compute it without rebuilding data infrastructure.

But:

\[
\text{BC val accuracy}
\not\Rightarrow
\text{win rate}.
\]

Selection is multi-signal, with competitive tournament performance carrying the highest direct relevance.

---

# 15. GitFlow is part of the research loop

Do not perform hours of uncommitted work.

Inspect the repository's actual GitFlow convention and follow it.

If the current run is directly on `develop` and the repository uses standard feature branches, create or continue an appropriate feature branch for this research, unless an existing autoresearch branch is already the intended branch.

Commit every coherent research milestone.

Examples:

```text
feat(rl): add persistent recurrent self-play
feat(rl): add trajectory-group grpo
feat(rl): add sibling-fiber grpo
feat(model): add multidimensional rope
perf(rl): batch self-play inference
exp(ar-018): compare grpo group constructions
fix(rl): preserve composite action logprob
```

The exact convention must match repository history.

Each experiment ledger entry should name its commit.

A failed experiment can also be committed if the commit is a useful reproducible negative result. Otherwise revert cleanly using Git and continue.

Do not confuse GitFlow with caution. GitFlow exists so experimentation can be aggressive and recoverable.

---

# 16. Sequential subagent orchestration

Use subagents sequentially, as requested:

```text
coordinator defines one concrete objective
    ↓
one worker
    ↓
HALT / native completion
    ↓
coordinator integrates result
    ↓
one reviewer
    ↓
HALT / native completion
    ↓
fix or accept
    ↓
commit
    ↓
next objective
```

Do not use polling, repeated sleeps, artificial schedulers, or interrupt-driven progress checking if the harness already reports completion.

The coordinator should spend its context on hypothesis selection and evidence integration.

Do not let review become a permission system that blocks the central experiment.

---

# 17. First sequence after reading this program

The next run should begin approximately as follows:

### AR-NEXT-1 — true recurrent self-play

Replace/extend `mirror_no_memory` with a true side-specific recurrent mirror.

Test:

- both sides retain their own recurrent state;
- legal actions remain valid;
- composite logprobs are finite;
- terminal outcome is assigned to the correct side.

Commit.

Review.

### AR-NEXT-2 — trajectory-group GRPO

Implement Variant A.

Use a small group size and short collection budget.

Perform an actual update.

Commit.

Run tournament.

### AR-NEXT-3 — sibling-fiber GRPO

Implement Variant B.

Start with branching-action credit only.

Perform an actual update.

Commit.

Run tournament.

### AR-NEXT-4 — compare A vs B

Same evaluation surface.

Increase games if ambiguous.

Choose next branch from evidence.

### AR-NEXT-5 — RoPE-ND integration

Add multidimensional RoPE to Stage 4.

Load existing weights plus new RoPE parameters.

Train with the better GRPO path or run a short adaptation first if required.

Commit.

Tournament.

### AR-NEXT-6 onward — autoresearch

Iterate on:

- group size;
- group construction;
- common-random-number strategy;
- rollout horizon;
- learner LR;
- clip epsilon;
- number of GRPO epochs;
- samples/update;
- entropy coefficient;
- KL coefficient if useful;
- opponent mixture;
- recurrent recompute vs detached boundary;
- RoPE axes;
- rotary dimension allocation;
- rotary scale/frequency;
- prospective shaping;
- self-play batching.

One coherent causal experiment at a time where practical.

---

# 18. Stop condition

Do not stop because an old report says exploration is closed.

Do not stop because one candidate regresses.

Do not stop because GRPO Variant A loses.

Do not stop because RoPE-ND loses one ablation.

Continue while the researcher leaves the run active and there remains a plausible experiment with positive expected value.

Final packaging/submission timing is the human researcher's decision unless explicitly delegated.

The autonomous agent's role is to keep producing increasingly informative and competitive candidates.

---

# 19. Compact research invariant

At every decision point ask:

\[
\boxed{
\text{What is the next executable change most likely to improve}
\;
\text{self-play GRPO strength per unit wall-clock?}
}
\]

Then implement it.
