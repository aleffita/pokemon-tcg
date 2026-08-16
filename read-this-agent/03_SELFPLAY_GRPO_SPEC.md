# Engineering Specification — True Recurrent Self-Play + Two GRPO Algorithms

## 1. Goal

Implement a fast self-play RL engine around the current Stage 4 policy.

The hot path is entirely new self-play experience.

Do not route the new experience through historical Parquet/BC ETL.

---

# 2. Runtime objects

For episode \(e\), maintain:

\[
S_t^e
\]

for environment state.

For each side:

\[
J_{t,0}^e,\qquad J_{t,1}^e.
\]

For the behavior policy snapshot:

\[
\pi_{\text{old}}.
\]

For the learner:

\[
\pi_\theta.
\]

A logical transition record should contain only data needed for the update:

```text
episode_id
group_id
side
logical_decision_id
substep structure
model inputs or in-process references
legal masks
actions
complete behavior logprob
terminal flag
terminal return
recurrent boundary state if needed
RoPE-ND coordinates
opponent/snapshot identity
seed/determinization identity if available
```

This buffer can be ephemeral.

It does not need historical-dataset provenance machinery.

---

# 3. True recurrent mirror

Implement a policy wrapper for each side with an independent state lane.

Pseudo-semantics:

```python
memory = {
    (episode_id, 0): initial_memory(),
    (episode_id, 1): initial_memory(),
}

def choose(side, obs):
    j_in = memory[(episode_id, side)]
    action, logp, j_out = policy.choose(obs, j_in)
    memory[(episode_id, side)] = j_out
    return action, logp
```

The actual update point for `j_out` must match the existing Stage 4 logical-action/substep contract.

Do not reset on every opponent decision.

At episode end, destroy both lanes.

---

# 4. Batched self-play

The model is small and the machine is dedicated.

Batch independent games.

At each scheduler step, gather all games that need a policy decision and form one model batch.

Maintain per-game recurrent state arrays.

Conceptual loop:

```text
N live games
↓
collect decision-ready states
↓
batch policy forward
↓
sample masked actions
↓
scatter recurrent outputs to lanes
↓
advance simulators
↓
repeat
```

Increase `N` until either:

- model throughput saturates;
- simulator CPU becomes bottleneck;
- unified-memory pressure causes harmful paging;
- latency/serialization overhead grows.

Optimize for games/hour and decisions/hour, not memory minimalism.

---

# 5. Behavior snapshot lifecycle

For one GRPO iteration:

1. snapshot current policy as \(\pi_{\text{old}}\);
2. generate groups with \(\pi_{\text{old}}\);
3. update learner \(\pi_\theta\);
4. evaluate;
5. next iteration snapshots the chosen current policy.

Do not mix behavior logprobs from multiple behavior policies in one update without explicitly identifying them.

---

# 6. Full logical-action logprob

Given sub-actions

\[
a_{1:m},
\]

store:

\[
\ell^{\mathrm{old}}
=
\sum_{j=1}^m
\ell_j^{\mathrm{old}}.
\]

During learner recomputation:

\[
\ell^\theta
=
\sum_{j=1}^m
\ell_j^\theta.
\]

Then

\[
\rho
=
e^{\ell^\theta-\ell^{\mathrm{old}}}.
\]

Unit test:

- force a known composite action;
- compute each conditional logprob manually from masked logits;
- verify stored complete logprob equals sum;
- verify ratio equals 1 when learner weights equal behavior weights.

This is a P0 test and should take minutes, not hours.

---

# 7. Variant A — trajectory-group GRPO

## 7.1 Group construction

A group should minimize irrelevant variation.

Useful group key:

```text
initial condition / seed
deck pair
side assignment
behavior snapshot
opponent snapshot
```

For group size \(K\), run \(K\) trajectories.

If full same-seed replication would produce identical deterministic games, vary controlled stochastic choices while preserving a meaningful common base.

## 7.2 Return

Start with terminal return:

\[
R_i\in\{-1,0,+1\}.
\]

Group statistics:

\[
\mu_G=\frac1K\sum_i R_i,
\]

\[
\sigma_G=
\sqrt{\frac1K\sum_i(R_i-\mu_G)^2}.
\]

Advantage:

\[
A_i
=
\frac{R_i-\mu_G}{\sigma_G+\epsilon}.
\]

If \(\sigma_G=0\), the group has no relative terminal signal.

Track:

\[
f_{\text{zero-var}}
=
\frac{\#\{\text{zero variance groups}\}}
{\#\{\text{groups}\}}.
\]

If this is high, change grouping or introduce a carefully tested intermediate return.

## 7.3 Update

Each trajectory action receives \(A_i\) initially.

For timestep \(t\):

\[
\rho_{i,t}
=
\exp(\ell_{i,t}^\theta-\ell_{i,t}^{old}).
\]

Objective:

\[
L_{\text{traj}}
=
-\frac1N
\sum_{i,t}
\min[
\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon_c,1+\epsilon_c)A_i
].
\]

Optional terms:

\[
-\alpha_H H(\pi_\theta)
\]

and

\[
+\beta D_{\mathrm{KL}}(\pi_\theta||\pi_{\text{ref}})
\]

are autoresearch variables.

Do not assume they are mandatory.

---

# 8. Variant B — sibling-fiber GRPO

## 8.1 Branch state

At a selected decision state:

\[
b=(S_t,J_t,\xi),
\]

with legal fiber:

\[
\mathcal F_b.
\]

Choose \(K\) admissible actions.

Selection options to test:

- stochastic samples without replacement;
- top-\(K\) plus stochastic diversity;
- actions above a probability floor;
- all actions when fiber is small.

## 8.2 Common random numbers

Where the simulator exposes controllable randomness, branch from the same determinization:

\[
\xi.
\]

Then

\[
\tau_i
=
\operatorname{Rollout}(b,a_i,\xi).
\]

This reduces variance from one branch simply getting luckier random events.

If exact state cloning is difficult, implement the smallest reliable snapshot/restore mechanism needed for branching.

## 8.3 Return and relative advantage

\[
R_i=R(\tau_i),
\]

\[
A_i
=
\frac{R_i-\bar R}{\sigma_R+\epsilon}.
\]

First implementation: apply \(A_i\) to the branching action only.

\[
L_{\text{fiber-root}}
=
-
\frac1K
\sum_i
\min[
\rho_iA_i,
\operatorname{clip}(\rho_i,1-\epsilon_c,1+\epsilon_c)A_i
].
\]

This has a clean causal interpretation.

Later branch:

apply the same branch advantage to downstream actions from that rollout.

Compare.

---

# 9. Selecting branch points

Do not branch every decision initially.

High-value candidates:

- action entropy above threshold;
- more than one materially different legal action;
- would-KO opportunity;
- prize-changing decision;
- retreat/switch decision;
- terminal approach;
- high prospective-head disagreement/uncertainty;
- policy/value disagreement.

But first prove sibling branching works on any valid decision.

Then optimize selection.

---

# 10. Recurrent learner update

## Mode D — detached recurrent boundary

Use stored \(J_t\) inputs.

Fast and directly comparable to prior PPO.

## Mode R — recurrent chunk replay

Store chunk boundary state \(J_{t_0}\), observations, actions, masks.

Recompute:

\[
J_{t+1}=F_\theta(J_t,\cdot)
\]

for a chunk.

Compute learner logprobs from the recomputed chain.

This allows the policy update to modify how information evolves inside \(J\).

Autoresearch comparison:

```text
GRPO-D
vs
GRPO-R
```

Do not redesign the entire trainer before the first GRPO-D candidate exists.

---

# 11. Prospective shaping branch

After terminal GRPO is functional:

Define a potential from existing prospective outputs:

\[
\Phi_t
=
w^\top h_{\text{prospective}}(S_t,J_t)
\]

or use one calibrated head.

Potential-shaped reward:

\[
r'_t
=
r_t+\eta(\gamma\Phi_{t+1}-\Phi_t).
\]

Because the shaping is telescoping under suitable conditions, it can provide denser credit while keeping terminal outcome central.

Compare:

```text
terminal only
vs
terminal + prospective potential
```

Tournament decides.

---

# 12. League evolution

Start:

```text
current vs current
current vs lagged
```

Then, if policy co-adaptation appears:

```text
current vs current
current vs lagged champion
current vs fixed useful public agent
```

Do not build an elaborate league manager before basic self-play GRPO works.

---

# 13. Essential metrics per update

Collect:

```text
behavior snapshot hash/commit
learner commit
self-play games
self-play decisions
games/sec
decisions/sec
group count
group size
zero-variance group fraction
return mean/std
policy entropy
mean/max importance ratio
clip fraction
KL to behavior
gradient norm
loss
candidate vs current champion tournament
candidate vs opponent set tournament
cheap BC validation accuracy
```

These make the next autoresearch choice possible.

---

# 14. Initial hyperparameters are disposable

Use modest starting values, then search empirically.

The program does not prescribe one magic:

- group size;
- clip epsilon;
- LR;
- entropy coefficient;
- number of update epochs;
- samples/update.

The agent should derive a safe micro-run from the scale of the existing PPO update, then quickly expand the promising regime.

---

# 15. Two-algorithm tournament

Once both algorithms produce candidates:

```text
Trajectory-GRPO candidate
vs
Sibling-Fiber-GRPO candidate
```

Run direct paired games if the tournament harness permits agent-vs-agent.

Also compare both against the same fixed opponent mixture.

The better next parent is not necessarily the one with higher aggregate score if one has a strong systematic weakness; inspect per-opponent results.

Then continue autoresearch.
