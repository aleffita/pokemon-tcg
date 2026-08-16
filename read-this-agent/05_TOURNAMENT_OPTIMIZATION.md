# Tournament, Validation, and Throughput Protocol

## 1. Objective

The competition agent is evaluated in an evolving opponent environment.

Therefore no single training loss or one direct candidate-vs-root score is sufficient.

Use a compact evaluation surface that can be repeated frequently.

---

# 2. Three classes of evidence

## A. Competitive gameplay

Highest direct relevance.

Track:

- W/L/D;
- overall win rate;
- per-opponent win rate;
- candidate-vs-current-best;
- side-conditioned win rate;
- paired seeds when available.

## B. Policy/representation diagnostics

Track:

- entropy;
- KL to behavior snapshot;
- clip fraction;
- gradient norm;
- return distribution;
- group variance;
- illegal-action rate;
- episode length;
- prospective-head outputs/losses when trained.

## C. Existing supervised validation

When already cheap to run, record:

- `val_acc`;
- attack/KO validation diagnostics;
- auxiliary losses;
- any existing Stage 4 validation metrics.

Do not rebuild historical data infrastructure merely to obtain them.

These are regression/competence diagnostics, not the primary objective.

---

# 3. Tournament ladder

## Smoke

Enough games to catch:

- broken loading;
- illegal action;
- deterministic collapse;
- total policy failure.

## Screen

Run against the current opponent panel.

Use enough games to determine whether more evaluation is worth it.

## Confirmation

Promising candidates get more games.

## Head-to-head

When two GRPO variants are both interesting, run them directly against each other if possible.

---

# 4. Opponent panel

Recover current available opponents from the repository.

At minimum retain the useful categories already used in autoresearch:

- random;
- first/baseline;
- a strong named public opponent such as the current equivalent of `lb826_alakazam_seok`;
- current Stage 4 or current best local champion;
- the other GRPO variant candidate when comparing algorithms.

If better/more current competition-relevant opponents are already available, include them.

Do not optimize only against Stage 4.

---

# 5. Adaptive game count

Do not hard-code one tournament size.

If a candidate is obviously broken, stop early.

If two candidates are close, run more games.

If a candidate appears materially stronger, increase confirmation games.

A simple uncertainty estimate for win probability \(p\) with \(n\) decisive games is:

\[
\hat p
=
\frac{W}{n}.
\]

Approximate standard error:

\[
SE
=
\sqrt{
\frac{\hat p(1-\hat p)}{n}
}.
\]

Use this only as a rough guide; paired seeds and opponent heterogeneity make the exact inference more complicated.

The practical decision is whether more games are likely to change the research choice.

---

# 6. Multi-objective candidate score

Do not collapse everything into a fixed hand-designed scalar unless it helps automation.

A candidate should generally satisfy:

1. competitive result is not clearly worse;
2. no catastrophic validation collapse;
3. no illegal-action/recurrent-state bug;
4. training remains numerically stable.

If needed, define an experimental dashboard rather than one score.

---

# 7. Leaderboard dynamics

The real leaderboard is not a stationary single opponent.

A candidate that only defeats its parent may be exploiting a private weakness.

Therefore:

\[
\text{candidate vs parent}
\]

is necessary but insufficient.

Use:

\[
\text{candidate vs opponent mixture}.
\]

Where the submission/leaderboard workflow is available and the human chooses to submit a candidate, treat leaderboard feedback as additional evidence, not as a reason to stop local autoresearch.

---

# 8. Throughput is a research metric

The relevant system metric is not minimum resource usage.

It is:

\[
\boxed{
\text{competitive information gained per hour}.
}
\]

Measure end-to-end:

\[
T_{\text{cycle}}
=
T_{\text{collect}}
+
T_{\text{update}}
+
T_{\text{eval}}.
\]

Then:

\[
\text{cycles/hour}
=
\frac{3600}{T_{\text{cycle}}}.
\]

A 2× faster collector may be more valuable than a tiny model tweak because it multiplies every subsequent experiment.

---

# 9. M3 Pro / 24 GB unified memory policy

Use the machine aggressively.

Useful tactics to benchmark:

- multiple simultaneous self-play games;
- batched policy inference;
- multiple resident policy snapshots;
- in-memory rollout groups;
- preallocated tensor buffers;
- minimal Python object creation in inner loops;
- avoid repeated serialization;
- avoid repeatedly loading weights;
- avoid framework conversion inside every iteration;
- use MLX or PyTorch/MPS according to measured end-to-end speed.

Do not optimize for low memory usage while the machine is idle.

Optimize for wall-clock.

---

# 10. Direct framework decision

The previous pipeline used MLX training and PyTorch inference in different contexts.

For the new RL loop, cross-framework boundaries may be unnecessary.

If the existing PPO update path is already working in one framework, extend it first.

Only benchmark another framework if there is a concrete speed bottleneck.

A valid quick decision experiment:

```text
same batch / same model
PyTorch-MPS update throughput
vs
MLX update throughput
```

Then use the faster end-to-end path.

Do not spend hours engineering parity between frameworks for aesthetic reasons.

---

# 11. Autoresearch selection loop

For every candidate:

```text
collect
↓
update
↓
quick diagnostics
↓
tournament screen
↓
keep / modify / reject
↓
commit result
↓
next hypothesis
```

Every few wins:

```text
larger tournament
↓
larger self-play budget
↓
confirmation
```

---

# 12. Validation accuracy interpretation

BC validation accuracy measures imitation fidelity on the historical validation distribution.

It is useful because a candidate that suddenly drops from a normal Stage 4 range to near-random may have destroyed broad competence.

But the research already observed:

\[
\boxed{
\text{Acc}_{val}
\not\Rightarrow
\text{WinRate}.
}
\]

Therefore use validation accuracy as a diagnostic axis, not as the selection target.

---

# 13. Rare-event metrics

Because prior training revealed strategically rare transitions, track per-event behavior if cheap:

- would-KO action rate / accuracy;
- prize-changing decisions;
- terminal decisions;
- attack decisions;
- retreat/switch if instrumented.

This can explain why two candidates with similar aggregate validation behave differently in tournament.

Again: do not create a new ETL project for these metrics.

Use what already exists or what self-play directly exposes.
