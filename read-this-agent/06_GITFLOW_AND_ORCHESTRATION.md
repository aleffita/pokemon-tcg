# GitFlow and Sequential Agent Orchestration

## 1. Git is the experimental notebook

The prior run did create many commits. Continue that practice deliberately.

Do not work for multiple hours with a giant uncommitted working tree.

The project is already under Git and has releases/history.

Git exists so experiments can be aggressive.

---

# 2. Inspect the repository's real GitFlow

Before creating a branch, inspect:

```bash
git status
git branch -vv
git log --oneline --decorate -30
git tag --sort=-creatordate | head
```

Determine the actual convention already used.

The export showed work on `develop`, but current repository truth wins.

If standard GitFlow is being used and no suitable branch exists, create/continue a feature branch from the appropriate integration branch.

Example only if consistent with repo history:

```text
feature/autoresearch-grpo-ropend
```

Do not invent a competing branching convention if the repository already has one.

---

# 3. Commit cadence

Commit after every coherent implementation milestone or scientifically meaningful experiment.

Examples:

```text
feat(rl): add recurrent two-sided self-play
test(rl): verify composite action log probabilities
feat(rl): implement trajectory-group grpo
feat(rl): implement sibling-fiber grpo
feat(model): add multidimensional rope
perf(rl): batch self-play policy inference
exp(ar-018): trajectory grpo tournament
exp(ar-019): sibling-fiber grpo tournament
fix(rl): preserve recurrent lanes across opponent decisions
```

A commit should answer:

> what changed, and why can we now run a new experiment?

---

# 4. Experiment IDs

Continue the current AR numbering rather than overwriting old experiments.

If AR-017 is the last durable experiment, continue with AR-018 or the next actually unused ID.

Each experiment directory should contain only what helps reproduce or inspect that experiment.

Suggested minimum:

```text
AR-XXX/
  report.md
  config.json
  metrics.json
  tournament.json
  logs/
```

Large ephemeral rollout buffers need not be committed unless repository conventions require it.

---

# 5. Ledger

Append one durable ledger entry per meaningful experiment.

Suggested fields:

```json
{
  "experiment_id": "AR-XXX",
  "parent": "AR-YYY",
  "git_commit": "...",
  "algorithm": "trajectory_grpo | sibling_fiber_grpo | hybrid",
  "ropend": "none | axes/config",
  "selfplay_mode": "...",
  "group_size": 0,
  "games_collected": 0,
  "decisions_collected": 0,
  "collect_seconds": 0,
  "update_seconds": 0,
  "training_steps": 0,
  "validation_accuracy": null,
  "entropy": null,
  "kl": null,
  "tournament": {},
  "decision": "keep | reject | iterate",
  "next_hypothesis": "..."
}
```

Do not make ledger schema work a research project. Extend the existing schema minimally.

---

# 6. Failed experiments

A failed experiment is useful if it identifies a branch.

If the implementation itself is clean and the negative result is scientifically meaningful:

commit it and record the failure.

If it is merely a broken intermediate edit:

fix or revert it and commit the corrected state.

Do not preserve every typo as an experiment.

---

# 7. Sequential subagents

The requested orchestration is intentionally serial.

For one objective:

### Coordinator

Writes a narrow task:

```text
Implement true recurrent mirror self-play using the existing trajectory collector.
Acceptance: both sides retain independent memory across decisions, complete logical-action logprobs remain valid, one 4-game smoke completes.
```

### Worker

Does the implementation and runs the acceptance probe.

### HALT

Wait for native completion.

No polling loop.

### Reviewer

Inspect:

- diff;
- test;
- runtime behavior;
- result.

Challenge the exact acceptance criteria.

### HALT

Wait.

### Coordinator

Resolve findings.

Commit.

Choose next objective.

---

# 8. Why sequential delegation

The point is context quality.

The coordinator retains:

- global hypothesis state;
- experiment history;
- selection logic.

Workers consume context on local code.

Reviewers independently attack one local result.

This prevents the coordinator from drowning in implementation details.

---

# 9. No scheduler theatre

Do not create:

- recurring status jobs;
- sleep-based monitors;
- progress cron;
- repeated “check if done” calls.

If a subprocess can run synchronously, run it.

If the harness supports background completion events, rely on completion events.

The only reason to interrupt training is:

- explicit experiment budget reached;
- process failure/hang;
- new evidence requiring termination.

---

# 10. Review must not become a veto bureaucracy

The previous run sometimes converted reviewer findings into long prerequisite chains before the central experiment could run.

Do not repeat that pattern.

Classify findings:

## P0 — invalidates the experiment

Examples:

- wrong behavior logprob;
- recurrent state assigned to wrong side;
- illegal action mask wrong;
- tournament loads wrong model;
- reward sign reversed.

Fix immediately before interpreting results.

## P1 — affects strength/throughput but experiment remains interpretable

Examples:

- self-play batching not optimal;
- one missing diagnostic;
- suboptimal serialization.

Record and continue unless fixing it has higher expected value.

## P2 — cleanup / provenance / style

Do not block the research loop.

Fix later if useful.

---

# 11. Merge/integration

Follow the repo's actual GitFlow.

The research agent may produce many feature commits.

Integration into `develop` or release branches should follow the existing project convention and the user's later submission decision.

Do not stop research merely because merge/release has not yet happened.
