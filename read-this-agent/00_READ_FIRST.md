# READ FIRST — Pokémon TCG Final Research Continuation

This directory supersedes stale strategic instructions in the previous `PROGRAM.md` where they conflict with this compendium.

The project is **not** restarting. The prior autoresearch run produced substantial working infrastructure and experimental evidence. The task now is to continue from that work and execute the research direction that was not yet actually tested.

## Canonical transcription normalization

The user frequently dictates through speech transcription. Treat nearby phonetic forms using project context before inventing new concepts.

For this run:

- `GRPU`, `GRPU`, or similar transcription noise → **GRPO**
- `ROPEND`, `RopeND`, `rope nd` → **RoPE-ND**
- `TVPTT`, `TBPBTT`, similar → **TBPTT**
- a transcription such as `HOPE` in this context does **not** introduce HOPE / Nested Learning or any other architecture

There is **no HOPE architecture task** in this research program.

Do not add a technology because a transcript happens to resemble its name.

## The research direction

The current Stage 4 code is the trained starting point.

The required continuation is:

\[
\boxed{
\text{Stage 4}
\rightarrow
\text{RoPE-ND addition}
\rightarrow
\text{true recurrent self-play}
\rightarrow
\text{GRPO}
\rightarrow
\text{tournament / validation}
\rightarrow
\text{autoresearch iteration}
}
\]

This is the research axis. Autoresearch chooses the best implementation and hyperparameters **inside and around this axis**.

The critical comparison requested by the researcher is:

1. **trajectory-group GRPO**
2. **sibling-fiber GRPO**

Implement both, train both, evaluate both, and let experimental evidence determine how to continue. A hybrid may be tested after the two primitives exist.

## What is explicitly not the task

Do not spend the remaining research window improving the historical BC dataset, Parquet ETL, provenance machinery, packed BC backend, or old curriculum unless a very small change is directly required to execute self-play GRPO.

Behavior cloning already did its job: it produced Stage 4.

Historical data may be used as an already-available validation probe, but it is not the new training source.

The new training source is:

\[
\boxed{\text{SELF-PLAY}}
\]

The new optimization regime is:

\[
\boxed{\text{GRPO / group-relative policy optimization}}
\]

The historical ETL is not on the critical path.

## Current empirical starting point

The previous autoresearch run already established important plumbing:

- a Stage 4 policy with 32 scratch registers;
- legal autoregressive action sampling;
- behavior log probabilities;
- recurrent memory input/output tracking;
- terminal returns;
- an in-process policy update path;
- candidate loading into the competition agent;
- repeated tournament execution;
- candidate-vs-root evaluation;
- several useful Git commits;
- a trajectory collector running around 64–74 rows/s in bounded probes.

It also established that the previous experiments were **not the requested final experiment**:

- no true GRPO was run;
- `mirror_no_memory` reset the opponent recurrent state and was not true recurrent self-play;
- PPO updates used detached recurrent inputs;
- every recorded `RoPEND_variant` remained `none`.

Therefore:

\[
\boxed{\text{the infrastructure was built; the central experiment remains to be done}}
\]

## Evaluation philosophy

Do not optimize one scalar blindly.

Track at least:

- tournament win rate overall and by opponent;
- paired candidate-vs-current-champion results;
- side bias;
- validation accuracy on the existing BC validation set when cheap;
- policy entropy;
- KL / policy displacement;
- legal-action correctness;
- return distribution;
- group variance;
- gradient norm;
- training throughput;
- self-play games/s or decisions/s;
- wall-clock per collect → update → tournament cycle.

The competition is dynamic. Local tournament strength is the primary development fitness signal, but validation and opponent-specific diagnostics matter because a candidate can overfit one local matchup.

## The implementation rule

Build first, measure immediately, then explain.

A theoretical explanation can be written later. For this competition window, mathematics exists to define the operators correctly and generate falsifiable branches, not to delay execution.

Read next:

1. `PROGRAM.md`
2. `01_MATHEMATICAL_THESIS.md`
3. `02_AUTORESEARCH_STATE_REVIEW.md`
4. `03_SELFPLAY_GRPO_SPEC.md`
5. `04_ROPEND_SPEC.md`
6. `05_TOURNAMENT_OPTIMIZATION.md`
7. `06_GITFLOW_AND_ORCHESTRATION.md`
8. `07_GOAL_BOOTSTRAP_PROMPT.md`
