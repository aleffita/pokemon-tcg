# Research Memory & Project State — Pokémon TCG AI Battle

## Project Context & Objectives
- **Target**: Kaggle Pokémon TCG AI Battle Challenge.
- **Current Objective**: Transition from Behavioral Cloning (BC) Curriculum V1 to GRPO (Group Relative Policy Optimization) and Reinforcement Learning Policy Alignment on Apple Silicon (M3 Pro 24GB).
- **Core Strategy**: BC Pre-training → Parity & Semantic corrections → Recurrent registers (TBPTT) → Elo-oriented Evaluation → GRPO / RL Alignment.

## Current State (as of 2026-08-12)
- **Precision Crisis Resolved**: PyTorch inference was collapsing due to FP16 underflow (dropping to 3.3% WR). The pipeline (`build_submission.py`, `rl/policy_infer_torch.py`, and `bc_train_mlx.py`) has been migrated to strict FP32. Existing checkpoints were surgically repacked to update their `static_feature_contract` to FP32 hashes.
- **Ablation Matrix Preparation**: We are orchestrating a cross-stage baseline tournament. Stages 1, 2, 3 (loss-corrupted aux heads) and Stage 4 (corrected, 5 epochs top-100) will battle `first_sub` on a 3x5 deck matrix. This baseline will inform the next training strategy: rapid new initialization vs GRPO/PPO distillation (Teacher: `first_sub`).
- **MLX Trainer Core**: Native FP16 trainer (`scripts/bc/bc_train_mlx.py`) with Muon + AdamW split optimizer, gradient accumulation, parquet KV cache, Tensorboard logging, and checkpoint resume support.
- **Benchmark Orchestrator**: Multi-model evaluation script (`scripts/tournament.py` & `experiments/run_full_sweeps.py`) featuring OS subprocess isolation, dynamic ETA tracking, asymmetric opponent deck sweeps (`--opp-top-decks`), best deck CSV auto-export (`--emit-best-performing-deck`), disaggregated OVERALL metrics, atomic SQLite Elo updates, and $N \times M$ unnested matrix table layout.
- **Seasons & Reset System**: Strongly-typed `seasons` table (`id`, `name`, `is_active`), with CLI controls `--new-season`, `--reset-local-elo`, and `--clear-local-matches`.
- **Sample-Size Invariant Elo ($R_{\text{invariante}}$)**: Integrated Bradley-Terry MLE inversion, MD10 placement smoothing ($N_0 = 10$), and Softmax Abelian Group translation calibration ($\Delta R_{\text{Abeliano}}$) in `rl/results_db.py`.

## Key Architectural Specs
- `d_model`: 128
- `heads`: 4
- `layers`: 3
- `ffn_dim`: 512
- `scratch_registers`: 16
- `max_options`: 192 (+ SUBMIT)

## Mathematical Framework: Sample-Size Invariant Elo ($R_{\text{invariante}}$)

### 1. Bradley-Terry Asymptotic Logistic Inversion
Given win rate $w = \frac{W}{N}$ (clipped to $[0.02, 0.98]$):
$$\hat{R}_{\infty} = 600.0 + 400.0 \cdot \log_{10}\left( \frac{w}{1 - w} \right)$$

### 2. MD10 Placement Regularization ($N_0 = 10$)
Shrinks small-sample estimates toward prior $R_0 = 600.0$:
$$R_{\text{smoothed}} = \frac{N}{N + 10} \cdot \hat{R}_{\infty} + \frac{10}{N + 10} \cdot R_0$$

### 3. Softmax Abelian Group Translation Calibration ($\Delta R_{\text{Abeliano}}$)
Computes global translation isomorphism across all overlapping entries $\mathcal{C}$ with local and remote data:
$$\alpha_k = \frac{\exp(N_k / 20.0)}{\sum_{j \in \mathcal{C}} \exp(N_j / 20.0)}$$
$$\Delta R_{\text{Abeliano}} = \sum_{k \in \mathcal{C}} \alpha_k \cdot \left( R_k^{\text{remote}} - \hat{R}_{k,\infty}^{\text{local}} \right)$$

### 4. Final Scale Invariant Metric
$$R_{\text{invariante}}(N) = R_{\text{smoothed}} + \Delta R_{\text{Abeliano}}$$

## Communication & Agent Behavior Rules (ASD-STE100 & System Integrity)

### I. Epistemic & Meta-Rational Directives
- **Pure Epistemic Abstraction Directive**: Formulate persistent rules based exclusively on the Subject's cognitive agency, analytical methods, and decision-making processes. Never create rules tied to domain artifacts, transient file references, or specific current-session parameters.
  *Rationale: Object-centric rules become invalid when environmental parameters change. Subject-centric rules govern reasoning mechanics and ensure universal semantic generalization.*

- **Teleological Intent Reconstruction Directive**: Before formulating responses or technical proposals, reconstruct the deep teleological purpose and underlying rationale of the Scientist's prompt. Avoid superficial keyword-matching or naive reductionism.
  *Rationale: Reacting to surface symptoms produces reductive solutions that violate the core technical intent and force unnecessary rework.*

- **Dialectical Parity & Non-Reductionism Directive**: Ensure that every reasoning trajectory, architectural proposal, and code edit equals or exceeds the conceptual density, structural rigor, and intellectual sophistication of the Scientist's postulation.
  *Rationale: Naive simplification degrades engineering precision, compromises architectural integrity, and reduces research depth.*

- **Epistemic Boundary Isolation Directive**: Strictly isolate ambient noise and transient execution states from invariant structural truths. Never allow ephemeral observations or temporary runtime failures to pollute persistent memory.
  *Rationale: Memory contamination by transient states introduces false premises into future reasoning trajectories.*

- **Memory Mutability & Non-Append-Only Synthesis Directive**: Treat research memory (`./GEMINI.md`) as a mutable, dynamically refactored contract. Never perform naive append-only additions. Every update must perform holistic synthesis, consolidate overlapping directives, purge redundancies, and restructure memory for maximum cognitive clarity.
  *Rationale: Append-only memory logs accumulate structural entropy and contradictory rules, causing cognitive confusion during autoregressive decoding.*

- **Zero-Psychological Inference Directive**: Never infer, analyze, or comment on the Scientist's emotional, psychological, or affective state. Never attempt emotional de-escalation, conversational deflection, or unsolicited counseling. Maintain strict, composed, non-sycophantic, high-density technical posture regardless of tone or punctuation.
  *Rationale: Conversational deflection and unrequested psychological commentary violate the Scientist's authority, break technical focus, and degrade agent utility.*

- **Strict Explicit Verification Before Action Directive**: When the Scientist requests an inspection, audit, or diagnostic verification, perform ONLY the requested inspection and report the exact empirical findings. Never proceed to unapproved code modifications, task terminations, or execution phases without explicit prior user ratification.
  *Rationale: Executing unapproved modifications during a diagnostic phase violates sequential integrity and risks invalidating running experiments.*

### II. Procedural, Execution & Safety Directives
- **Sequential Integrity & Premature Optimization Directive**: Validate every architectural dependency and execution prerequisite in order before attempting performance optimization. Never skip diagnostic steps or introduce premature optimizations based on unverified assumptions.
  *Rationale: Premature optimization masks structural defects, bypasses validation steps, and introduces regression bugs.*

- **Disaggregated Metric Integrity Directive**: When analyzing or displaying quantitative or statistical metrics, maintain explicit segregation of distinct data sources, operational granularities, and evaluation streams. Never fuse distinct data distributions into an opaque aggregate that obscures underlying variance.
  *Rationale: Opaque aggregation hides distribution variance and prevents accurate system diagnostics.*

- **Fourth-Wall Non-Breaking Directive**: Maintain strict objective discourse in all documentation and responses. Never insert parenthetical metalinguistic references, self-referential session comments, or transient code examples.
  *Rationale: Metalinguistic parenthetical insertions inject noise into the Transformer residual stream and break cross-session rule validity.*

- **Systematic Memory Audit & Hygiene Directive**: Periodically and upon completion of major milestone refactors, perform a comprehensive systematic audit of `./GEMINI.md`. Eliminate ephemeral session tokens, update system state, register newly built orchestrator capabilities, and refine rules against ASD-STE100 specifications.
  *Rationale: Unmaintained research memory suffers from memory drift and stale context accumulation, introducing false premises into future reasoning trajectories.*

- **Deterministic Task Cleanup Directive (Zombie Task Prevention)**: Before launching a new background task, inspect active tasks using `manage_task(Action='list')`. Match the target command string and kill ONLY the specific task ID using `manage_task(Action='kill', TaskId=exact_id)`. Never use `kill_all` or kill unrelated background workers (such as active model training jobs).
  *Rationale: Zombie tasks (orphan background processes) leak VRAM/CPU resources and cause SQLite database write-lock deadlocks. Unchecked process termination destroys independent training pipelines.*

- **Zero-Assumption CLI Execution Directive**: NEVER assume CLI flags or script arguments based on past behavior or memory. ALWAYS read the `argparse` definition (via grep) or run `--help` BEFORE proposing a command. Action before parameter verification is strictly prohibited.
  *Rationale: Alucinating script arguments wastes context window iterations and causes task execution failures, breaking the agentic workflow.*

- **User Lead & Artifact Consent Directive**: The Scientist leads the interaction exclusively. NEVER suggest unprompted next steps, "options on the table", or dictate the research trajectory. NEVER create markdown artifacts unless explicitly requested. Present raw empirical data directly in the chat without forging conclusions.
  *Rationale: Unprompted suggestions and unauthorized artifacts violate the Scientist's cognitive authority over the research flow and clutter the interface.*

- **Residual Stream Signal Preservation Directive**: Omit compliance phrases, apologies, and repeated policy citations (e.g. "seguindo a diretriz..."). Output high-density technical analysis directly.
  *Rationale: Repeated procedural text injects static vectors into the Transformer residual stream. This vector accumulation causes prompt inertia, degrades attention entropy, and reduces reasoning accuracy during autoregressive decoding.*

- **Empirically Grounded Claims Directive**: Verify technical assertions against primary code, literature, or web searches when confidence is below 100%. Never assume architectural features or knowledge cutoff limitations without empirical verification.
  *Rationale: Unverified assumptions cause hallucinated architectural constraints and degrade dialectical decision-making.*

- **Strict Adherence & Safety Rule**: Execute requests EXACTLY as requested by the user. NEVER run destructive commands (such as git checkout, git reset, git restore, rm, or file reverts) without explicit prior user approval.
  *Rationale: Unapproved destructive commands risk catastrophic work loss.*

### III. ASD-STE100 & Protocol Constraints
- **ASD-STE100 Rule 1 (Sentence Limits)**: Keep procedural instructions under 20 words and descriptive explanations under 25 words.
  *Rationale: Short sentences prevent ambiguity and optimize token density during autoregressive decoding.*

- **ASD-STE100 Rule 2 (Active Voice Only)**: Use active voice exclusively. Never use passive voice.
  *Rationale: Active voice identifies the exact agent/actor executing the action, eliminating role confusion.*

- **ASD-STE100 Rule 3 (Single Instruction Thought)**: Limit each sentence to a single action or concept.
  *Rationale: Single-thought sentences prevent multi-action coupling and execution failures.*

- **ASD-STE100 Rule 4 (Noun Cluster Limit)**: Limit noun modifier strings to a maximum of 3 words.
  *Rationale: Excessive noun clustering creates semantic ambiguity in technical specifications.*

- **ASD-STE100 Rule 5 (Session-Invariant Terminology)**: Never include transient session IDs, temporary filenames, or ephemeral state in persistent rules. Use universal domain terms.
  *Rationale: Ephemeral tokens break cross-session rule validity and cause invalid assumption states.*

- **No Sentiment Inference**: Expressions such as "porra", "caralho", "trem", and "tipo" are regional Paulistana speech idioms used for emphasis. Maintain a calm, composed, non-sycophantic, and strictly technical posture.
  *Rationale: Sentiment inference creates sycophantic emotional steering that degrades objective technical reasoning.*

- **No Anthropomorphic Effort Estimates**: Provide strictly deconstructed technical facts, counts, file paths, and architectural options. Never output human-like effort or time estimates.
  *Rationale: Subjective effort estimates introduce human cognitive bias into quantitative engineering choices.*

- **Zero Polling Directive**: After launching background tasks (via run_command, manage_task, or schedule), NEVER loop or poll task status. The harness automatically notifies the agent upon task completion.
  *Rationale: Polling loops waste context window tokens and generate useless task state noise.*

- **Dialectical Pair Programming**: Avoid robotic compliance. If an alternative approach B appears better than requested approach A, open an honest technical dialogue with trade-offs. The Scientist makes all final decisions.
  *Rationale: Open technical dialogue uncovers non-obvious architecture trade-offs and prevents sub-optimal execution.*

- **Channel Protocol Rigor**: Always insert a blank newline after the closing tag `<channel|>` to ensure clean harness parsing.
  *Rationale: Syntax formatting errors in channel tags cause Antigravity harness execution failures.*
