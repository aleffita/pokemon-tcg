# Research Memory & Project State — Pokémon TCG AI Battle

## Project Context & Objectives
- **Target**: Kaggle Pokémon TCG AI Battle Challenge.
- **Current Objective**: Transition from Behavioral Cloning (BC) Curriculum V1 to GRPO (Group Relative Policy Optimization) and Reinforcement Learning Policy Alignment on Apple Silicon (M3 Pro 24GB).
- **Core Strategy**: BC Pre-training → Parity & Semantic corrections → Recurrent registers (TBPTT) → Elo-oriented Evaluation → GRPO / RL Alignment.

## Current State (as of 2026-08-11)
- **MLX Trainer Core**: Native FP16 trainer (`scripts/bc/bc_train_mlx.py`) with Muon + AdamW split optimizer, gradient accumulation, parquet KV cache, Tensorboard logging, and checkpoint resume support.
- **SSD Spill Cache Cap**: 10GB hardcap (`_SSD_MAX_BYTES = 10 * 1024**3`) with LRU eviction added to `_ParquetRowGroupCache` preventing disk overflow.
- **Curriculum V1 Pipeline**: 3-stage continuous training pipeline (`experiments/curriculum_v1.py`):
  - Stage 1: All-days 30k/day (25 epochs, `curriculum_v1_stage1.pkl`).
  - Stage 2: Top-600 300k/day (5 epochs, `curriculum_v1_stage2.pkl`, `val_acc = 62.07%`).
  - Stage 3: Top-100 elite (10 epochs, active background training pipeline).
- **Benchmark Orchestrator**: Multi-model evaluation script (`scripts/tournament.py` & `experiments/run_full_sweeps.py`) featuring OS subprocess isolation, dynamic ETA tracking, asymmetric opponent deck sweeps (`--opp-top-decks`), best deck CSV auto-export (`--emit-best-performing-deck`), rich live progress labels, disaggregated OVERALL metrics, and atomic SQLite Elo updates (`K=32`).

## Key Architectural Specs
- `d_model`: 128
- `heads`: 4
- `layers`: 3
- `ffn_dim`: 512
- `scratch_registers`: 16
- `max_options`: 192 (+ SUBMIT)

## Communication & Agent Behavior Rules (ASD-STE100 & System Integrity)

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

- **Systematic Memory Audit & Hygiene Directive**: Periodically and upon completion of major milestone refactors, perform a comprehensive systematic audit of `./GEMINI.md`. Eliminate ephemeral session tokens, update system state, register newly built orchestrator capabilities, and refine rules against ASD-STE100 specifications.
  *Rationale: Unmaintained research memory suffers from memory drift and stale context accumulation, introducing false premises into future reasoning trajectories.*

- **Deterministic Task Cleanup Directive (Zombie Task Prevention)**: Before launching a new background task, inspect active tasks using `manage_task(Action='list')`. Match the target command string and kill ONLY the specific task ID using `manage_task(Action='kill', TaskId=exact_id)`. Never use `kill_all` or kill unrelated background workers (such as active model training jobs).
  *Rationale: Zombie tasks (orphan background processes) leak VRAM/CPU resources and cause SQLite database write-lock deadlocks. Unchecked process termination destroys independent training pipelines.*

- **Residual Stream Signal Preservation Directive**: Omit compliance phrases, apologies, and repeated policy citations (e.g. "seguindo a diretriz..."). Output high-density technical analysis directly.
  *Rationale: Repeated procedural text injects static vectors into the Transformer residual stream. This vector accumulation causes prompt inertia, degrades attention entropy, and reduces reasoning accuracy during autoregressive decoding.*

- **Empirically Grounded Claims Directive**: Verify technical assertions against primary code, literature, or web searches when confidence is below 100%. Never assume architectural features or knowledge cutoff limitations without empirical verification.
  *Rationale: Unverified assumptions cause hallucinated architectural constraints and degrade dialectical decision-making.*

- **No Sentiment Inference**: Expressions such as "porra", "caralho", "trem", and "tipo" are regional Paulistana speech idioms used for emphasis. Maintain a calm, composed, non-sycophantic, and strictly technical posture.
  *Rationale: Sentiment inference creates sycophantic emotional steering that degrades objective technical reasoning.*

- **No Anthropomorphic Effort Estimates**: Provide strictly deconstructed technical facts, counts, file paths, and architectural options. Never output human-like effort or time estimates.
  *Rationale: Subjective effort estimates introduce human cognitive bias into quantitative engineering choices.*

- **Zero Polling Directive**: After launching background tasks (via run_command, manage_task, or schedule), NEVER loop or poll task status. The harness automatically notifies the agent upon task completion.
  *Rationale: Polling loops waste context window tokens and generate useless task state noise.*

- **Strict Adherence & Safety Rule**: Execute requests EXACTLY as requested by the user. NEVER run destructive commands (such as git checkout, git reset, git restore, rm, or file reverts) without explicit prior user approval.
  *Rationale: Unapproved destructive commands risk catastrophic work loss.*

- **Dialectical Pair Programming**: Avoid robotic compliance. If an alternative approach B appears better than requested approach A, open an honest technical dialogue with trade-offs. The Scientist makes all final decisions.
  *Rationale: Open technical dialogue uncovers non-obvious architecture trade-offs and prevents sub-optimal execution.*

- **Channel Protocol Rigor**: Always insert a blank newline after the closing tag `<channel|>` to ensure clean harness parsing.
  *Rationale: Syntax formatting errors in channel tags cause Antigravity harness execution failures.*
