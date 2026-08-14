# RFC-20260814: Master Technical Handoff & Sovereign Research Blueprint

**Project**: Kaggle Pokémon TCG AI Battle Challenge  
**Team**: Fitalabs AI Research  
**Authors**: Research Director & Alefita (Lead AI Scientist)  
**Date**: August 14, 2026  
**Document Classification**: Technical RFC / IEEE Convention Paper / Master Knowledge Index  
**Target Ingestion Models**: GPT-5.6 Sol, DeepSeek-V4-Pro, Codex, Claude 3.7 Sonnet / Flash  

---

## Executive Summary & Abstract

This document establishes the sovereign, authoritative technical handoff for the **Pokémon TCG AI Battle Challenge** codebase. It bridges four weeks of empirical research, mathematical proofs, architectural breakthroughs, and agentic governance evolution conducted on Apple Silicon (M3 Pro 24GB Unified Memory). 

The goal of this repository is to build and deploy an autonomous, competitive neural policy capable of dominating the **"Locked Meta" Evaluation Phase (August 16–31, 2026)** of the Kaggle Pokémon TCG Challenge.

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                                 FITALABS RESEARCH EVOLUTION                                       |
|                                                                                                   |
|  [July 27] first_sub (67.16% WR) ───► [Aug 03] Curriculum V1 (BC Pretraining)                      |
|                                                    │                                              |
|                                       [Aug 12] FP16 Precision Crisis                              |
|                                       (Underflow 3.3% WR -> FP32 Hash Migration)                  |
|                                                    │                                              |
|                                       [Aug 12] Ablation Matrix (420 matches/stage)                |
|                                       (Discovery: Val Acc decoupled from Win Rate)                |
|                                                    │                                              |
|                                       [Aug 13] 3-Tier Idempotent ETL & Abelian Elo                |
|                                       (139,783 matches synced, 0 residual discrepancy)            |
|                                                    │                                              |
|                                       [Aug 14+] Magnum Opus MoE + RoPEND + Metanoia Suite         |
|                                       (Apex Mode Activation for Locked Meta)                      |
+───────────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 1. Antigravity Suite & System Environment Map

The incoming Lead Scientist must leverage the following persistent filesystem paths to access raw transcripts, logs, and database states without operational bias:

### 1.1. Antigravity Brain & Execution Logs
* **Conversation ID**: `9189fa2e-93c2-4a04-9bf0-6d090880de27`
* **Artifact Directory Path**:  
  `/Users/alefita/.gemini/antigravity-cli/brain/9189fa2e-93c2-4a04-9bf0-6d090880de27`
* **Transcript Log (Compact JSONL)**:  
  `/Users/alefita/.gemini/antigravity-cli/brain/9189fa2e-93c2-4a04-9bf0-6d090880de27/.system_generated/logs/transcript.jsonl`
* **Full Transcript Log (Untruncated Raw)**:  
  `/Users/alefita/.gemini/antigravity-cli/brain/9189fa2e-93c2-4a04-9bf0-6d090880de27/.system_generated/logs/transcript_full.jsonl`
* **Scratch Probes & Diagnostic Scripts**:  
  `/Users/alefita/.gemini/antigravity-cli/brain/9189fa2e-93c2-4a04-9bf0-6d090880de27/scratch/`

### 1.2. TensorBoard Experiment Runs (`runs/`)
The repository contains 23 distinct TensorBoard logging directories tracking loss curves, auxiliary heads, gradient norms, and learning rates:
* **Curriculum V1 Training Stages**:
  * `runs/curriculum_v1_stage1_1786173005`, `runs/curriculum_v1_stage1_1786208292`, `runs/curriculum_v1_stage1_1786220216`, `runs/curriculum_v1_stage1_1786220595`
  * `runs/curriculum_v1_stage2_1786409797`, `runs/curriculum_v1_stage2_1786410351`
  * `runs/curriculum_v1_stage3_1786448581`
  * `runs/curriculum_v1_stage4_1786515473`
  * `runs/curriculum_v1_stage5_1786544951`
* **MLX Native Baseline Runs**:
  * `runs/bc_best_mlx_1786512621`, `runs/bc_best_mlx_1786513169`
* **Temporal / Dataset Window Ablations**:
  * `runs/1d_10ep_OFF_1786055550`, `runs/1d_10ep_ON_1786057164`
  * `runs/3d_1ep_OFF_1786059065`, `runs/3d_1ep_ON_1786059784`, `runs/3d_10ep_OFF_1786060568`, `runs/3d_10ep_ON_1786067401`
  * `runs/5d_1ep_OFF_1786076260`, `runs/5d_1ep_ON_1786077401`, `runs/5d_10ep_OFF_1786078697`, `runs/5d_10ep_ON_1786088444`, `runs/5d_10ep_ON_1786088982`, `runs/5d_10ep_ON_1786090062`

### 1.3. Persistent Storage & Database
* **SQLite Core**: `model/results.db` (Schema Version: `2.0.0`, 139,783 indexed matches).
* **Live Kaggle Leaderboard Cache**: `data/kaggle_leaderboard.csv` (Enforces 28h physical TTL).
* **Replay Dumps**: `data/bc_replay_zip/` (31 daily archives from `2026-07-14.zip` to `2026-08-12.zip`, ~138,138 raw JSON episodes).

---

## 2. Lineage & Provenance of Historical Documents (`CLAUDE.md`)

An exhaustive audit of `CLAUDE.md` establishes its exact historical chronology:

```
2026-07-23 ────────────────► 2026-07-25 ────────────────► 2026-07-16..22 ──────────► 2026-08-06..07 ──────────► 2026-08-07 ──────────► 2026-08-08+
PyTorch Reference            Repo Initialization          Historical BC Pipeline       BC Curriculum Suite         Current Phase Log        Antigravity Migration
`reference/mikaelzinho-      `tcg-pokemon-agent-          - Checkpoints v1, v2         Ablations (1d-5d,           - Strict Split (MLX      - Living `GEMINI.md`
pytorch/`                    mlx-port.zip`                - Submissions (889.7 LB)     1-10ep, top-elo)            Train / PyTorch Infer)   - `.agents/rules/`
Drive ID: 1IwESPm29...       Drive ID: 1R3wCNKX...        - M1 Air 8GB CPU bounds      - Best: 5d_10ep_OFF (21%)   - Parquet KV Cache       - Metanoia Suite
```

### Provenance Audit Milestones
1. **2026-07-23 (PyTorch Reference Snapshot)**: Sourced from `tcg-pokemon-agent-main.zip` (Drive ID: `1IwESPm29-6bGByS6qHcrPchhR2-QjEld`), vendored locally at `reference/mikaelzinho-pytorch/` as read-only architectural reference.
2. **2026-07-25 (Repository Initialization)**: Sourced from `tcg-pokemon-agent-mlx-port.zip` (Drive ID: `1R3wCNKXlnJ5jEHbYtqkjyQ_aOkxvEe_x`), migrating PyTorch to Apple Silicon MLX.
3. **2026-08-07 (Phase Log Freeze)**: Completed the BC Curriculum Suite, established the MLX Training / PyTorch Inference split, implemented the Parquet KV Cache, and froze `CLAUDE.md` as an authoritative historical archive.
4. **2026-08-08 onwards (Antigravity Suite Migration)**: Shifted governance from static markdown (`CLAUDE.md`) to the self-evolving cognitive contract (`GEMINI.md`), modular rule enforcement (`.agents/rules/`), and the Metanoia cognitive swarm framework.

---

## 3. Deep Technical Architecture Modules (The 3 Levels of Depth)

For deep technical implementation specifications, consult the following dedicated monographs:

### Level 1 Depth: Neural Engine & Tokenization
Detailed specification of tensor shapes, Vortex stream aggregation, Muon Newton-Schulz iterations, and TBPTT scratch register recurrence.
* Reference: [`docs/neural_engine_and_tokenization_spec.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/neural_engine_and_tokenization_spec.md)

### Level 2 Depth: Dataset Compilation & Oracle Pipeline
Detailed specification of replay ingestion, off-by-one pointer shifts, C++ engine oracle simulations (`would_ko`), telescoping backward rewards, and Parquet KV Cache memory management.
* Reference: [`docs/dataset_compilation_and_oracle_pipeline.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/dataset_compilation_and_oracle_pipeline.md)

### Level 3 Depth: Empirical Ablations & Game-Theoretic Meta Analysis
Deep empirical analysis of the 420-match cross-stage tournament, mathematical proof of the decoupling between validation accuracy and game-theoretic win rate, and the FP16 underflow root-cause analysis.
* Reference: [`docs/empirical_ablation_monograph.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/empirical_ablation_monograph.md)

---

## 4. The Metanoia Suite: Agentic Architecture & Governance

The meta-analysis of the agentic workflow and harness engineering is documented in the `docs/metanoia/` subfolder:

1. **The Channel Protocol & Cognitive Swarm**: Formal state-machine definition (`INIT` $\to$ `GENERATE` $\to$ `DEBATE` $\to$ `RANK` $\to$ `EVOLVE` $\to$ `META_REVIEW`), anti-pollution boundaries, and DeepMind Co-Scientist mathematical correspondence.  
   * Reference: [`docs/metanoia/01_channel_protocol_and_cognitive_swarm.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/metanoia/01_channel_protocol_and_cognitive_swarm.md)

2. **Rule Provenance & Epistemic Evolution**: Philosophical laws (Poincaré Incubation, Parity Law, Metanoia, Zero-Trust), ASD-STE100 specifications, and transition from `CLAUDE.md` to `GEMINI.md`.  
   * Reference: [`docs/metanoia/02_rule_provenance_and_epistemic_evolution.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/metanoia/02_rule_provenance_and_epistemic_evolution.md)

3. **Model Adherence & Failure Mode Pathology**: Empirical evaluation of model families (Gemini 3.1 Pro, 3.5 Flash, 3.6 Flash vs. 3.7 Flash High), KaTeX-Markdown collision mechanics, channel preamble leakage, and sycophancy expungement.  
   * Reference: [`docs/metanoia/03_model_adherence_and_failure_mode_analysis.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/metanoia/03_model_adherence_and_failure_mode_analysis.md)

4. **Tensorized Scaling & Subagent Swarms**: 3D scaling tensor (Vertical reasoning depth, Horizontal subagent swarms, Orthogonal domain isolation), context redaction with provenance ledgers (Buzz / ArXiv:2608.09867 / Headroom), and Terence Tao's mathematical counterproof parallels.  
   * Reference: [`docs/metanoia/04_tensorized_scaling_and_subagent_orchestration.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/metanoia/04_tensorized_scaling_and_subagent_orchestration.md)

---

## 5. Mathematical Framework: Sample-Size Invariant Elo

The system rejects raw Elo in favor of the **Sample-Size Invariant Elo**:

### 1. Bradley-Terry Asymptotic Logistic Inversion
Given win rate $w = \frac{W}{N}$, clipped to $[0.02, 0.98]$:

$$
\hat{R}_{\infty} = 600.0 + 400.0 \cdot \log_{10}\left( \frac{w}{1 - w} \right)
$$

### 2. MD10 Placement Smoothing ($N_0 = 10$)
Regularizes small-sample estimates ($N < 10$) toward the base prior $R_0 = 600.0$:

$$
R_{\text{smoothed}} = \left(\frac{N}{N + 10}\right) \cdot \hat{R}_{\infty} + \left(\frac{10}{N + 10}\right) \cdot 600.0
$$

### 3. Softmax Abelian Group Translation ($\Delta R_{\text{Abeliano}}$)
Computes the global translation isomorphism bridging local tournament performance to the official Kaggle ladder scale across overlapping deck set $\mathcal{C}$:

$$
\alpha_k = \frac{\exp(N_k / 20.0)}{\sum_{j \in \mathcal{C}} \exp(N_j / 20.0)}
$$

$$
\Delta R_{\text{Abeliano}} = \sum_{k \in \mathcal{C}} \alpha_k \cdot \left( R_k^{\text{remote}} - \hat{R}_{k,\infty}^{\text{local}} \right)
$$

### 4. Final Scale Invariant Metric

$$
R_{\text{invariante}}(N) = R_{\text{smoothed}} + \Delta R_{\text{Abeliano}}
$$

---

## 6. Comprehensive Documentation & Knowledge Index

| Document | File Path | Scope & Focus |
| :--- | :--- | :--- |
| **Master Handoff RFC** | [`docs/technical_handoff_rfc.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/technical_handoff_rfc.md) | Sovereign master index and end-to-end technical specification |
| **Level 1: Neural Engine Spec** | [`docs/neural_engine_and_tokenization_spec.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/neural_engine_and_tokenization_spec.md) | Concrete tensor shapes, Vortex stream math, Muon split optimization |
| **Level 2: Dataset & Oracle Spec** | [`docs/dataset_compilation_and_oracle_pipeline.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/dataset_compilation_and_oracle_pipeline.md) | ETL realignments, C++ would_ko oracles, KV cache, RoPEND schema |
| **Level 3: Empirical Ablations** | [`docs/empirical_ablation_monograph.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/empirical_ablation_monograph.md) | 420-match matrix, Val Acc decoupling proof, FP16 numerical underflow |
| **Metanoia 01: Channel Protocol** | [`docs/metanoia/01_channel_protocol_and_cognitive_swarm.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/metanoia/01_channel_protocol_and_cognitive_swarm.md) | State machine, Co-Scientist compression, anti-pollution rules |
| **Metanoia 02: Rule Provenance** | [`docs/metanoia/02_rule_provenance_and_epistemic_evolution.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/metanoia/02_rule_provenance_and_epistemic_evolution.md) | Epistemological laws, ASD-STE100, CLAUDE.md to GEMINI.md lineage |
| **Metanoia 03: Model Adherence** | [`docs/metanoia/03_model_adherence_and_failure_mode_analysis.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/metanoia/03_model_adherence_and_failure_mode_analysis.md) | Failure mode taxonomy across Gemini 3.1..3.7 families |
| **Metanoia 04: Tensorized Scaling** | [`docs/metanoia/04_tensorized_scaling_and_subagent_orchestration.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/metanoia/04_tensorized_scaling_and_subagent_orchestration.md) | 3D cognitive tensor, Buzz/Headroom context ledger, Tao counterproofs |
| **Academic Monograph** | [`docs/Pokemon_TCG_AI_Monograph.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/Pokemon_TCG_AI_Monograph.md) | 8-chapter complete project monograph |
| **Abelian Group Elo Formulation** | [`docs/abelian_group_elo_formulation.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/abelian_group_elo_formulation.md) | Algebraic proof of Bradley-Terry translation invariance |
| **SQLite Schema & ERD** | [`docs/database_schema.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/database_schema.md) | Full Mermaid ERD and Schema 2.0.0 table definitions |
| **ETL Architecture & Auditing** | [`docs/etl_architecture_and_auditing.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/etl_architecture_and_auditing.md) | Zero-Trust disk auditing and 3-Tier idempotency model |
| **Entity Normalization Heuristics** | [`docs/normalization_heuristics.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/normalization_heuristics.md) | Cantor diagonal L1/L2 entity resolution rules |
| **Kaggle Platform Dynamics** | [`docs/kaggle_platform_dynamics.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/kaggle_platform_dynamics.md) | Analysis of the 95% missing match export bias |
| **MoE Pipeline Blueprint** | [`docs/architecture/moe_pipeline_blueprint.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/architecture/moe_pipeline_blueprint.md) | MoE, RoPEND, and Apex Mode architectural contract |
| **Modernized PTCG Results Skill** | [`.agents/skills/ptcg-results-api/SKILL.md`](file:///Users/alefita/workdir/pokemon-tcg/.agents/skills/ptcg-results-api/SKILL.md) | Updated skill for Schema 2.0.0, Invariant Elo, 28h TTL |
| **Modernized MoE Architecture Skill** | [`.agents/skills/ptcg-moe-architecture/SKILL.md`](file:///Users/alefita/workdir/pokemon-tcg/.agents/skills/ptcg-moe-architecture/SKILL.md) | Updated skill for 4D RoPEND, Draft, Apex Mode |
