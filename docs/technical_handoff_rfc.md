# RFC-20260814: Comprehensive Technical Handoff & Epistemological Blueprint
**Project**: Kaggle Pokémon TCG AI Battle Challenge  
**Team**: Fitalabs AI Research  
**Authors**: Research Director & Alefita (Lead AI Scientist)  
**Date**: August 14, 2026  
**Document Classification**: Technical RFC / IEEE Convention Paper / System Handoff  
**Target Ingestion Models**: GPT-5.6 Sol, DeepSeek-V4-Pro, Codex, Claude 3.7 Sonnet / Flash  

---

## Executive Summary & Abstract

This document establishes the authoritative, end-to-end technical handoff for the **Pokémon TCG AI Battle Challenge** codebase. It bridges four weeks of empirical research, architectural refactoring, and machine learning breakthroughs conducted on Apple Silicon (M3 Pro 24GB Unified Memory). 

The mission of this repository is to engineer an autonomous, competitive neural agent capable of dominating the **"Locked Meta" Evaluation Phase (August 16–31, 2026)** of the Kaggle Pokémon TCG Challenge.

```
+---------------------------------------------------------------------------------------------------+
|                                 FITALABS RESEARCH EVOLUTION                                       |
|                                                                                                   |
|  [July 27] first_sub (67.16% WR) ---> [Aug 03] Curriculum V1 (BC Pretraining)                      |
|                                                    |                                              |
|                                       [Aug 12] FP16 Precision Crisis                              |
|                                       (Underflow 3.3% WR -> FP32 Hash Migration)                  |
|                                                    |                                              |
|                                       [Aug 12] Ablation Matrix (420 matches/stage)                |
|                                       (Discovery: Val Acc decoupled from Win Rate)                |
|                                                    |                                              |
|                                       [Aug 13] 3-Tier Idempotent ETL & Abelian Elo                |
|                                       (139,783 matches synced, 0 residual discrepancy)            |
|                                                    |                                              |
|                                       [Aug 14+] Magnum Opus MoE + RoPEND + GRPO                   |
|                                       (Apex Mode Activation for Locked Meta)                      |
+---------------------------------------------------------------------------------------------------+
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

## 2. Chronological Git & Research Trajectory (July 27 – August 14, 2026)

### Phase I: Inception & The First Sub Anchor (July 27)
* **Milestone**: Deployment of `first_sub_kaggle_2707.tar.gz`.
* **Empirical Outcome**: Achieved a verified **67.16% Win Rate** on the Kaggle public leaderboard (Leaderboard Score ~1200+).
* **Role**: Serves as our permanent **Teacher Model** and empirical benchmark anchor for all ablation tournaments.

### Phase II: Behavioral Cloning Curriculum V1 (July 28 – August 07)
* **Architecture**: 4-Layer Transformer Decoder, $D=128$, 19 Type Embeddings, Muon + AdamW split optimizer, streaming TBPTT over Parquet files.
* **Curriculum Design**:
  * *Stage 1*: Full historical replay corpus (all Elo ratings).
  * *Stage 2*: High-Elo replays (Kaggle score $\ge 600$).
  * *Stage 3*: Elite tier replays (Top 100 ladder).
  * *Stage 4*: Loss-corrected Top 100 fine-tuning (5 epochs).
* **The Auxiliary Loss Corruption Bug**: During Stages 1–3, auxiliary classification heads (winner prediction, prize delta, would_ko) had misaligned loss scaling, backpropagating chaotic gradients into the shared trunk. Stage 4 was retrained with corrected loss weights.

### Phase III: The FP16 Precision Crisis (August 12)
* **Incident**: The native MLX model scored 45.0% WR vs random baseline. When exported to PyTorch (`build_submission.py`), win-rate collapsed catastrophically to **3.3% WR** (0.0% vs `first_sub`).
* **Root Cause Diagnosis**: Because the neural network is compact (~15MB), FP16 quantization during PyTorch inference caused numerical underflow in attention Softmax and LayerNorm, zeroing decision logits.
* **Remediation**:
  1. Converted PyTorch inference (`rl/policy_infer_torch.py`) to strict **`float32`**.
  2. Updated `scripts/bc/bc_train_mlx.py` to generate SHA256 contract hashes over FP32 static feature arrays.
  3. Surgically repacked all stage checkpoints (`stage1_fp32.tar.gz` to `stage4_fp32.tar.gz`) to update the cryptographic contract without losing learned weights.
  4. Restored win-rate to 35.0%–45.0% immediately.

### Phase IV: The Cross-Stage Ablation Matrix (August 12–13)
* **Tournament Setup**: Orchestrated a massive $3 \times 5$ deck sweep (420 matches per stage) pitting Stages 1, 2, 3, and 4 against the Top 5 decks of `first_sub`.
* **Results**:
  * Stage 1: 14.3% Overall WR (Peak: 22.9% on Yan Deck #633).
  * Stage 2: 15.2% Overall WR (Peak: 26.4% on Yan Deck #633).
  * Stage 3: 13.8% Overall WR (Peak: 22.9% on Yan Deck #633).
  * **Stage 4: 17.1% Overall WR (Peak: 27.9% on Yan Deck #633)**.
* **The Epistemological Breakthrough (The "Verstappen in an Aston Martin" Thesis)**:
  * Pure Behavioral Cloning saturated at 17.1% vs `first_sub`.
  * **Validation Accuracy and Win Rate are decoupled**: A model can achieve 78% next-token prediction accuracy imitating mediocre human play while getting crushed by an optimal agent.
  * **Vehicle vs. Pilot**: Deck composition dictates the theoretical win ceiling. Deck #633 achieved 27.9% WR, whereas the submission default Deck #251 scored only 12.9% WR under the exact same neural weights.

### Phase V: Entity Normalization & Kaggle Sampling Discovery (August 13)
* **The Problem**: 144 high-volume teams abruptly stopped appearing in replay dumps prior to the August 10 merge deadline, raising fears of mass bans (~12% platform collapse).
* **The Cantor Diagonal & Footprint Resolution (L1/L2)**:
  * L1 (Leaderboard Cross-Match): Proved that **142 out of 144 teams** simply merged into consolidated teams (e.g., Fitalabs), causing Kaggle's 2-active-submission limit to deactivate their legacy `EpisodeIds`.
  * L2 (Deck Footprint Correlation): Resolved 1 additional team via 60-card array fingerprint tracking.
  * Only 1 true extinction event remained across the entire competition ("Dieter", 1,068 matches).
* **Kaggle Sampling Bias**: Discovered that Kaggle only exports **1.05% to 5.57%** of daily ladder matches. 95% of active games remain unexported.

### Phase VI: 3-Tier Idempotent ETL & Abelian Elo (August 13–14)
* **Parity Crisis**: Physical audit revealed 4,968 missing matches in `results.db` due to previous ETL script crashes.
* **Refactoring**: Built a 3-tier idempotent synchronization engine with a 28h TTL Kaggle API cache.
* **Outcome**: Ingested all missing matches, achieving **100.0% exact parity** (139,783 matches in SQLite = 139,783 JSONs on disk).

---

## 3. Data Engineering & ETL Architecture

### 3.1. 3-Tier Synchronization Engine (`scripts/build_card_stats.py`)
To eliminate $O(N)$ filesystem scanning and prevent CPU deadlocks during bulk ingestion:

```
[Sync Entrypoint: tcg-build-card-stats]
   │
   ├── 1. db.sync_kaggle_leaderboard(ttl_hours=28)
   │      ├── Check mtime on data/kaggle_leaderboard.csv
   │      └── Age < 28h ? Read Cache (0 API calls) : Fetch API & Persist
   │
   ├── 2. Daily Archive Ingestion Loop:
   │      ├── Tier 1 (Macro Fast-Skip): COUNT(matches) == len(zip) ? Skip ZIP (0.001s)
   │      └── Tier 2 (Micro Delta): If gap exists, fetch existing_members set, 
   │                                parse missing JSONs with recompute_elo=False
   │
   └── 3. Tier 3 (Decoupled Global Recomputation):
          ├── db.compute_daily_elos(source="remote")
          └── db.compute_daily_elos(source="local")
```

### 3.2. Mathematical Framework: Sample-Size Invariant Elo

The system rejects raw Elo ($R_{\text{raw}}$) in favor of the **Sample-Size Invariant Elo ($R_{\text{invariante}}$)**:

#### 1. Bradley-Terry Asymptotic Logistic Inversion
Given win rate $w = \frac{W}{N}$, clipped to $[0.02, 0.98]$:

$$
\hat{R}_{\infty} = 600.0 + 400.0 \cdot \log_{10}\left( \frac{w}{1 - w} \right)
$$

#### 2. MD10 Placement Smoothing ($N_0 = 10$)
Regularizes small-sample estimates ($N < 10$) toward the base prior $R_0 = 600.0$:

$$
R_{\text{smoothed}} = \left(\frac{N}{N + 10}\right) \cdot \hat{R}_{\infty} + \left(\frac{10}{N + 10}\right) \cdot 600.0
$$

#### 3. Softmax Abelian Group Translation ($\Delta R_{\text{Abeliano}}$)
Computes the global translation isomorphism bridging local tournament performance to the official Kaggle ladder scale across overlapping deck set $\mathcal{C}$:

$$
\alpha_k = \frac{\exp(N_k / 20.0)}{\sum_{j \in \mathcal{C}} \exp(N_j / 20.0)}
$$

$$
\Delta R_{\text{Abeliano}} = \sum_{k \in \mathcal{C}} \alpha_k \cdot \left( R_k^{\text{remote}} - \hat{R}_{k,\infty}^{\text{local}} \right)
$$

#### 4. Final Scale Invariant Metric

$$
R_{\text{invariante}}(N) = R_{\text{smoothed}} + \Delta R_{\text{Abeliano}}
$$

---

## 4. Neural Architecture & Training Engine

```
+---------------------------------------------------------------------------------------------------+
|                               TRANSFORMER DECODER TOPOLOGY                                        |
|                                                                                                   |
|  Input State ----> [19 Type Embeddings + Vortex Unit Aggregator]                                 |
|                               |                                                                   |
|                               v                                                                   |
|                    [Bayesian GameTracker Shadows]                                                 |
|                    (drawable_emb, hand_certain_emb)                                               |
|                               |                                                                   |
|                               v                                                                   |
|                    [4x Transformer Decoder Layers] <==== [Scratch Registers / TBPTT Memory]       |
|                    (D=128, N_heads=4, SwiGLU / MLP)                                               |
|                               |                                                                   |
|        +----------------------+----------------------+----------------------+                     |
|        |                      |                      |                      |                     |
|        v                      v                      v                      v                     |
|  [Action Logits]      [Winner Prediction]     [Prize Delta Head]     [Energy Attachment]          |
|  (Cross-Entropy)      (Binary BCE)            (MSE Regression)       (Categorical CE)             |
+---------------------------------------------------------------------------------------------------+
```

### 4.1. Permutation Invariance & Spatial Tokenizer
* **Abolition of 1D Sequence RoPE**: Card order in hand/bench is arbitrary. 1D sequence RoPE was eliminated in favor of **19 Spatial Type Embeddings** (`T_SELF_HAND`, `T_OPP_ACTIVE`, `T_STADIUM`, `T_SELF_BENCH_1..5`, etc.).
* **Vortex Unit Aggregator (`_unit_stream`)**: Condenses a multi-card Pokémon entity (Base + Evolutions + Attached Energies + Tools + Damage Counters + Status) into a single dense token vector:

$$
\mathcal{U}_{\text{Vortex}} = \text{Base}_{\text{emb}} + \sum_{i=1}^{n} \text{PreEvo}_{\text{emb}} + \text{Tool}_{\text{emb}} + \sum_{j=1}^{e} \text{Energy}_{\text{emb}} + \text{UnitProj}(\text{Damage}, \text{Status})
$$

### 4.2. Bayesian GameTracker (Imperfect Information Modeling)
* **Shadow Tokens (`drawable_emb`)**: Assigns probability embeddings to cards remaining in deck based on public game history.
* **Certainty Markers (`hand_certain_emb`)**: Tracks unrevealed opponent cards vs publicly tracked cards, preventing tactical hallucinations against opponent bluffs.

### 4.3. Unified Memory Optimization: `_ParquetRowGroupCache`
To prevent SSD thrashing on Apple Silicon M3 Pro during TBPTT training:
* Implements a 3-tier hierarchical cache: **Hot Zone** (active memory), **Transient Zone** (LRU), and **SSD Spill** (`.cache_spill/`).
* Provides $O(1)$ batch delivery for sequential multi-step episodes.

---

## 5. Magnum Opus Blueprint: RoPEND, MoE & Apex Mode

Designed specifically to exploit the **Locked Meta Phase (August 16–31)**:

### 5.1. N-Dimensional Rotary Positional Embeddings (RoPEND)
Decomposes embedding dimension $D=128$ into 4 orthogonal 32-dim sub-vectors:
1. **$c_1$ (Match Step)**: Discrete game turn progression.
2. **$c_2$ (Meta-Epoch)**: Calendar day offset from competition start.
3. **$c_3$ (Urgency / Clock)**: Normalized remaining game compute time (countdown from 600s).
4. **$c_4$ (Elo / Hierarchy)**: Estimated continuous ranking.

The attention inner product computes relative distance across all 4 axes simultaneously:

$$
\langle \mathbf{q}', \mathbf{k}' \rangle = \sum_{i=1}^4 \mathbf{q}_i^\top R_{\Theta_i, c_i^k - c_i^q} \mathbf{k}_i
$$

### 5.2. Ephemeral Sandbox Stochastic Elo Inference
Because Kaggle sandboxes are stateless (memoryless between games), the agent computes its expected global rank in real time:

$$
R_{\text{internal}} = \alpha (R_0 + f(\Delta T)) + (1 - \alpha) \hat{R}_{\text{opp}}
$$

Where $R_0$ is the hardcoded August 16 anchor Elo, $\Delta T$ is derived via `datetime.now(UTC)`, and $\hat{R}_{\text{opp}}$ is inferred in-game by the opponent modeling auxiliary head.

### 5.3. Apex Mode Trigger (Airgap Strategy)
During inference, when `datetime.now(UTC) >= 2026-08-16`, the agent activates the **Apex Mode Token**, shifting the MoE router from exploratory play to exploitative, deterministic meta-countering.

---

## 6. Open Engineering Roadmap & Next Sprints

### Sprint 1: Clean Base Model (V2) & GRPO Alignment
1. **Dataset Compilation**: Extract the **Elite Match Pool** ($\text{Elo} \ge 1100$, ~100k matches) with corrected auxiliary targets and RoPEND metadata.
2. **Base Training**: Train Clean Base Model V2 (or initialize from Stage 4 FP32).
3. **GRPO / RL Self-Play**: Deploy Group Relative Policy Optimization against `first_sub` to break the 17.1% BC saturation ceiling.

### Sprint 2: MoE Router & Final Submission Packaging
1. **MoE Expansion**: Freeze base representation layers; train Softmax gating network over specialized experts (Aggro, Control, Mirror).
2. **PyTorch Export**: Package into `submission_apex.tar.gz` with FP32 feature contracts.
3. **Verification**: Run `scripts/tournament.py` (500 matches vs `first_sub`) to confirm win rate $> 60\%$.

---

## 7. Artifact & Documentation Index

| File | Purpose |
| :--- | :--- |
| [`docs/Pokemon_TCG_AI_Monograph.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/Pokemon_TCG_AI_Monograph.md) | Comprehensive 8-chapter academic monograph of the project |
| [`docs/abelian_group_elo_formulation.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/abelian_group_elo_formulation.md) | Algebraic proof of Bradley-Terry translation invariance |
| [`docs/database_schema.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/database_schema.md) | Full Mermaid ERD & SQLite Schema 2.0.0 specification |
| [`docs/etl_architecture_and_auditing.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/etl_architecture_and_auditing.md) | Blueprint for Zero-Trust ETL and disk auditing |
| [`docs/normalization_heuristics.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/normalization_heuristics.md) | Cantor diagonal L1/L2 entity resolution rules |
| [`docs/kaggle_platform_dynamics.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/kaggle_platform_dynamics.md) | Analysis of Kaggle's 95% unexported replay sampling bias |
| [`docs/architecture/moe_pipeline_blueprint.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/architecture/moe_pipeline_blueprint.md) | MoE, RoPEND, and Apex Mode architectural contract |
| [`docs/architecture/01_ropend_theory.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/architecture/01_ropend_theory.md) | 4D Rotary Positional Embedding mathematical derivation |
| [`docs/architecture/02_stochastic_elo_inference.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/architecture/02_stochastic_elo_inference.md) | Bayesian in-game Elo estimation formulas |
| [`rl/results_db.py`](file:///Users/alefita/workdir/pokemon-tcg/rl/results_db.py) | Core SQLite declarative database layer & Invariant Elo |
| [`scripts/build_card_stats.py`](file:///Users/alefita/workdir/pokemon-tcg/scripts/build_card_stats.py) | 3-Tier Idempotent ETL synchronization engine |
| [`scripts/tournament.py`](file:///Users/alefita/workdir/pokemon-tcg/scripts/tournament.py) | Multi-model ablation benchmark & tournament runner |
