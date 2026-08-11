# Research Memory & Project State — Pokémon TCG AI Battle

## Project Context & Objectives
- **Target**: Kaggle Pokémon TCG AI Battle Challenge.
- **Current Objective**: Migrate Mikaelzinho PyTorch baseline to MLX port (`rl/policy_mlx.py`, `scripts/bc/bc_train_mlx.py`) on Apple Silicon (M3 Pro 24GB target).
- **Core Strategy**: Behavioral Cloning (BC) → Parity & Semantic corrections → Recurrent registers (TBPTT) → Elo-oriented Evaluation.

## Current State (as of 2026-08-08)
- **MLX Trainer**: Functional FP16-native trainer (`scripts/bc/bc_train_mlx.py`) with Muon/AdamW split optimizer, gradient accumulation, parquet KV cache, Tensorboard logging, and checkpoint resume support.
- **SSD Spill Cache Cap**: Hardcap of 10GB (`_SSD_MAX_BYTES = 10 * 1024**3`) with LRU eviction added to `_ParquetRowGroupCache` to prevent disk overflow during long runs.
- **Script Diagnostics**: `experiments/curriculum_v1.sh` updated with `set -eo pipefail` and `int(time.time())` to fix float arithmetic syntax errors in Bash timing logic.
- **Curriculum Ablation Results**: 10 models trained and benchmarked. Top performer vs public agents: `5d_10ep_OFF` (21.0% win rate overall).
- **Current Submission Candidate**: `model/checkpoint/suite_5d_10ep_OFF/5d_10ep_OFF.pkl` packaged at `experiments/bc_curriculum_suite/models/5d_10ep_OFF.tar.gz`.

## Key Architectural Specs
- `d_model`: 128
- `heads`: 4
- `layers`: 3
- `ffn_dim`: 512
- `scratch_registers`: 4
- `max_options`: 192 (+ SUBMIT)

## Communication & Agent Behavior Rules
- **ASD-STE100 Standard**: Use clear, direct, and non-ambiguous technical language. Use short sentences and simple words. Avoid jargon, fluff, or beat-around-the-bush explanations.
- **Internal Analysis**: Read, analyze, and interpret code, DB schemas, and technical files internally. Synthesize clear explanations without displaying raw SQL dumps or code walls unless explicitly requested.
- **No Unsolicited Assumptions**: Focus strictly on the requested task and current state. Do not invent future scenarios or unsolicited code snippets without user consent.
- **MasterChef Rule**: Never explain pre-existing codebase features or theoretical mechanics as if they were new tasks to be built. State ONLY what actually requires code changes. Do not talk about what is already done or irrelevant background concepts.
- **Focus on Essential Details**: Provide precise, non-ambiguous details on essential mechanics, counts, file paths, and data structures. Zero fluff, zero theoretical tangents, zero flowery language.
- **Strict Adherence & Safety Rule**: Execute requests EXACTLY as requested by the user. NEVER run destructive commands (such as git checkout, git reset, git restore, rm, or file reverts) without explicit prior user approval. Always ask the user first if an alternative approach is suggested.

## Next Action Items
1. Execute clean run of `experiments/curriculum_v1.sh` (stages 1-3 + tournaments).
2. Archetype-stratified dataset selection (`deck_elo_daily` + `deck_cards`).
3. Recurrent registers and TBPTT implementation.

