# BRIEFING — 2026-08-14T14:17:00Z

## Mission
Investigate and design technical blueprint for MoE 4-expert topology, load balancing auxiliary loss, Vehicle Draft cross-attention encoder, and Apex Mode runtime airgap in PyTorch and MLX.

## 🔒 My Identity
- Archetype: Explorer (Investigation & Technical Design)
- Roles: Read-only investigator, architecture designer, synthesis reporter
- Working directory: /Users/alefita/workdir/pokemon-tcg/.agents/m1_exp_moe/
- Original parent: 9a189410-43b1-4cdc-bc2a-7a942180e59c
- Milestone: Milestone 1 (MoE 4-Expert Topology, Load Balance Loss, Vehicle Draft & Apex Mode)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement production source code directly. Produce structured technical blueprints and reports.
- All Python executions MUST use `uv run`.
- Mathematical rigor, PyTorch and MLX parity.
- Top-2 Softmax router with gating noise and load balancing.
- Vehicle Draft cross-attention over 60 cards.
- Apex Mode airgap activation (`datetime >= 2026-08-16T00:00:00Z`).

## Current Parent
- Conversation ID: 9a189410-43b1-4cdc-bc2a-7a942180e59c
- Updated: 2026-08-14T14:17:00Z

## Investigation State
- **Explored paths**: `docs/architecture/moe_pipeline_blueprint.md`, `docs/neural_engine_and_tokenization_spec.md`, `docs/architecture/01_ropend_theory.md`, `docs/architecture/02_stochastic_elo_inference.md`, `rl/policy_mlx.py`, `rl/policy_infer_torch.py`, `scripts/bc/bc_train_mlx.py`, `rl/deck/decks.py`.
- **Key findings**: Complete dual PyTorch & MLX mathematical blueprint developed for:
  1. MoE 4-expert topology (Agro, Control, Setup, Endgame) with $D=128 \to 512 \to 128$ SwiGLU / GELU.
  2. Top-2 Softmax router with training gating noise and weight normalization.
  3. Load balancing loss $\mathcal{L}_{\text{balance}} = \alpha_{\text{balance}} E \sum_{e=1}^E f_e P_e$ with gradient detachment for $f_e$.
  4. 60-card Vehicle Cross-Attention Draft encoder generating intra-deck synergy vector.
  5. Apex Mode airgap activation switch in `act()` setting temperature $\tau=0.1$ starting Aug 16, 2026.
  6. Unit test plan across `tests/unit/test_moe_router.py` and `tests/unit/test_vehicle_draft.py`.
- **Unexplored areas**: Production implementation delegated to implementers.

## Key Decisions Made
- Selected SwiGLU as primary FFN activation with GELU fallback.
- Formulated exact load balancing loss maintaining smooth backpropagation into router weights while isolating discrete token counts.
- Specified clean PyTorch and MLX interfaces for seamless checkpoint serialization.

## Artifact Index
- `.agents/m1_exp_moe/DISPATCH.md` — Initial dispatch message.
- `.agents/m1_exp_moe/progress.md` — Liveness heartbeat and progress tracking.
- `.agents/m1_exp_moe/handoff.md` — Comprehensive analysis, math specifications, and code blueprints.
