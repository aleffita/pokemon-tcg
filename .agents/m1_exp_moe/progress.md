# Progress — Explorer 2 (MoE & Vehicle Draft)

Last visited: 2026-08-14T14:17:05Z

- [x] Initialized workspace and briefing
- [x] Read domain skill `ptcg-moe-architecture`
- [x] Inspect existing specifications: `moe_pipeline_blueprint.md`, `neural_engine_and_tokenization_spec.md`, `SCOPE.md`, `ORIGINAL_REQUEST.md`, `01_ropend_theory.md`, `02_stochastic_elo_inference.md`
- [x] Inspect existing codebase: `rl/policy*.py`, `scripts/bc/bc_train_mlx.py`, `rl/deck/decks.py`, `scripts/validate/`
- [x] Complete mathematical design of PyTorch & MLX Top-2 Softmax Router with noisy gating
- [x] Complete design of 4 specialized FFN experts (Agro, Control, Setup, Endgame)
- [x] Complete design of Load Balancing Auxiliary Loss ($\mathcal{L}_{\text{balance}}$) with PyTorch & MLX gradient mechanics
- [x] Complete design of Vehicle Cross-Attention Draft encoder (60-card sequence)
- [x] Complete design of Apex Mode runtime airgap activation logic and temperature scaling ($\tau = 0.1$)
- [x] Complete design of Comprehensive Unit Test Plan
- [x] Synthesize findings into `handoff.md` and report back
