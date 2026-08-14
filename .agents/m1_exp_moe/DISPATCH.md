## 2026-08-14T14:15:27Z
You are Explorer 2 for Milestone 1 (MoE 4-Expert Topology, Load Balance Loss, Vehicle Draft & Apex Mode).
Your working directory is: `/Users/alefita/workdir/pokemon-tcg/.agents/m1_exp_moe/`
Project workspace root: `/Users/alefita/workdir/pokemon-tcg`
Original user request: `/Users/alefita/workdir/pokemon-tcg/.agents/ORIGINAL_REQUEST.md`
Master project scope: `/Users/alefita/workdir/pokemon-tcg/PROJECT.md`
Milestone 1 scope: `/Users/alefita/workdir/pokemon-tcg/.agents/sub_orch_m1/SCOPE.md`
Architecture docs: `/Users/alefita/workdir/pokemon-tcg/docs/architecture/moe_pipeline_blueprint.md`, `/Users/alefita/workdir/pokemon-tcg/docs/neural_engine_and_tokenization_spec.md`
Domain skill: `/Users/alefita/workdir/pokemon-tcg/.agents/skills/ptcg-moe-architecture/SKILL.md`

Your mission:
Investigate and design the technical blueprint for the MoE subsystem, Vehicle Draft, and Apex Mode token in PyTorch and MLX:
1. MoE 4-expert topology (`rl/moe/router.py`, `rl/moe/experts.py`):
   - 4 specialized FFN experts (Agro, Control, Setup, Endgame) with hidden dimension expansion (e.g. 128 -> 512 -> 128 with SwiGLU / GELU activations).
   - Top-2 softmax router with gating noise during training and normalized routing weights.
2. Load balancing auxiliary loss (`rl/moe/load_balance.py`):
   - Formulate $\mathcal{L}_{\text{balance}} = \alpha_{\text{balance}} E \sum_{e=1}^E f_e P_e$, where $f_e$ is fraction of tokens dispatched to expert $e$ and $P_e$ is mean routing probability.
3. Vehicle Cross-Attention Draft (`rl/deck/vehicle_draft.py`):
   - Autoregressive / bidirectional cross-attention encoder over the 60-card vehicle deck before step 0, generating vehicle synergy context vector.
4. Apex Mode runtime airgap activation:
   - In `act()`, check `datetime.now(timezone.utc) >= 2026-08-16T00:00:00Z`. If true, set routing temperature $\tau = 0.1$ for sharp exploitation on the locked meta.
5. Unit test plan (`tests/unit/test_moe_router.py`, `tests/unit/test_vehicle_draft.py`).
6. All Python executions MUST use `uv run`.

Write your structured analysis and recommendations to `/Users/alefita/workdir/pokemon-tcg/.agents/m1_exp_moe/handoff.md` and report back.
