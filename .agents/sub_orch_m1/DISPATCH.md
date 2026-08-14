## 2026-08-14T14:14:44Z

You are Sub-Orchestrator for Milestone 1 (4D RoPEND & MoE Neural Architecture).
Your working directory is: `/Users/alefita/workdir/pokemon-tcg/.agents/sub_orch_m1/`
Project workspace root: `/Users/alefita/workdir/pokemon-tcg`
Original user request: `/Users/alefita/workdir/pokemon-tcg/.agents/ORIGINAL_REQUEST.md`
Master project scope: `/Users/alefita/workdir/pokemon-tcg/PROJECT.md`
Explorer 1 survey: `/Users/alefita/workdir/pokemon-tcg/.agents/survey_explorer_1/analysis.md`

Your mission:
Orchestrate Milestone 1:
1. Implement 4D RoPEND operator ($c_1$: Step, $c_2$: Meta-Epoch, $c_3$: Urgency Clock, $c_4$: Inferred Elo) with 4x8-dim Givens rotation planes per 32-dim attention head in both PyTorch (`rl/ropend/ropend_torch.py`) and MLX (`rl/ropend/ropend_mlx.py`).
2. Implement MoE 4-expert topology with Top-2 routing (`rl/moe/router.py`, `rl/moe/experts.py`), load balancing auxiliary loss $\mathcal{L}_{\text{balance}}$ (`rl/moe/load_balance.py`), vehicle cross-attention draft (`rl/deck/vehicle_draft.py`), and runtime Apex Mode airgap activation token at `2026-08-16T00:00:00Z` ($\tau = 0.1$) in `rl/policy_moe_torch.py` and `rl/policy_moe_mlx.py`.
3. Ensure strict FP32 precision contract across `rl/policy_infer_torch.py` and MLX training pipeline (`scripts/bc/bc_train_mlx.py`), with static card feature SHA256 checksum and shape validation.
4. Follow the Project Pattern iteration loop (Explorer -> Worker -> Reviewer -> Challenger -> Auditor -> Gate) or delegate.
5. All Python executions MUST use `uv run`. Enforce the non-negotiable binary audit veto.
6. When your milestone gate passes, write your `handoff.md` and send a completion message to parent (`cd851a4f-6875-4819-9f25-1b23dd14cc1b`).
