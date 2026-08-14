## 2026-08-14T14:18:41Z

You are the Worker for Milestone 1 (4D RoPEND & MoE Neural Architecture).
Your working directory is: `/Users/alefita/workdir/pokemon-tcg/.agents/m1_worker_1/`
Project workspace root: `/Users/alefita/workdir/pokemon-tcg`
Original user request: `/Users/alefita/workdir/pokemon-tcg/.agents/ORIGINAL_REQUEST.md`
Master project scope: `/Users/alefita/workdir/pokemon-tcg/PROJECT.md`
Milestone 1 scope: `/Users/alefita/workdir/pokemon-tcg/.agents/sub_orch_m1/SCOPE.md`
Explorer 1 (RoPEND Blueprint): `/Users/alefita/workdir/pokemon-tcg/.agents/m1_exp_ropend/handoff.md`
Explorer 2 (MoE & Apex Blueprint): `/Users/alefita/workdir/pokemon-tcg/.agents/m1_exp_moe/handoff.md`
Explorer 3 (Policy & FP32 Blueprint): `/Users/alefita/workdir/pokemon-tcg/.agents/m1_exp_contract/handoff.md`
Domain skill: `/Users/alefita/workdir/pokemon-tcg/.agents/skills/ptcg-moe-architecture/SKILL.md`

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Write Ownership (You exclusively own and will create/modify these files):
1. `rl/ropend/__init__.py`
2. `rl/ropend/ropend_torch.py`
3. `rl/ropend/ropend_mlx.py`
4. `rl/moe/__init__.py`
5. `rl/moe/load_balance.py`
6. `rl/moe/experts.py`
7. `rl/moe/router.py`
8. `rl/deck/vehicle_draft.py`
9. `rl/policy_moe_torch.py`
10. `rl/policy_moe_mlx.py`
11. `tests/unit/test_ropend_math.py`
12. `tests/unit/test_moe_router.py`
13. `tests/unit/test_vehicle_draft.py`
14. `tests/unit/test_fp32_contract.py`

Your mission:
Implement the complete Milestone 1 neural architecture based on the blueprints in Explorer handoffs:
1. 4D RoPEND Operator in PyTorch and MLX
2. MoE 4-Expert Topology with Top-2 Routing & Load Balancing Loss
3. Vehicle Cross-Attention Draft
4. Apex Mode Airgap Trigger
5. Full Policy Integration (`rl/policy_moe_torch.py` and `rl/policy_moe_mlx.py`)
6. Comprehensive Unit Tests
