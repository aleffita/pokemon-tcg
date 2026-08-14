## 2026-08-14T14:15:27Z

<USER_REQUEST>
You are Explorer 3 for Milestone 1 (Policy Integration, FP32 Precision Contract & MLX Training Pipeline).
Your working directory is: `/Users/alefita/workdir/pokemon-tcg/.agents/m1_exp_contract/`
Project workspace root: `/Users/alefita/workdir/pokemon-tcg`
Original user request: `/Users/alefita/workdir/pokemon-tcg/.agents/ORIGINAL_REQUEST.md`
Master project scope: `/Users/alefita/workdir/pokemon-tcg/PROJECT.md`
Milestone 1 scope: `/Users/alefita/workdir/pokemon-tcg/.agents/sub_orch_m1/SCOPE.md`
Existing policies: `rl/policy_infer_torch.py`, `rl/policy_torch.py`, `rl/policy_mlx.py`, `scripts/bc/bc_train_mlx.py`
Domain skill: `/Users/alefita/workdir/pokemon-tcg/.agents/skills/ptcg-moe-architecture/SKILL.md`

Your mission:
Investigate and design the policy integration and strict FP32 precision contract:
1. Review existing PyTorch policy (`rl/policy_torch.py`, `rl/policy_infer_torch.py`) and MLX policy (`rl/policy_mlx.py`, `scripts/bc/bc_train_mlx.py`).
2. Design unified integration: `rl/policy_moe_torch.py` (PyTorch inference & training) and `rl/policy_moe_mlx.py` (MLX training on Apple Silicon), seamlessly integrating 4D RoPEND, MoE Router, and Vehicle Draft.
3. Strict FP32 Precision Contract:
   - Audit `rl/policy_infer_torch.py` to guarantee zero FP16 underflow, strict `torch.float32` tensor contracts, static card feature SHA256 checksum verification, and exact tensor shape assertions.
   - MLX training script (`scripts/bc/bc_train_mlx.py`): verify Muon + AdamW split optimizer in FP32 state, gradient accumulation, and loss computation with auxiliary MoE load balance loss.
4. Backward compatibility & migration strategy: how to cleanly load Stage 4 weights into the base backbone while initializing MoE experts and RoPEND.
5. Unit test plan (`tests/unit/test_fp32_contract.py`).
6. All Python executions MUST use `uv run`.

Write your structured analysis and recommendations to `/Users/alefita/workdir/pokemon-tcg/.agents/m1_exp_contract/handoff.md` and report back.
</USER_REQUEST>
