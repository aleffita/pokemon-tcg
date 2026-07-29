# Pokemon TCG AI Battle

Current line: recurrent MLX behavioral-cloning training with would-KO,
Prospective V2 supervision, and PyTorch FP16 submission inference.

Read first:

- `CLAUDE.md` - operational contract for agents.
- `docs/README.md` - internal documentation map.
- `TASK.md` - exhaustive target contract for the local platform overhaul.

## Current Shape

- Training: MLX, `scripts/bc/bc_train_mlx.py`.
- Inference/submission: PyTorch FP16 checkpoint packaged by `tcg-build`.
- Data: replay ZIPs -> BC NPY directory -> optional `prospective_v2/` sidecar.
- Local platform: partial SQLite/dashboard/tournament implementation.

## Common Commands

Light checks:

```bash
uv run tcg-train --help
uv run tcg-build --help
uv run tcg-tournament --help
git diff --check
```

Build a submission from an explicit checkpoint:

```bash
uv run tcg-build --checkpoint model/checkpoint/bc_temporal_v2_mlx.pkl
```

Run a local tournament only when explicitly authorized:

```bash
uv run tcg-tournament --games 20 --no-sweep --note "describe the run"
```

Full dataset builds and full training runs are intentionally not default agent
actions. Use smoke-sized checks unless Alefita explicitly asks for a real run.

## Documentation

Canonical project memory lives in Wikifita:

- `/Users/alefita/Claude/wikifita/kaggle/pokemon_tcg_ai_battle.md`
- `/Users/alefita/Claude/wikifita/kaggle/pokemon_tcg_tbptt_training_contract.md`
- `/Users/alefita/Claude/wikifita/kaggle/pokemon_tcg_prospective_v2.md`
- `/Users/alefita/Claude/wikifita/kaggle/pokemon_tcg_data_pipeline.md`
- `/Users/alefita/Claude/wikifita/kaggle/pokemon_tcg_local_platform_status.md`
