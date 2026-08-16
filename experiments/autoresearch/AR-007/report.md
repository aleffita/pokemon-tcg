# AR-007 report

## Decision

The three AR-006 reviewer gaps are covered by a bounded micro probe. Packed
data remains opt-in and is **not approved for self-play yet**. No on-policy
trajectory, self-play, tournament, or promotion gate was run.

## Changes

- Refactored the production optimizer-step body into
  `_apply_optimizer_step`, which is called by `bc_train_mlx.main`. The new
  integration test uses the real Muon/AdamW optimizer, performs two updates,
  proves parameter and optimizer-state mutation, proves scheduler phase
  advancement and LR movement, and covers both fresh and explicit reset phase
  setup.
- Promoted the packed TBPTT one-step-lookahead prefetch iterator and packed
  validation entry into production helpers used by `main`. The tiny fixture
  traverses both paths, consumes the packed store, and runs the sequential
  validation loss entry without long training.
- Added a committed rejection test for the project-relative Stage 4 root
  when the process cwd is outside the project.

The frozen Stage 4 root was not modified. Its approved SHA-256 remains
`b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Verification

```text
uv run --locked pytest -q tests/test_packed_data.py tests/test_packed_tbptt_integration.py tests/test_trainer_resume_integration.py scripts/validate/test_checkpoint.py
51 passed in 0.64s

git diff --check
passed
```

AR-007 does not modify `pyproject.toml`, `uv.lock`, model/loss/inference
semantics, SQLite, deck code, or binary artifacts. No long training run or
submission command was executed.

## Limitations

This is a loader and optimizer/scheduler micro probe, not proof that packed
data is safe for on-policy use. The packed backend remains opt-in pending
real trajectory/log-probability boundary evidence and the separate self-play
review gate. The focused tests do not replace a full training run, self-play,
tournament, or promotion evaluation.
