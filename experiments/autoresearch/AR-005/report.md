# AR-005 report

## Decision

AR-004 reviewer rework is implemented and focused validation is green. The
packed backend remains opt-in and is **not ready for self-play**: these tests
establish BC data/resume safety, not an on-policy self-play trajectory
contract, tournament result, or promotion decision.

## Changes

- The legacy Stage 4 warm-start is bound to the exact project-relative path
  `experiments/autoresearch/root/stage4_root.pkl` and SHA-256
  `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
  Arbitrary basename copies, non-root legacy files, packed legacy resumes, and
  optimizer/scheduler resume modes are rejected.
- `bc_train_mlx.main()` now uses shared production resume, optimizer-phase, and
  scheduler-phase setup helpers. The integration fixture loads the approved
  root without entering training, proves optimizer state is initialized from
  zero, and proves scheduler phase starts at zero.
- Resume identity coverage now rejects partial identities and source, split,
  seed, dedup, TBPTT, and backend mismatches. The seed variant uses distinct
  seeds while preserving the documented seed-independent episode split.
- The inverted-source-order test builds a manifest for `[A, B]` and validates
  `[B, A]`. The packed TBPTT fixture includes `opt_group`, duplicate canonical
  labels, actual dedup relabeling, and sequential row consumption. Packed
  without TBPTT remains fail-closed.

## Verification

Focused locked validation passed:

```text
uv run --locked pytest -q tests/test_packed_data.py tests/test_packed_tbptt_integration.py tests/test_trainer_resume_integration.py scripts/validate/test_checkpoint.py
41 passed in 0.62s
```

`git diff --check` passed. The separate existing semantic suite was also run:
4 tests passed and 2 failed during its own baseline assumptions. One fixture
omits `structured_weight_decay`; another requires an absent real smoke corpus.
Those failures are outside AR-005 and no protected model, loss, inference,
SQLite, deck, `pyproject.toml`, or `uv.lock` file was changed.

No long training, tournament, self-play run, binary generation, or Stage 4
artifact mutation was performed.
