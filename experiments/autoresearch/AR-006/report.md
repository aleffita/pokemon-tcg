# AR-006 report

## Decision

AR-005 reviewer rework is implemented and the focused locked gate is green.
The packed backend remains opt-in and is **not approved for self-play yet**:
AR-006 validates the behavioral-cloning loader and resume infrastructure, but
no on-policy trajectory, self-play, tournament, or promotion gate was run.

## Changes

- Added a production `_prepare_training_phases` path and a positive matching-
  identity resume integration test. The test proves optimizer state continues
  at phase step 12 and scheduler state continues at step 13 of 20. Existing
  Stage 4 reset behavior, packed/legacy rejection, and identity mismatch
  rejection remain covered.
- Extracted the production `_load_temporal_batch` implementation so the packed
  test uses the same loader as `main()`. The fixture exercises `opt_group`
  relabeling, ordered decision lengths, cross-batch recurrent memory
  continuation, and the sequential TBPTT loss path without long training.
- Parameterized the frozen Stage 4 artifact gate for canonical project-relative,
  canonical absolute, regular-copy, hardlink, and symlink paths. Relative
  candidates are documented as cwd-relative and are accepted only when that
  path is the canonical project path and its SHA-256 matches.
- Corrected seed-test wording, added distinct seed identity coverage, and
  removed the dangling no-op expression.

## Verification

Focused locked validation passed:

```text
uv run --locked pytest -q tests/test_packed_data.py tests/test_packed_tbptt_integration.py tests/test_trainer_resume_integration.py scripts/validate/test_checkpoint.py
47 passed in 0.66s
```

`git diff --check` passed. No tournament, self-play, long training, binary
generation, Stage 4 root mutation, model/loss/inference/SQLite/deck change,
`pyproject.toml` change, or `uv.lock` change was included in AR-006.
