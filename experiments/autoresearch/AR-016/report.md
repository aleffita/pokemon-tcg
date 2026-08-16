# AR-016 report

## Scope

Increase the bounded `games_per_mode` limit in the trajectory probe from 1-2
to 1-4. This is a collection-throughput change only. The probe still runs the
same `random` and `mirror_no_memory` modes, preserves the existing per-episode
seed structure, and does not change model architecture, opponent semantics,
recurrence handling, or Stage 4 root selection.

## Changes

- Added `MAX_GAMES_PER_MODE = 4` as the single runtime limit.
- Updated the Python validation error and CLI help/choices to accept 1, 2, 3,
  or 4 and reject values above 4 before probe work begins.
- Added focused logical coverage for existing value 2, new value 4, and
  rejection of value 5.
- Documented the bounded collection contract.

## Validation

- `uv run --locked pytest -q tests/test_trajectory_probe.py tests/test_ar010_candidate_path.py`
  - 28 passed in 2.03s
- `uv run --locked python -m py_compile scripts/rl/trajectory_probe.py tests/test_trajectory_probe.py tests/test_ar010_candidate_path.py`
  - passed
- `git diff --check`
  - passed

No tournament, package, submission, or binary regeneration was performed.
AR-015 artifacts were not modified.
