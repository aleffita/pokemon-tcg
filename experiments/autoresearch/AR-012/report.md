# AR-012 expected candidate SHA preflight

## Scope

AR-012 closes the AR-011 P1 candidate-byte integrity finding without changing
the frozen Stage4 root, AR-010 candidate artifacts, model semantics, inference
semantics, SQLite/deck behavior, or tournament state. The opt-in path now
requires `PTCG_EXPECTED_MODEL_SHA256` whenever `PTCG_MODEL_PATH` is set. It
hashes candidate bytes before candidate provenance validation and before the
strict inference loader; missing, malformed, or mismatched values fail closed.
When `PTCG_MODEL_PATH` is unset, the default public path remains unchanged.

The exact binary package and frozen-root hashes are recorded in
[`AR-012-candidate-receipt.txt`](../AR-010/AR-012-candidate-receipt.txt).
The receipt also records the absolute-path, repository-local, no-sweep,
no-packaging, and no-submission constraints. Candidate, manifest, bundle, and
frozen-root binaries were not added to the commit.

## Focused coverage

- Candidate tensor tampering while retaining embedded provenance is rejected by
  the expected candidate SHA before provenance/model loading.
- Missing and mismatched `PTCG_EXPECTED_MODEL_SHA256` are rejected on the
  opt-in path.
- The real public agent's first `choose()` remains covered and returns `[0]`
  with the expected candidate SHA.
- The default public path remains outside the candidate gate, even when the
  expected-SHA variable is present.

## Validation

```text
$ uv run --locked pytest -q tests/test_ar010_candidate_path.py tests/test_trajectory_probe.py
24 passed in 1.86s

$ uv run --locked python -m py_compile scripts/rl/ppo_micro_update.py public_agents/submissions/latest-submission-300elo/main.py tests/test_ar010_candidate_path.py
exit=0

$ git diff --check
exit=0
```

The full command output is retained in
[`logs/tests.log`](logs/tests.log). No tournament or submission was run.

## Tournament gate

The code and receipt now enforce the candidate-byte preflight contract for a
future opt-in run. This worker did not approve or execute a tournament command:
execution remains a separate operator-authorized action and must satisfy the
receipt's absolute-path, repository-local, no-sweep, no-packaging, and
no-submission constraints.
