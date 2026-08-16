# AR-004 report (amended by AR-005)

## Decision

The packed backend P1 contract holes are closed in the opt-in trainer path.
Runtime validation is fail-closed, checkpoint identity is explicit, and the
adversarial focused suite passes. No long training run or tournament was run.

The backend is **not ready for self-play**. This change validates the packed BC
data plane and resume boundary only; it does not establish an on-policy
self-play trajectory contract or tournament evidence. Stage 4 remains the
frozen root and packed data remains opt-in.

## Runtime packed validation

`bc_train_mlx.py` now opens every required trainer column when it constructs the
packed metadata store, including the complete order key:
`episode_id`, `side`, `step_id`, `decision_id`, and `substep`.

Strict construction validates:

- packed format/version, required columns, shape/dtype, file hashes, and value digests;
- source digest against the ordered live Parquet source list;
- required dedup and TBPTT metadata;
- val/train boundary and row counts;
- ordered split episode membership;
- independent val/train row-order digests.

Missing metadata or unopened order columns cannot fall through to a partial
validation path. Packed plus `tbptt_chunk <= 0` raises the existing explicit
no-fallback error.

## Checkpoint identity and legacy policy

New checkpoints record `data_identity` containing:

- ordered source digest;
- selection parameters and selected episode IDs;
- val/train episode membership, row counts, boundary, and row-order contract;
- packed backend name/version/data digest, or the explicit Parquet backend;
- seed, dedup mode, and TBPTT chunk contract.

Resume compares this object exactly. Explicit source, selection, split, packed,
backend, seed, dedup, or TBPTT mismatches fail before training starts. A
checkpoint with a missing or partial identity is rejected.

The preserved Stage 4 root is the only compatibility exception. AR-004's
basename wording was incomplete: AR-005 binds the exception to the exact
project-relative path `experiments/autoresearch/root/stage4_root.pkl` and the
approved SHA-256
`b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
Same-content copies, non-root files, packed resumes, and optimizer/scheduler
continuations are rejected. The allowed warm-start resets both phases and is
tested through the production trainer setup as
`legacy-stage4-warmstart-no-data-identity`.

## Adversarial coverage

The focused tests cover tampered boundary, tampered row order, inverted source
order, missing required order column, missing dedup metadata, missing TBPTT
metadata, partial and source/split/seed/dedup/TBPTT/backend resume identity
mismatches, exact-root legacy warm-start policy, seed variants with
`max_rows=0`, packed `opt_group` relabeling during sequential TBPTT row
consumption, and packed data without TBPTT.

## Verification

The final focused command passed:

```text
uv run --locked pytest -q tests/test_packed_data.py tests/test_packed_tbptt_integration.py scripts/validate/test_checkpoint.py
19 passed in 0.55s
```

The full output is in [`logs/tests.log`](logs/tests.log). A separate attempt to
include `scripts/validate/test_bc_train_progress.py` failed during collection
because `_standard_microbatch_count` is absent from the preserved baseline
trainer. That unrelated symbol was not changed by AR-004.

No files under `pyproject.toml`, `uv.lock`, the Stage 4 root, model/loss/
inference/SQLite/deck paths, or prior autoresearch artifacts were changed by
AR-005. See `AR-005/report.md` and `AR-005/logs/tests.log` for the rework
evidence and the separate pre-existing semantic-suite failures.
