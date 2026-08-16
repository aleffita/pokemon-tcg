# AR-001 reviewer report

Captured 2026-08-16. Reviewer decision: **KEEP** commit `dccef4f744b6cdb5b0f93de9d37f6c660f9912d6` as an infrastructure correction; rework the acceptance interpretation before using the benchmark as a baseline.

## Findings

1. **P1: component timings are not one ETL-to-training chain.** The ETL probe wrote a temporary `2026-08-12` Parquet artifact with 426 rows, while load and train read the existing `2026-08-08` artifact with 2,082 rows. The component measurements remain useful, but end-to-end parity was not demonstrated.
2. **P1: conversion provenance is incomplete.** The frozen checkpoint and FP32 tarball hashes are recorded, and the model in the tarball matches the current FP32 model artifact, but there is no conversion manifest/hash linking the frozen MLX checkpoint directly to the tarball.
3. **P1: tests were not collected.** The original test command failed before collection because `pytest` was unavailable then. This is neither a code failure nor a pass.
4. **P2: deck restoration was not exercised.** The run used `--no-sweep`; inspection shows the package's temporary `deck.csv` path is isolated and restoration logic is present, but the sweep path still needs a targeted test.
5. **P2: tournament observability should record explicit tournament and match IDs.** The corrected JSON and ResultsDB rows contain `our_deck_id=284` and `opp_deck_id=7`; the stdout label is ambiguous.
6. **P3: pre-existing tar extraction uses unvalidated `extractall`; this is not introduced by AR-001.**

## Ground-truth check

The Stage 4 smoke result is valid as a deck-conditioned execution/provenance gate: our deck 284 versus `lb600_dragapult_ex` deck 7, sides alternated, 0 wins, 2 losses, 0 draws. It is not competitive-strength evidence.

## Rare-event review

The report's reconstruction is source-backed. The current Parquet auxiliary loss sums masked terms without division by `valid_sum`; the historical prospective planner used inverse-frequency weights before the sidecar was removed. The report now distinguishes the probe's 0.5 weights from the `TrainConfig` dataclass's 1.0 `aux_return_weight` default.

## Next gate

Run a data-identical compact-shard benchmark on one fixed Parquet source, preserving episode/side/step ordering, labels and Stage 4 semantics. Do not change the model or loss until parity is proven.
