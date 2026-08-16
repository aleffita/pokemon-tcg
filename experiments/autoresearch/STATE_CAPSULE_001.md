# State Capsule 001 — AR-001 reviewed

Captured: 2026-08-16 after reviewer completion.

## Current champion

Frozen Stage 4 remains the champion/fallback. No candidate model has been promoted. Root checkpoint: `experiments/autoresearch/root/stage4_root.pkl`, SHA-256 `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`. Root FP32 package: `experiments/autoresearch/root/stage4_root_fp32.tar.gz`, SHA-256 `32add97ad0848cc097a983a45a75a935532a807645b9e36d972bd6fee1c49751`.

## Architecture

Stage 4: `d_model=128`, `nhead=4`, `nlayers=4`, `ff_dim=512`, `scratch_registers=32`, static/split-heads enabled, structured heads disabled, 1,302,151 parameters. The checkpoint payload stores epoch 36 using the trainer's zero-based convention; the historical log prints the resumed final epoch as 37.

## Data plane

Current training reads day-partitioned Parquet with a row-group cache. AR-001 observed approximately 1.401 GB decoded for 2,048 logical rows, 88 SSD spills and 9.14 GiB of spill files during the tiny training probe. This is the leading pipeline bottleneck hypothesis, not yet a validated optimization target.

## Experiment just completed

AR-001 measured ETL, load, training, rollout and tournament components and corrected packaged-agent deck provenance in `scripts/tournament.py` at commit `dccef4f744b6cdb5b0f93de9d37f6c660f9912d6`.

## Observed

- ETL probe: 4.03 s, 426 rows, temporary `2026-08-12` artifact.
- Load probe: 1.113 s, 2,048 rows fetched from existing `2026-08-08` data.
- Train probe: 189.23 s, 9 optimizer steps, 11.0 rows/s, no swap.
- Rollout probe: 0.448 s for one game.
- Tournament smoke: 2 games in 0.751 s matchup time, Stage 4 deck 284 versus `lb600_dragapult_ex` deck 7, 0/2/0.
- The component timings are not an ETL-to-training chain because the source artifacts differ.
- Rare-event reconstruction: current `_aux_loss` sums masked auxiliary terms without a valid-row denominator; the older prospective planner had explicit inverse-frequency event weights and was later removed.
- The user confirmed that the `pytest` dependency changes in `pyproject.toml` and `uv.lock` are intentional and authorized.
- QA after the dependency was available: 76 passed, 5 failed in 22.19 s. Failures are the known SQLite FK debt (2,946,336 violations), two derived referential-integrity assertions, one PageRank formula assertion and one missing `DECKS_GENERATED` symbol. Full output: `experiments/autoresearch/AR-001/pytest.log`.

## Inference

The tournament patch is an infrastructure improvement worth keeping. The benchmark supports investigating compact/model-ready shards, but it does not yet support claiming faster end-to-end research or a model change. The next experiment must first establish data identity before comparing throughput.

## Rejected interpretation

Do not treat 0-2 in two games as a policy ranking. Do not treat component timings as one end-to-end pipeline measurement. Do not claim that Stage 4's packaged FP32 model is conversion-linked to the frozen MLX checkpoint until a manifest proves it.

## Decision

Keep `dccef4f`. Mark AR-001 as `infrastructure_keep` with an inconclusive competitive baseline. Do not modify the Stage 4 root, model architecture or loss. Proceed to a single worker for a data-identical compact-shard benchmark.

## Next best experiment

AR-002: create or benchmark a compact shard representation from exactly one fixed Parquet source and compare row/label/order digests, decoded bytes per row, load/train rows per second, RSS, spills and swaps against the existing cache path.
