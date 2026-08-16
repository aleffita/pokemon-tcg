# AR-003 report

Captured 2026-08-16 in the designated `develop` checkout.

## Decision

Keep the packed backend opt-in. Do not promote it to the default and do not
claim a stable end-to-end speedup yet. Multi-day parity and materialization
pressure improved materially, but the one-time ETL was not amortized by every
tested horizon: candidate source-to-ready was faster at 2 epochs and slower at
1 and 3 epochs. This is inconclusive for promotion under the stated gate.

No model, loss, policy, inference, SQLite, or deck code changed. No tournament
was run because inference behavior did not change.

## Source and bounded selection

- Consecutive sources: `data/bc_data/2026-08-08.parquet` and
  `data/bc_data/2026-08-09.parquet`.
- Source SHA-256 values: `ed4b392a41dc363d7175bffcdfdca105a21450fc209c36320a561e3631a8f1f0` and
  `c6a31adfa9dbf196b9461a04b97ce4e04a150d704c4ccc6fcc9de3e8c705cf48`.
- Combined ordered source digest:
  `dcfa720558140946957dd28f2d6d0b6691c196bb0c7c023c2861beb03091858a`.
- Selection: `max_rows=10000`, `val_frac=0.1`, trainer seed
  `13971479023478`, `tbptt_chunk=16`, batch `1024`.
- Effective selection: 59 episodes and 10,035 rows, with 880 validation rows
  and 9,155 training rows. The episode-boundary cap explains why the result is
  above 10,000 and above the AR-002 2,082-row probe.
- The 10k-row limit is deliberate for the Apple M3 Pro 24 GB unified-memory
  machine: it exercises two days and row-group pressure while keeping the
  packed payload at 418,820,760 bytes rather than attempting the roughly
  1.6M-row source corpus.
- Both day manifests report schema 3 and would-KO enabled, computed, and
  `bc_wk_nvar=10`.
- One packed store was built once and reused by all candidate load/train
  processes. ETL wall time observed by the build process was approximately
  30.2 s and is charged once in every candidate source-to-ready comparison.

## Contract changes

The packed format is version 2.

- Builder accepts an ordered list of Parquet sources and explicitly partitions
  rows into validation first and training second.
- The manifest records the exact 5-field row key
  `episode_id, side, step_id, decision_id, substep`, the val/train boundary,
  and independent per-split row-order digests.
- `PackedArrayStore` validates the row-order contract, boundary counts, and
  split episode membership when the complete order columns are open.
- The independent parity reader reads Parquet directly and compares every
  packed array, dtype, shape, value, and row-level key sequence. It does not
  use the production row-group cache and does not accept ID sets alone.
- The independent trainer contract requires 86 columns: 67 encoder inputs,
  action/legal masks, `y`, `is_attack`, `opt_group`, all five `aux_*` targets,
  and the complete TBPTT/order metadata. A partial manifest is rejected even
  when the current run does not enable deduplication or an auxiliary head.
- `split_episode_ids()` is now the exact implementation used by parity for
  `max_rows=0` and every other cap. Seed is explicitly not used for episode
  membership or val/train split; trainer seed controls shuffle and is recorded
  as provenance only.
- Multiple Parquet sources are supported in the packed store. Unsupported
  packed plus non-TBPTT execution is rejected with an explicit error and no
  Parquet fallback.

## Parity result

`parity.json` passed:

- rows: 10,035 selected, 880 val, 9,155 train;
- required columns: 86/86;
- row-order digest:
  `7eecbe3cc5d1927cc5d6d272fe76cc39b95fe600261d66481893ee672ace0a94`;
- packed data digest:
  `332fbacd00a30cefc5446c1424f0eacae9e88fe5e2a98a9dcb82705f534474a5`;
- mismatches: 0;
- row order, model inputs, labels/masks, and auxiliary digests: all equal.

## Separate-process load measurements

RSS is process RSS from `psutil.Process().memory_info().rss`, with peak RSS
from the same process. System available memory and percent are reported as
system-scoped pressure signals. No global `psutil.swap_memory()` counter is
used as a per-process metric. Spills are backend counters.

| Metric | Baseline Parquet/cache | Candidate packed |
| --- | ---: | ---: |
| One-time ETL | 0.000 s, source already ready | 30.200 s, one build |
| Load | 1.972093 s | 0.486741 s |
| Source-to-ready incl. ETL | 1.972268 s | 30.686741 s |
| Rows/s during load | 5,411.7 | 171,879.2 |
| Decoded/returned bytes | 2,858,248,224 | 418,820,760 |
| Bytes/selected row | 284,827.9 | 41,736.0 |
| Process peak RSS | 9,992,110,080 B | 916,111,360 B |
| Process major faults | 172 | 0 |
| System available memory at sample | 8,198,144,000 B | 14,248,148,992 B |
| Spills | 0 | 0 |

The candidate load materially lowers row-group amplification and load-process
RSS. The trainer's own `mx.get_peak_memory()` remains approximately 18.91 to
18.96 GiB in both backends, so the measured benefit is host materialization and
pressure reduction, not a claim of lower MLX activation memory.

## Separate-process training measurements

Each row is a fresh process from the same frozen Stage 4 root, same config,
same source selection, same seed, same batch, and same TBPTT plan. `rows/s` is
`epochs * 9,155 / real_seconds`; optimizer steps are 15, 30, and 45 for 1, 2,
and 3 epochs respectively. Final metric fields are copied from the logs, not
projected.

The six training wrappers recorded exact process wall/user/sys time with
`/usr/bin/time -p`; the trainer logs record `mx.get_peak_memory()`. Process RSS
was captured directly in the separate load processes above, where RSS is the
relevant backend materialization comparison. The MLX active-memory figure is
not relabeled as RSS.

| Horizon | Backend | Real s | Source-to-ready s | Rows/s | Optimizer steps/s | Peak MLX GiB | Final/best val acc | Loss | Cache decoded train + val | Spills |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline | 133.02 | 134.992 | 68.83 | 0.1128 | 18.91 | 0.6080 | 1.1853 | 1,336.3 + 1,336.3 MiB | 0 |
| 1 | candidate | 168.50 | 199.187 | 54.48 | 0.0890 | 18.96 | 0.6080 | 1.1853 | 357.3 + 34.3 MiB | 0 |
| 2 | baseline | 364.59 | 366.562 | 50.22 | 0.0823 | 18.91 | 0.6136 | 1.1945 | 1,336.3 + 1,336.3 MiB | 0 |
| 2 | candidate | 300.23 | 330.917 | 61.00 | 0.0999 | 18.91 | 0.6136 | 1.1945 | 714.6 + 68.7 MiB | 0 |
| 3 | baseline | 486.99 | 488.962 | 56.40 | 0.0924 | 18.96 | 0.6125 best | 1.1948 at epoch 39 | 1,336.3 + 1,336.3 MiB | 0 |
| 3 | candidate | 468.52 | 499.207 | 58.62 | 0.0961 | 18.91 | 0.6114 best | 1.1948 at epoch 39 | 1,071.8 + 103.0 MiB | 0 |

The 3-epoch candidate train process is 18.47 s faster before ETL, but 10.245
s slower after charging the one-time ETL. The 2-epoch candidate is 64.36 s
faster before ETL and 35.645 s faster after ETL. The 1-epoch candidate is
35.48 s slower before ETL and 64.68 s slower after ETL. These observations do
not support promotion at the tested horizon.

The final auxiliary metrics were also observed directly. Candidate 1 matched
baseline 1 through displayed precision. Candidate 2 matched baseline 2 except
for a displayed `aux_prize_mse` difference of 0.0001. Candidate 3 matched
baseline 3's displayed epoch-39/epoch-40 metrics except its best validation
accuracy was 0.6114 versus 0.6125. This is runtime numeric variation, not a
data parity failure.

## Artifacts and reproduction

- `packed_manifest.json`: manifest and hashes for the one reusable store.
- `parity.json` and `parity.log`: independent multi-day parity.
- `load_baseline.json`, `load_candidate.json` and matching logs: separate
  process load and RSS/pressure measurements.
- `train_baseline_1.log` through `train_baseline_3.log` and
  `train_candidate_1.log` through `train_candidate_3.log`: bounded training
  runs.
- `tests.log`: focused test result, 5 passed.
- The packed `.npy` shard itself remains outside the repository and is not
  committed. Its manifest records the data digest and source hashes.

Commands used:

```bash
uv run python scripts/bc/build_packed_cache.py \
  --source data/bc_data/2026-08-08.parquet \
  --source data/bc_data/2026-08-09.parquet \
  --out PACKED_STORE --max-rows 10000 --val-frac 0.1 \
  --seed 13971479023478

uv run python experiments/autoresearch/AR-002/parity.py \
  --source data/bc_data/2026-08-08.parquet \
  --source data/bc_data/2026-08-09.parquet \
  --packed PACKED_STORE --max-rows 10000 --val-frac 0.1 \
  --seed 13971479023478 --out experiments/autoresearch/AR-003/parity.json

uv run pytest -q tests/test_packed_data.py tests/test_packed_tbptt_integration.py
```

The train commands add `--epochs 1`, `--epochs 2`, or `--epochs 3`; baseline
omits `--packed-data`, and candidate adds `--packed-data PACKED_STORE`. Both
use the same `--days`, `--max-rows`, `--val-frac`, `--seed`, `--batch`,
`--tbptt-chunk`, root checkpoint, optimizer reset, scheduler reset, and config.

## Next hypothesis

The next useful test is not a promotion. Profile the packed TBPTT gather and
prefetch path against the Parquet cache at a fixed CPU/Metal scheduling state,
with per-process training RSS instrumentation and repeated runs at the 2-epoch
horizon. The current evidence says the store removes materialization pressure
but its row-gather/prefetch path can cost startup or compute wall time. Keep it
opt-in until that overhead is isolated or removed.
