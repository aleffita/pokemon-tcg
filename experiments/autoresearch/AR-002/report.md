# AR-002 report

Captured 2026-08-16 in the designated `develop` checkout.

## Decision

**KEEP as an opt-in infrastructure backend.** The candidate is data-identical
for the fixed probe, reduces decoded training bytes by 32.9x, reduces clean
load RSS from 7.98 GiB to 273 MiB, removes row-group materialization from the
training path, and produces the same Stage 4 probe metrics. The default
Parquet/cache path is unchanged. No model, loss, SQLite database, Stage 4
root, or inference behavior was changed.

Implementation commit: `4d0a217` (`feat(data): add opt-in mmap BC training backend`).

The throughput difference in this bounded run is small and should not be
treated as a stable compute benchmark: the candidate process took 20.82 s and
the baseline took 21.51 s for the same nine microbatches and nine optimizer
steps. The materialization and memory result is the stronger causal signal.

## Experimental contract

- Source: `data/bc_data/2026-08-08.parquet`
- Source SHA-256: `ed4b392a41dc363d7175bffcdfdca105a21450fc209c36320a561e3631a8f1f0`
- Selection: `max_rows=2048`, episode-boundary cap, 12 episodes, 2,082 rows
- Episode IDs: `90848813, 90848904, 90848967, 90848974, 90848975, 90849002, 90849004, 90849005, 90849006, 90849007, 90849010, 90849011`
- Split: first 1 episode for validation, 11 for training, 127/1,955 rows
- Seed: `13971479023478`
- Stage 4 configuration: `d_model=128`, `nhead=4`, `nlayers=4`, `ff_dim=512`,
  `scratch_registers=32`, `static=true`, `split_heads=true`,
  `structured=false`, `batch=1024`, `accum_steps=1`, `tbptt_chunk=16`,
  auxiliary weights `0.5/0.5/0.5/0.5`, `would_ko=true`
- Root checkpoint: `experiments/autoresearch/root/stage4_root.pkl`,
  SHA-256 `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`
- FP32 package: `experiments/autoresearch/root/stage4_root_fp32.tar.gz`,
  SHA-256 `32add97ad0848cc097a983a45a75a935532a807645b9e36d972bd6fee1c49751`
- Packaged deck hash: `4f1b76821b9dd638a25ed976701cad1bdfdddf4507c804bc2e950eae00099c97`

## Candidate representation

The candidate is a fixed-width store with one C-contiguous `.npy` array per
column and a JSON manifest. Arrays are opened read-only with NumPy mmap. The
manifest records the source hash, selection contract, split episode IDs,
dtype/shape, per-file hashes, per-column value digests, and a combined data
digest.

The store contains 86 columns: all 67 model input columns, `y`,
`is_attack`, `opt_group`, all five auxiliary targets, and the episode/side/order
and scalar state columns required to preserve trainer semantics. The complete
selected payload is 86,894,352 bytes. The current trainer consumes 74 columns
for this non-deduplicated Stage 4 probe, 85,199,604 bytes after selection.

The layout was selected from measured properties, not convention. The Parquet
row group containing the selection has 34,242 physical rows. The current cache
decodes the full row group for 2,082 logical rows. The packed store writes only
the selected rows and gives the TBPTT loader direct row-index gathers from
memory-mapped arrays.

## Exact parity result

Parity passed with zero mismatches:

- row count: `2,082` selected, `1,955` train, `127` validation
- columns compared: `86`
- per-column dtype and shape: equal
- per-column value arrays: equal
- episode/side/step/decision/substep digest: equal
- model input digest: equal
- labels and masks digest: equal
- auxiliary target digest: equal
- candidate combined data digest:
  `6d15639abce2fbe240e92c6a86c2bff32330684105a1b7f07f27e379702949c3`

The exact parity artifact is `parity.json`. The candidate and baseline
checkpoint model arrays were not bit-identical after separate MLX executions,
with maximum absolute parameter difference `8.06e-06`; this is a runtime
numeric nondeterminism observation, not a data mismatch. Both runs produced
the same validation accuracy, loss, auxiliary metrics, parameter count, and
optimizer-step count.

## Benchmark

The load measurements below were run in separate clean processes. Baseline
ETL is zero because the existing Parquet is already the current cache source.
Candidate ETL is the one-time build from that same fixed Parquet source.

| Metric | Baseline Parquet/cache | Candidate mmap store |
| --- | ---: | ---: |
| ETL/build time | 0.00 s, pre-existing | 1.60 s |
| load time, source already ready | 1.866 s | 0.162 s |
| source-to-ready including ETL | 1.866 s | 1.762 s |
| decoded/returned bytes | 2,802,502,248 | 85,199,604 |
| decoded bytes per selected row | 1,346,062.6 | 40,922.0 |
| load rows/sec | 1,155 | 170,789 |
| clean-process RSS peak | 7.98 GiB | 273 MiB |
| process swap count | 0 | 0 |
| train process wall | 21.51 s | 20.82 s |
| train rows/sec, 1,955 rows | 90.9 | 93.9 |
| microbatches/sec, 9 batches | 0.418 | 0.432 |
| optimizer steps/sec, 9 steps | 0.418 | 0.432 |
| MLX peak memory | 6.24 GiB | 6.18 GiB |
| train cache decoded | 1,336.3 MiB | 76.3 MiB |
| validation cache decoded | 1,336.3 MiB | 5.0 MiB |
| SSD spills in this run | 0 | 0 |

The one-epoch process wall including candidate ETL is approximately 22.42 s
versus 21.51 s for an already-built baseline Parquet source. That upfront cost
is not hidden. It becomes amortized when the packed store is reused for later
epochs or phases. The prior AR-001 baseline also recorded 88 SSD spills and
9.14 GiB of spill files under the same TBPTT cache design; this fresh bounded
run did not cross the pressure threshold, so AR-002 does not claim that every
baseline run spills.

Both integrated training runs reported:

```text
val_acc=0.6693
loss=0.9671
aux_ko_bce=0.2927
aux_prize_mse=0.2797
aux_terminal_bce=0.1129
aux_return_mse=2.6297
params=1,302,151
microbatches=9
optimizer_steps=9
```

## Observations

1. The source selection and candidate arrays are exactly equal under the
   trainer's normalized dtype and shape contract.
2. The baseline cache's dominant physical cost is row-group decoding, not the
   logical selected rows. The candidate removes this amplification for the
   fixed selection.
3. The candidate integration is behind `--packed-data`; without that flag the
   existing Parquet/cache path remains the default.
4. The candidate uses the same TBPTT plan, model, loss, optimizer, seed,
   checkpoint, and auxiliary targets. No inference path was changed, so no
   tournament was run.
5. Focused tests pass: `2 passed in 0.05s`.

## Inferences

- For a repeated training loop over this fixed corpus, the compact
  representation is a justified integration target because the data-path
  reduction is large, measurable, and data-identical.
- The bounded one-epoch compute result is consistent with a small positive
  throughput change, but it is not enough to claim a stable model-compute
  speedup. The acceptance case rests on lower materialization/RSS and the
  absence of spill pressure, with throughput non-regression.
- The store should be built once per immutable source and selection, then
  reused across epochs. Rebuilding it before every epoch would erase the
  benefit.

## Limitations and falsification boundaries

- Only the fixed 12-episode probe was packed. Multi-day and full-corpus
  packing, disk-space planning, and concurrent workers were not benchmarked.
- The candidate currently requires one Parquet source and the Stage 4 TBPTT
  path. It intentionally fails early for other combinations rather than
  silently falling back.
- The candidate shard itself is not committed. Re-run the ETL command and
  verify `packed_manifest.json` and its hashes before use.
- The one-time ETL plus one epoch is slightly slower than an already-built
  Parquet source. Reuse across at least several epochs is part of the intended
  operating contract.
- Full repository QA did not collect: the pre-existing
  `scripts/validate/test_bc_train_progress.py` imports the absent
  `_standard_microbatch_count` symbol from the trainer. The focused AR-002
  tests pass. This collection error is outside the AR-002 data backend.

## Reproduction commands

Use a temporary directory represented below by `PACKED_STORE`.

```bash
uv run python scripts/bc/build_packed_cache.py \
  --source data/bc_data/2026-08-08.parquet \
  --out PACKED_STORE \
  --max-rows 2048 \
  --val-frac 0.1 \
  --seed 13971479023478

uv run python experiments/autoresearch/AR-002/parity.py \
  --source data/bc_data/2026-08-08.parquet \
  --packed PACKED_STORE \
  --max-rows 2048 \
  --val-frac 0.1 \
  --seed 13971479023478 \
  --out experiments/autoresearch/AR-002/parity.json

uv run python experiments/autoresearch/AR-002/benchmark.py \
  --source data/bc_data/2026-08-08.parquet \
  --packed PACKED_STORE \
  --max-rows 2048 \
  --val-frac 0.1 \
  --seed 13971479023478 \
  --out experiments/autoresearch/AR-002/load_benchmark.json

uv run tcg-train \
  --config configs/train_config.json \
  --days 2026-08-08 \
  --epochs 1 \
  --batch 1024 \
  --tbptt-chunk 16 \
  --max-rows 2048 \
  --val-frac 0.1 \
  --seed 13971479023478 \
  --resume experiments/autoresearch/root/stage4_root.pkl \
  --optimizer-state reset \
  --scheduler-state reset \
  --packed-data PACKED_STORE \
  --out RUN_DIR/probe.pkl \
  --phase-id AR-002-packed

uv run pytest -q tests/test_packed_data.py
```

## End-to-end validity and next hypothesis

The fixed-probe chain is now valid:

```text
fixed Parquet
  -> exact episode selection and split
  -> packed ETL with manifest/digests
  -> parity against the current cache
  -> opt-in trainer backend
  -> Stage 4 TBPTT micro-run and checkpoint
```

The next hypothesis is to benchmark a reusable packed store over a larger
multi-day strategic subset, with a bounded two-to-three-epoch run, and measure
whether ETL amortization remains favorable without increasing unified-memory
pressure. Do not change the model or loss until that data-plane result is
known.
