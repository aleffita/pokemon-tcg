# State Capsule 004 - AR-003

Captured 2026-08-16 after the AR-003 implementation, parity gate, and
bounded multi-day amortization run.

## Decision

Stage 4 remains the frozen competitive root. The packed backend remains opt-in
and is not promoted. Multi-day data parity and host materialization pressure
improved, but one-time ETL was not amortized at every observed horizon:
candidate source-to-ready was faster at 2 epochs and slower at 1 and 3 epochs.
Promotion is inconclusive under the required gate.

## Frozen root

- Root checkpoint: `experiments/autoresearch/root/stage4_root.pkl`
- SHA-256: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`
- FP32 package SHA-256:
  `32add97ad0848cc097a983a45a75a935532a807645b9e36d972bd6fee1c49751`
- Architecture, loss, policy, inference, SQLite, and deck are unchanged.

## AR-003 implementation

- `rl/packed_data.py`: format version 2, independent 86-column trainer
  contract, explicit row-level order digests, source-list digest, honest seed
  semantics, and required-column/order validation.
- `scripts/bc/build_packed_cache.py`: ordered multi-Parquet ETL, val-first then
  train row layout, one reusable store per immutable source selection.
- `scripts/bc/bc_train_mlx.py`: packed contract validation against all required
  columns and multi-source digest; explicit no-fallback error for packed without
  TBPTT.
- `experiments/autoresearch/AR-002/parity.py`: independent Parquet reader,
  exact `split_episode_ids()` use including `max_rows=0`, row-level parity.
- `experiments/autoresearch/AR-002/benchmark.py`: one backend per process,
  process RSS/peak RSS and process fault metrics; no global swap counter.
- Focused integration test crosses `PackedArrayStore` and the trainer's
  `_build_tbptt_decision_groups` / `_build_tbptt_plan` with action masks and
  auxiliary targets.

## Validation evidence

- Sources: consecutive `2026-08-08` and `2026-08-09` Parquets.
- Selection: `max_rows=10000`, effective 10,035 rows, 880 val, 9,155 train,
  59 episodes, `val_frac=0.1`, seed `13971479023478`.
- Packed logical bytes: `418,820,760`.
- Packed data digest:
  `332fbacd00a30cefc5446c1424f0eacae9e88fe5e2a98a9dcb82705f534474a5`.
- Independent parity: 86/86 columns, row order, shapes, dtypes, values,
  labels/masks, auxiliary targets, and row-order groups equal; mismatches 0.
- Load baseline/candidate process RSS: 9,992,110,080 B versus 916,111,360 B.
- Load decoded bytes: 2,858,248,224 B versus 418,820,760 B.
- Training real seconds baseline/candidate: 133.02/168.50 at 1 epoch,
  364.59/300.23 at 2 epochs, 486.99/468.52 at 3 epochs.
- Candidate total source-to-ready including one ETL: 199.187/330.917/499.207
  seconds at 1/2/3 epochs. Baseline source-to-ready: 134.992/366.562/488.962.
- All runs used 15/30/45 optimizer steps and displayed matching model/loss
  metrics within runtime numeric variation. All cache runs reported zero
  spills.

## Artifacts

`experiments/autoresearch/AR-003/` contains the report, logs, parity result,
load results, six training logs, tests log, and packed manifest. The packed
binary store is outside Git. No tournament was needed because inference was
unchanged.

## Next control point

Keep packed data opt-in. Measure per-process training RSS and isolate packed
TBPTT gather/prefetch overhead with repeated 2-epoch runs before reconsidering
promotion.
