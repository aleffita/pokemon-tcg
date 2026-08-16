# State Capsule 002 - AR-002 reviewed locally

Captured: 2026-08-16 after the AR-002 worker run.

## Decision

Keep the opt-in fixed-width mmap backend. Default training remains the
Parquet/cache path. No model, loss, Stage 4 root, SQLite database, or
inference code changed.

Implementation commit: `4d0a217` (`feat(data): add opt-in mmap BC training backend`).

## Contract

The benchmark uses `data/bc_data/2026-08-08.parquet`, SHA-256
`ed4b392a41dc363d7175bffcdfdca105a21450fc209c36320a561e3631a8f1f0`,
`max_rows=2048`, `val_frac=0.1`, `seed=13971479023478`, 12 selected episodes,
2,082 rows, 1,955 train rows, and 127 validation rows. Stage 4 has 1,302,151
parameters and resumes from `stage4_root.pkl`, SHA-256
`b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Evidence

- Parity: 86 columns, dtypes/shapes/values and episode-side-order digests
  equal, zero mismatches.
- Candidate data digest:
  `6d15639abce2fbe240e92c6a86c2bff32330684105a1b7f07f27e379702949c3`.
- Clean load: baseline 1.866 s and 2,802,502,248 decoded bytes versus
  candidate 0.162 s and 85,199,604 bytes.
- Clean RSS: baseline 7.98 GiB versus candidate 273 MiB.
- Integrated train: baseline 21.51 s versus candidate 20.82 s, both 9
  microbatches, 9 optimizer steps, `val_acc=0.6693`, and identical logged
  losses/auxiliary metrics.
- Focused tests: 2 passed.

## Caveats

Candidate ETL took 1.60 s. One epoch including that upfront cost is slightly
slower than an already-built Parquet source, so reuse across epochs is
required. The fixed-probe candidate requires one source day and TBPTT. No
tournament was run because inference behavior is unchanged.

## Next experiment

Pack a larger multi-day subset and run a bounded two-to-three-epoch workload to
measure ETL amortization and pressure behavior before considering broader
default integration.
