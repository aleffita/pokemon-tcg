# State Capsule 003 — AR-002 reviewer gate

Captured: 2026-08-16 after AR-002 review.

## Current champion

Stage 4 remains the only competitive champion and frozen fallback. The packed
backend is infrastructure only and has not changed policy weights or deck.

## Architecture

Unchanged Stage 4 root: `d_model=128`, `nhead=4`, `nlayers=4`, `ff_dim=512`,
`scratch_registers=32`, static/split-heads enabled, structured disabled,
1,302,151 parameters. Root hash remains
`b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Data plane

AR-002 added an opt-in fixed-width NPY mmap backend. On one fixed 2,082-row
selection from `2026-08-08`, parity passed for 86 columns with combined digest
`6d15639abce2fbe240e92c6a86c2bff32330684105a1b7f07f27e379702949c3`. Clean
load decoded 85.2 MB versus 2.80 GB for Parquet/cache and used 272.6 MiB RSS
versus the clean baseline's approximately 7.44 GiB binary RSS.

## Experiment just completed

AR-002 implementation commits `4d0a217` and `2a63b97`; reviewer verdict:
`REWORK before multi-day promotion`, not discard.

## Observed

- Candidate load: 0.162 s; baseline: 1.866 s.
- Candidate ETL: 1.60 s upfront; candidate one-epoch train 20.82 s versus
  baseline 21.51 s, so one epoch including ETL is slower (22.42 versus 21.51).
- The 1/2/3 epoch amortization claim has not been measured directly.
- The combined benchmark JSON's candidate RSS and swap fields are invalid as
  per-run metrics; clean-process logs are canonical.
- The backend does not yet independently validate row-level split/order, has a
  `max_rows=0` parity divergence, and lacks an automated trainer/TBPTT
  integration test.

## Inference

The backend is a high-value infrastructure direction because it eliminates
row-group amplification under exact fixed-probe parity. Its competitive value
is indirect: it earns more research iterations only if the contract remains
correct on larger multi-day data and ETL amortizes.

## Rejected interpretation

Do not promote `--packed-data` to default based on one probe. Do not cite the
combined JSON's 7.77 GiB candidate RSS or global swap counters. Do not call a
0.7 s training difference a stable speedup.

## Decision

Keep the backend opt-in. Rework the contract before any multi-day comparison;
do not change model, loss, root checkpoint, SQLite or inference.

## Next best experiment

AR-003: fix split/order and independent parity contracts, then measure clean
baseline versus reusable packed store over one, two and three epochs on a
larger multi-day strategic subset, with ETL accounted once and separate-process
RSS/pressure telemetry.
