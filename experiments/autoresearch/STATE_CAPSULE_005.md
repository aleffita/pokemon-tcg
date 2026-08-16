# State Capsule 005 — AR-003 reviewed

Captured: 2026-08-16 after the AR-003 reviewer gate.

## Current champion

Stage 4 remains the frozen competitive root and only promoted policy. Packed
data is infrastructure, opt-in, and not a model candidate.

## Architecture

Unchanged Stage 4 root: `d_model=128`, `nhead=4`, `nlayers=4`, `ff_dim=512`,
`scratch_registers=32`, static/split-heads enabled, structured disabled,
1,302,151 parameters. Root checkpoint SHA-256 remains
`b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Data plane

AR-003 hardened packed format v2 for two Parquet days and 10,035 selected
rows. Independent parity passed 86/86 columns with zero mismatches. Load
decoded bytes fell from 2,858,248,224 to 418,820,760 and process peak RSS
from 9,992,110,080 B to 916,111,360 B. Source-to-ready candidate was faster
only at the two-epoch horizon after one-time ETL.

## Experiment just completed

AR-003 implementation commit `245ae42`; reviewer verdict `keep opt-in /
inconclusive`.

## Observed

- 1 epoch: baseline 134.992 s, candidate 199.187 s source-to-ready.
- 2 epochs: baseline 366.562 s, candidate 330.917 s.
- 3 epochs: baseline 488.962 s, candidate 499.207 s.
- Candidate and baseline used fresh processes, same root/config/selection/seed,
  and 15/30/45 optimizer steps. Peak MLX memory remained about 18.9 GiB in
  both backends; this is not a host RSS reduction.
- Five focused tests passed. The full repository suite remains separately red
  for known database/docs/generated-symbol issues.

## Inference

Packed data removes row-group amplification and is promising for repeated
training, but the current evidence does not justify default promotion or
using it as the RL data backend. The next highest-value work is closing
runtime provenance holes, not another speculative model change.

## Rejected interpretation

Do not call the two-epoch win a universal speedup. Do not treat equal displayed
validation metrics as proof of bit-identical training. Do not begin RL/self-play
from a backend whose resume can silently mix corpus identity.

## Decision

Keep `245ae42` and packed data opt-in. Do not revert the backend. Before any
RL/self-play run that uses packed data, enforce full runtime order/boundary
validation and checkpoint data/backend identity, then run adversarial tests.

## Next best experiment

AR-004: harden runtime packed-store validation and checkpoint-resume identity,
run adversarial focused tests, and repeat one controlled two-epoch measurement
only if the code changes affect the measured path. Then inspect the current
self-play/RL implementation boundary and choose the smallest valid on-policy
trajectory probe.
