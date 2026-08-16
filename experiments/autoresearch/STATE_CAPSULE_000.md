# State Capsule 000 — Bootstrap

Captured: 2026-08-16T00:25:52-03:00

## Current champion

Baseline candidate is the frozen Stage 4 artifact, pending a fresh valid tournament baseline. The working-tree source remains at `20d7d0d` on `develop`; `PROGRAM.md` is an untracked user-supplied operating program and has not been modified.

## Architecture

Stage 4 root checkpoint: `experiments/autoresearch/root/stage4_root.pkl`

- `d_model=128`, `nhead=4`, `nlayers=4`, `ff_dim=512`
- `scratch_registers=32`, `static=true`, `split_heads=true`, `structured=false`
- checkpoint architecture version `1.0.0`, token schema `1.0.0`
- checkpoint payload epoch `36`, model parameter count `1,302,151`
- dtype recorded by the checkpoint: MLX `float16`; current project inference contract is strict FP32 after conversion

Frozen root hashes:

- `stage4_root.pkl`: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`
- `stage4_root_fp32.tar.gz`: `32add97ad0848cc097a983a45a75a935532a807645b9e36d972bd6fee1c49751`

## Data plane

The checkpoint manifest records four model-ready Parquet days, 3,232,363 rows total, with would-KO enabled and schema version 3. The repository currently contains 30 day-partitioned Parquet files through `2026-08-12`, about 1.4 GiB under `data/bc_data`; the Stage 4 training log reports a 320,235-row top-100/80k-per-day filtered run.

## Experiment just completed

Bootstrap evidence recovery and root preservation. No training, self-play or new tournament result is claimed.

## Observed

- Repository root and designated workspace agree: `pokemon-tcg` on `develop`, clean except `PROGRAM.md`.
- HEAD is `20d7d0d`; 144 reachable commits; origin is `origin/develop`.
- Hardware observed: Apple M3 MacBook Pro, 24 GB unified memory, 8 logical CPUs.
- Historical Stage 4 training: epochs 33–37, about 1h35m–1h43m per epoch, peak memory about 11.1 GiB, final `val_acc=0.5949`.
- The historical `stage4_tourn.json` recorded zero games because its command targeted the current `agent/main.py`, not the Stage 4 checkpoint package. It is not baseline tournament evidence.
- The last nonempty historical tournament artifact is a deck-conditioned sweep of a packaged submission against `first_sub_kaggle_2707`; it must not be relabeled as Stage 4 evidence.

## Inference

The first high-value objective is a valid end-to-end throughput and baseline benchmark using the frozen Stage 4 package, followed by a compact data-path probe. The training log suggests the current hot loop is dominated by repeated Parquet/cache decoding rather than model memory capacity, but this is a hypothesis until measured in the new run.

## Rejected interpretation

Validation accuracy and the zero-game Stage 4 report are not evidence of competitive strength. The project documents and historical reports are provenance inputs, not substitutes for a fresh deck-conditioned tournament against named opponents.

## Decision

Keep the Stage 4 root immutable. Dispatch one worker for the first major objective: benchmark the actual Stage 4 package and current dataset→batch→training→rollout→tournament path, recover the exact rare-event loss implementation/history, and propose the smallest measured pipeline improvement. The worker must return artifacts and commit provenance; a single reviewer will attack the result before any subsequent experiment.

## Next best experiment

Fresh screening benchmark of frozen Stage 4 with fixed deck/opponent settings plus ETL/training/rollout/tournament timing, then decide whether the first code change should be a model-ready packed dataset or an evaluation/package correction.
