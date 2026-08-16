# AR-001 worker report

Captured 2026-08-16. The designated checkout was `develop` at starting HEAD
`20d7d0dc9e5b7c71a1c8d1a7e409db2cde366270`. The only committed source change
is `dccef4f744b6cdb5b0f93de9d37f6c660f9912d6`, which fixes tournament deck
provenance for packaged agents and records the opponent deck ID in the
no-sweep JSON path. `PROGRAM.md` and the AR-001 artifacts were pre-existing or
worker-generated untracked files and were not committed.

## Frozen Stage 4 identity

- Root MLX checkpoint: `experiments/autoresearch/root/stage4_root.pkl`
  - SHA-256: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`
- Root FP32 package: `experiments/autoresearch/root/stage4_root_fp32.tar.gz`
  - SHA-256: `32add97ad0848cc097a983a45a75a935532a807645b9e36d972bd6fee1c49751`
- Architecture: `d_model=128`, `nhead=4`, `nlayers=4`, `ff_dim=512`,
  `scratch_registers=32`, `static=true`, `split_heads=true`,
  `structured=false`; 1,302,151 parameters; checkpoint epoch 36.
- Root package deck hash: `4f1b76821b9dd638a25ed976701cad1bdfdddf4507c804bc2e950eae00099c97`.

## Synchronous component benchmark

The five timings below are component measurements, not one demonstrated
ETL-to-training chain. The ETL probe wrote a temporary Parquet artifact from
`2026-08-12` (426 rows), while the load and train probes read the existing
`2026-08-08` artifact (2,082 rows). Treat the component timings as observed
measurements and the end-to-end chain as unverified.

All commands used `uv run`; full stdout/stderr and `/usr/bin/time -l` output
are retained in this directory.

| Stage | Command/result | Time | Throughput | Memory/pressure |
| --- | --- | ---: | ---: | --- |
| ETL | 2 episodes from `data/bc_replay_zip/2026-08-12.zip` -> Parquet; 426 rows; `would_ko=computed` | 4.03 s | 105.7 rows/s | 393 MiB RSS; 0 swaps |
| Load | Stage-4 columns, 12 episodes/2,082 rows; 2,048 rows fetched | 1.113 s | 1,839 rows/s | 5.63 GiB RSS; 1.401 GB decoded row-group; 0 swaps |
| Train | Stage-4 root resume, one epoch, 9 TBPTT micro/optimizer steps, 2,082 rows | 189.23 s / 3m07s | 11.0 rows/s; 0.0476 steps/s | 6.20 GiB MLX peak; 5.79 GiB RSS; 0 swaps |
| Rollout | One real game from the FP32 package vs `lb600_dragapult_ex` | 0.448 s | 2.23 games/s | 612 MiB RSS; 0 swaps |
| Tournament | 2 real games, one packaged Stage-4 deck vs named `lb600_dragapult_ex` | 0.751 s matchup; 4.72 s process | 2.66 games/s matchup; 0.424 games/s process | 623 MiB RSS; 0 swaps |

Benchmark evidence:

- `etl.log`
- `load.log`
- `train.log`
- `rollout.log`
- `tournament_fixed.log`
- `tournament.json`
- `tests.log`

The valid tournament result is deck-conditioned: our Stage-4 deck `284` vs
opponent `lb600_dragapult_ex`, opponent deck `7`, 2 games, W/L/D `0/2/0`, WR
`0.0%`. This is a smoke gate only; with `n=2` it has no stable competitive
interpretation. The earlier `tournament.log` is retained as a diagnostic run
and is not evidence because it reported `opp_deck_id: null`.

## Operational path recovered

```text
replay ZIP
  -> scripts/bc/build_bc_from_zips.py
  -> per-day Parquet, schema 3, would-KO manifest
  -> bc_train_mlx.py episode scan + train/val split
  -> TBPTT metadata/location scan
  -> Parquet row-group cache -> MLX FP16 forward/backward, FP32 reductions
  -> FP32 tar package -> main.py -> policy_infer_torch.py
  -> cabt rollout -> scripts/tournament.py -> ResultsDB/report JSON
```

The measured load path decodes a whole Parquet row-group for 2,048 requested
rows: 1.401 GB of arrays for a small logical sample. The one-epoch training
probe also reported 88 SSD spills and 9.14 GiB of spill files. This is the
strongest current pipeline bottleneck signal. It is a harness/data-layout
finding, not evidence for changing the model.

## Rare-event loss reconstruction

### OBSERVATION: current Parquet auxiliary objective

`scripts/bc/build_bc_dataset.py` computes four row targets. `aux_ko` and
`aux_prize_delta` look ahead through the remaining valid states of the same
turn and intentionally repeat across decisions in that turn. `reward` and
`aux_return` use the next valid decision, then reverse-accumulate a telescoping
return; the terminal row adds `+1/-1` for win/loss. Invalid prize/turn states
get all auxiliary fields zeroed with `aux_valid=0`.

`rl/policy_mlx.py` has four heads: `ko_head_aux` (BCE), `prize_head_aux`
(MSE), `terminal_head_aux` (BCE), and `return_head_aux` (MSE). With
`split_heads=true`, all four read the same `value_tok` source; otherwise they
read CLS.

In the live trainer, `aux_valid` masks every head. The configured Stage-4
The AR-001 probe used weights 0.5 for all four heads. The checked-in JSON
configuration also uses 0.5 for all four heads, while the `TrainConfig`
dataclass default for `aux_return_weight` is 1.0. The current `_aux_loss` code
computes, for the configured probe weights:

```text
0.5 * sum(valid * ko_bce)
+ 0.5 * sum(valid * prize_mse)
+ 0.5 * sum(valid * terminal_bce)
+ 0.5 * sum(valid * return_mse)
```

There is no `max(sum(valid), 1)` denominator in the current implementation,
despite the stale docstring saying the function returns a valid-row mean. The
outer optimizer step divides gradients by total batch examples, so the
effective backprop scale is `sum(valid * loss) / total_rows`, not
`sum(valid * loss) / valid_rows`. Reporting metrics use the valid-row
denominator independently.

### OBSERVATION: Stage 1/2/3/4 behavior

The curriculum logs show the same four-head objective and weights in all four
stages. The stage changes are data selection and optimizer schedule:

- Stage 1: no top-Elo filter, 25 days, 30,000 rows/day, 752,085 rows in the
  logged continuation, epochs 11-25.
- Stage 2: top-600 on both sides, 2 days, 300,000 rows/day, 600,136 rows,
  epochs 26-30.
- Stage 3: top-100 on both sides, 2 days, 300,000 rows/day, 445,823 rows;
  the logged run reached epochs 31-32 before Stage 4 resumed.
- Stage 4: top-100 on both sides, 4 days, 80,000 rows/day, 320,235 rows,
  epochs 33-37; final historical `val_acc=0.5949`.

### OBSERVATION: exact historical change

Commit `0dec40f7115a244152ce374721a9acbd44d3422c` changed the current Parquet
aux path by removing `valid_sum = max(sum(valid), 1)`, removing division by
that value from all four heads, and removing the TBPTT `aux_mean *
total_aux_rows` rescaling. The Stage-4 training log after that history reports
per-step logged auxiliary sums around `900-1,013`, while its validation
auxiliary metrics remain around `0.08-1.42`. No later source change before the
worker commit restores the denominator.

### OBSERVATION: separate older prospective planner

Before the Parquet aux-head rewrite, commit `10c3750b` contained a different
prospective planner objective. It counted valid and positive targets on the
training split and assigned each rare class the inverse-frequency ratio
`(valid - positive) / positive`, with fallback `1.0`. Its `mean_valid` denominator
was `max(sum(valid * event_weight), 1)`. That rare weighting affected scalar
return, scalar value, KO BCE, terminal BCE, prize MSE, and uncertainty loss;
the group-relative policy loss had its own validity mask and was not directly
rare-weighted. Commit `a9423734` removed that planner/sidecar path and replaced
it with the four Parquet auxiliary heads. Therefore the available source does
not justify treating the older prospective weighting as the exact Stage-1 to
Stage-4 loss.

### INTERPRETATION

The repository narrative that calls Stage 4 “loss-corrected” is not consistent
with the live `0dec40f` source change: the current objective is normalized by
all rows at the optimizer boundary, while its reporting metrics are normalized
by valid rows. The historical rare-event mechanism is therefore two distinct
episodes: an explicitly inverse-frequency-weighted prospective planner, then
the later Parquet aux-head objective whose denominator was removed. This is a
provenance finding, not a recommendation to alter the frozen model in AR-001.

## Limitations and smallest next hypothesis

- Full QA after `pytest` was installed: 76 passed, 5 failed in 22.19 s. The
  failures are the known SQLite foreign-key/referential-integrity debt
  (2,946,336 violations), one PageRank formula-presence assertion and one
  missing `DECKS_GENERATED` symbol. The complete output is in `pytest.log`;
  this is not evidence that the AR-001 harness patch failed.
- The ETL, load and train measurements are not a single data-identical chain;
  this is a P1 evidence gap, not a performance conclusion.
- No recorded conversion manifest/hash links the frozen MLX checkpoint
  directly to the FP32 tarball. The package/model hashes are stable, but the
  conversion provenance remains incomplete.
- The training probe is deliberately tiny and its validation accuracy is not
  comparable to the historical Stage-4 run.
- The tournament has only two games against one named opponent; it cannot
  estimate competitive strength.
- The current SQLite database and a small set of documentation/generated-symbol
  assertions remain red; those are separate from the throughput experiment.
- The current cache materializes full row-groups and spills frequently. The
  smallest next pipeline hypothesis is a read-only packed or smaller-row-group
  model-ready shard benchmark with the same columns, labels, episode/side/step
  ordering, aux fields, and Stage-4 model semantics. Compare row-group load
  bytes, train rows/s, RSS, and swaps before changing any model or loss.
