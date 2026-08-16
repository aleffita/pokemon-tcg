# AR-008 report

## Decision

The smallest real Stage4 trajectory probe is implemented and executed. The
trajectory contract is sufficient to begin a next group-relative/PPO
hypothesis at the collection boundary, but it is not promotion-grade evidence
for recurrent self-play or a training update. No GRPO, full self-play
training, long RL update, or tournament was run.

## Implementation

- `scripts/rl/trajectory_probe.py` loads
  `experiments/autoresearch/root/stage4_root.pkl` only through
  `rl.policy_infer_torch.load_inference_checkpoint`; loader failures propagate.
- `CabtEnv` receives the explicit 60-card `agent/deck.csv` deck and the same
  deck for the opponent.
- `--meta-date` is required. The local date-bound encoder injects the explicit
  date only when the engine omits it and rejects conflicting dates. It does not
  modify the global `MetaLookup`.
- Random opponent collection and frozen-weight `mirror_no_memory` collection
  both use the real FP32 model. The mirror callback resets recurrent memory at
  every opponent decision, so it is not called full recurrent self-play.
- Learner rows are ordered JSONL records for every environment sub-action,
  including multi-select substeps, with legal-mask, log-probability, entropy,
  value, reward, terminal/done state, memory input/output digests, side,
  episode/decision/substep indices, opponent mode, deck/model hashes, and
  metadata date.
- The default Parquet file is inspected for schema, row count, and SHA-256
  provenance only. No Parquet rows and no packed store are used as model input.

## Probe evidence

Command:

```text
uv run --locked python scripts/rl/trajectory_probe.py \
  --meta-date 2026-08-12 --games-per-mode 1 --seed 8008
```

| Mode | Episodes | Rows | Terminal rows | Outcome |
| --- | ---: | ---: | ---: | --- |
| `random` | 1 | 114 | 1 | loss, reward -1 |
| `mirror_no_memory` | 1 | 119 | 1 | win, reward +1 |

Total collection time was 3.62 seconds, or 64.45 rows/s. The frozen model
SHA-256 is `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
The deck SHA-256 is
`606a775392ffe25e058b19c17801d58a4bf30f7cd8c62782388d3de7e7eb5283`.
The metadata-only Parquet context is `data/bc_data/2026-08-12.parquet`, with
806,653 rows and SHA-256
`c9e19e462c053c2476502dbaa14c8316ff73972b5bd7089a6531d20e8ce281dc`.

The exact committed trajectory has 233 rows and SHA-256
`df8824943665afe899f31089dfab3ae18ed5b11e37af17926ea1d9f52db1114c`.

## Invariants and limitations

The collector enforces legal sampled actions, finite action log-probabilities,
ordered multi-select substeps, episode-boundary memory reset, one terminal row
and terminal reward per episode, and no truncation before terminal reward.
Learner action sampling uses an isolated seeded `torch.Generator` per episode.

Full trajectory replay is not guaranteed by `CabtEnv(seed=...)`: repeated fresh
processes shared the same initial prefix but diverged after engine state changed.
`BattleStart` exposes no seed through this boundary, so this is a blocker for
strong fixed-seed episode reproducibility, not a reason to claim deterministic
self-play. The current SQLite catalog also has `2026-08-16` as day 31 without a
registered `competition_day`; using that date fails loudly with
`MetaLookupError`. The completed `2026-08-12` date is therefore explicit in the
probe and artifacts.

The two-episode result is a contract smoke probe, not a performance estimate.
Before a group-relative/PPO update, the next gate should decide how to seed or
snapshot engine determinization and whether the opponent callback must retain
recurrent memory. It should then collect a larger reviewed sample before any
training or tournament claim.

## Verification

```text
uv run --locked pytest -q tests/test_trajectory_probe.py
6 passed in 0.58s

uv run --locked pytest -q tests/test_trajectory_probe.py
6 passed in 0.58s

uv run --locked python scripts/rl/trajectory_probe.py --meta-date 2026-08-16 ...
exit 1: MetaLookupError for incomplete day 31

git diff --check
passed
```

The frozen Stage4 root and prior artifacts were preserved. No changes were
made to `pyproject.toml`, `uv.lock`, model/loss/inference/SQLite/deck code, or
binary artifacts. No push was performed.
