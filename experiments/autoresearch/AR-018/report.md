# AR-018 - True recurrent two-sided self-play

Captured on 2026-08-16. This experiment is the corrected self-play
correctness gate after the bounded PPO branch. It does not train, mutate
weights, run GRPO, run RoPE-ND, run a tournament, or package a submission.

## Implementation

- Frozen Stage 4 foundation: `3867171`.
- Initial probe and tests: `434d3f6`.
- Correctness repair: `28c2b96` (`fix(rl): close AR-018 recurrent reset gate`).
- Executable: `scripts/rl/true_recurrent_selfplay_probe.py`.
- Hot path: direct `CabtEnv` observations through the Stage 4 encoder and
  PyTorch inference model. No Parquet rows and no packed dataset are read.
- Frozen root was loaded strictly and was not modified.

The mirror owns one memory lane per accepted episode. It passes the same
`memory_in` to every conditional substep of a logical decision and commits
only the `memory_out` from the last executed substep, matching the semantics
in `agent/main.py`. `CabtEnv` supplies the mirror's own tracker, ability
slots, and deck context. The learner and mirror use independent memory
tensors.

The P0 found by review was retry leakage: `CabtEnv.reset()` can discard a
battle after the opponent has already acted. The repaired environment calls an
explicit reset hook before every battle-start attempt, so a discarded attempt
cannot carry mirror memory or events into the accepted episode. The focused
test exercises a two-attempt reset with an opponent call on the discarded
attempt.

Each substep retains `action_logprob`. Once the logical decision is complete,
all of its records receive `logical_action_logprob` and `decision_logprob`,
both equal to the finite sum of the conditional substep logprobs. An
independent learner-snapshot recomputation test verifies the complete-action
importance ratio is one for identical behavior and learner snapshots.

## Final smoke evidence

Command:

```text
uv run --locked python scripts/rl/true_recurrent_selfplay_probe.py \
  --meta-date 2026-08-12 --games 4 --seed 18000
```

| Metric | Result |
| --- | ---: |
| Games | 4 |
| Agent decisions | 353 |
| Mirror decisions | 319 |
| Total logical decisions | 672 |
| Total substep records | 759 |
| Collection time | 6.412162 s |
| Records/s | 118.369 |
| Decisions/s | 104.801 |
| Agent sides | 1, 0, 0, 1 |
| Agent terminal returns | -1, +1, -1, +1 |
| Mirror terminal returns | +1, -1, +1, -1 |
| Agent lane continuity | true for all games |
| Mirror lane continuity | true for all games |
| Parquet used | false |
| Packed path used | false |

Both lanes reset to the same Stage 4 learned-init digest
`266b74f0f530281c0419451c836465bd276dcc563ad34b557751ea5272c46916` in all
four games. The compact JSONL records sides, legal action sets, actions,
substep and logical logprobs, memory input/output digests, and mirror terminal
rewards. Each game has at least one terminal mirror event record.

## Validation

- `uv run --locked pytest -q tests/test_trajectory_probe.py tests/test_ar010_candidate_path.py`: `32 passed in 2.21s`.
- `uv run --locked python -m py_compile rl/env/env.py scripts/rl/trajectory_probe.py scripts/rl/true_recurrent_selfplay_probe.py tests/test_trajectory_probe.py`: exit 0.
- `git diff --check`: exit 0.
- The four-game metadata-bound executable smoke completed with exit 0.
- The focused reset-retry test, end-to-end logprob recomputation test, and
  existing recurrent/logprob tests all passed.

The manifest is `manifest.json`; per-substep evidence is in
`logs/selfplay.jsonl`; the compact run summary is `logs/selfplay.log`. No
model tensors or serialized rollout buffers are stored.

## Provenance

- Frozen Stage 4 root SHA-256:
  `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- Model SHA-256 equals the frozen root SHA-256.
- `manifest.json` SHA-256:
  `674357f618ed2aceb204a190f066c0b3be40eb21fe7eccbf8c87228ffb724f60`.
- `logs/selfplay.jsonl` SHA-256:
  `468a1ca5cd6958ea32a618c29771aed5699d1e6208d20c4c718d8edaa1772b3c`.
- Root and historical candidate binaries were not changed or added by
  AR-018.

## Limitations and next control point

This is a four-game correctness and throughput smoke, not a competitive
result. The terminal outcomes are not a strength estimate. Collection is
still serial and the recurrent learner update is not implemented in this
experiment. The next task is the first trajectory-group GRPO micro-update
using this two-sided buffer, followed by a comparable tournament gate.
