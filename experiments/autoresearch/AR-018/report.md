# AR-018 - True recurrent two-sided self-play

Captured on 2026-08-16. This experiment implements the first self-play
correctness gate after the bounded PPO branch. It does not train, mutate
weights, run GRPO, run RoPE-ND, run a tournament, or package a submission.

## Implementation

- Foundation commit: `3867171` (`aaaaaaa inferno`), which adds the stateful
  mirror lane and complete logical-action logprob fields to the existing
  collector.
- Probe and test commit: `434d3f6` (`feat(rl): add true recurrent two-sided self-play probe`).
- Executable: `scripts/rl/true_recurrent_selfplay_probe.py`.
- Hot path: direct `CabtEnv` observations through the Stage 4 encoder and
  PyTorch inference model. No Parquet rows and no packed dataset are read.
- Frozen root was loaded strictly and was not modified.

The mirror owns one memory lane per episode. It passes the same
`memory_in` to every conditional substep of a logical decision and commits
only the `memory_out` from the last executed substep, matching the semantics
in `agent/main.py`. `CabtEnv` continues to supply the mirror's own tracker,
ability slots, and deck context. The learner and mirror use independent
memory tensors.

Each substep retains `action_logprob`. Once the logical decision is complete,
all of its records receive `logical_action_logprob` and `decision_logprob`,
both equal to the finite sum of the conditional substep logprobs. The test
also verifies that an identical learner and behavior snapshot produces an
importance ratio of exactly one within tolerance.

## Final smoke evidence

Command:

```text
uv run --locked python scripts/rl/true_recurrent_selfplay_probe.py \
  --meta-date 2026-08-12 --games 1 --seed 18018
```

| Metric | Result |
| --- | ---: |
| Games | 1 |
| Agent decisions | 85 |
| Mirror decisions | 73 |
| Total logical decisions | 158 |
| Total substep records | 168 |
| Collection time | 3.198051 s |
| Records/s | 52.532 |
| Decisions/s | 49.405 |
| Agent side | 0 |
| Mirror side | 1 |
| Agent terminal return | -1.0 |
| Mirror terminal return | +1.0 |
| Agent lane continuity | true |
| Mirror lane continuity | true |
| Parquet used | false |
| Packed path used | false |

Both lanes reset to the same Stage 4 learned-init digest
`266b74f0f530281c0419451c836465bd276dcc563ad34b557751ea5272c46916`.
Their decision input and committed-output chains are independent:

- agent input chain: `7badd7447184b22a8e1d17a0fc69a9a2b9ddb432e9fd46f3e2d117ce5b9f190f`
- agent output chain: `5d47a67edb2073cb500a7220dca593a042777bae0c5cdfd619f341fe6a91ad1f`
- mirror input chain: `d84c108e6dc481b563af4edd722af1d70d667e95fa5a3fa21c757adf17e15f8b`
- mirror output chain: `35a063e46fdd990b6e0c3ecce2aa17dbc80da638a45f588be60bbfb5a3de61e2`

## Validation

- `uv run --locked pytest -q tests/test_trajectory_probe.py tests/test_ar010_candidate_path.py`: `30 passed in 2.30s`.
- `uv run --locked python -m py_compile scripts/rl/trajectory_probe.py scripts/rl/true_recurrent_selfplay_probe.py tests/test_trajectory_probe.py`: exit 0.
- `git diff --check`: exit 0.
- The executable probe help path returned exit 0.

The compact manifest is `manifest.json`; per-substep evidence is in
`logs/selfplay.jsonl`. The JSONL contains digests, sides, legal action sets,
actions, logprobs, and decision indices, but no model tensors or serialized
rollout buffers.

## Provenance

- Frozen Stage 4 root SHA-256:
  `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- `manifest.json` SHA-256:
  `2bc7c8b9dae8afdc9001c1f5528cc59ade9033cbc50109ae4275b51e4a985e40`.
- `logs/selfplay.jsonl` SHA-256:
  `4878a6c2b7af7fa26d907cd63ede14acee39b3e2daf9b139b1a0545cda387ac7`.
- Root and historical candidate binaries were not changed or added by
  AR-018.

## Limitations and next control point

This is a one-game correctness and throughput smoke, not a competitive result.
The engine remains stochastic, so the observed `-1/+1` terminal outcome is not
a strength estimate. The collector is not batched yet, and the recurrent
learner update is not implemented in this experiment. The next task is the
first real trajectory-group GRPO micro-update using this two-sided buffer,
followed by a comparable tournament gate.
