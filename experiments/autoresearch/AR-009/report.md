# AR-009 report

## Decision

The smallest update-ready Stage4 policy experiment is complete. It uses one
bounded FP32 PPO epoch over one random episode and one frozen-weight
`mirror_no_memory` episode. No GRPO, long training, or tournament was run.

The frozen Stage4 root was not modified. Its SHA-256 is
`b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Implementation

- `scripts/rl/trajectory_probe.py` verifies the approved root hash before the
  strict inference loader is called, rejects conflicts in both
  `current.date` and `current.archive_date`, and keeps explicit `--meta-date`
  plus day 31 fail-closed behavior.
- The parsed deck-content SHA-256 and source file-byte SHA-256 are separate:
  `606a775392ffe25e058b19c17801d58a4bf30f7cd8c62782388d3de7e7eb5283` and
  `337186f9422f300e50225d6305570f008eb262ac46f519c62aa115df6dcc75d2`.
- `scripts/rl/ppo_micro_update.py` retains model inputs, real action masks,
  detached memory inputs, actions, behavior logprobs, values, rewards, and
  done flags in collection order. It writes the full compressed bundle and a
  tensor-digest-linked sample manifest. The update treats each detached
  recurrent memory input as the truncated-BPTT boundary, so no gradient crosses
  a collected step's memory input.
- `public_agents/submissions/latest-submission-300elo/main.py` accepts an
  opt-in `PTCG_MODEL_PATH`. When unset, its existing checkpoint search remains
  unchanged. When set, it uses the repository's strict loader and the selected
  checkpoint's dtype.

## Collection and PPO evidence

| Quantity | Result |
| --- | ---: |
| Episodes | 2 |
| Random rows | 92 |
| Mirror rows | 83 |
| Total samples | 175 |
| Terminal rows | 2 |
| Collection time | 3.626605 s |
| PPO epochs | 1 |
| Return range | -1.0 to 1.0 |
| Normalized advantage standard deviation | 1.0 |
| PPO loss | 0.7965782881 |
| Value loss | 1.5931565762 |
| Root-reference KL mean | 0.0005297892 |
| Root-reference parameter L2 delta | 0.0092003815 |
| Root-reference max parameter delta | 0.0000101328 |

The sample manifest SHA-256 is
`cf1a580cb808b600bd84cd8194b1fab7c8d3d53e0ee33d1a7a471cd314e56def`.
The compressed full bundle SHA-256 is
`e7bdd424865b1d3e7b0fb714cce1ea9615edd9cc10f2e37b49d0e29fb8d6fd75`.

The candidate is
`experiments/autoresearch/AR-009/candidate.pt`, with SHA-256
`c23ec42ce559db77894e7accd46e131462a276c1092cafac74ec1c66f1291542`.
Its payload contains the exact root hash, sample manifest hash, PPO config,
diagnostics, and FP32 state dict. It was loaded successfully through
`rl.policy_infer_torch.load_inference_checkpoint` and through the opt-in agent
import smoke check.

## Verification

```text
uv run --locked pytest -q tests/test_trajectory_probe.py
10 passed in 1.19s

uv run --locked python scripts/rl/trajectory_probe.py --meta-date 2026-08-16 ...
exit 1: MetaLookupError, day_id 31 is not registered in the meta catalog
```

The existing `scripts/validate/test_policy_infer_torch.py` suite was also
attempted. Its five tests require the absent legacy path
`model/bc_model/bc_best_mlx_final.pkl`; all five failed at that missing-file
precondition. The AR-009 candidate strict-load test uses the available frozen
Stage4 root and passes in the focused suite.

`git diff --check` passed. Tournament evaluation remains a reviewer-gate action
and was not run.

## Limitations

This is a two-episode contract and mutation smoke experiment, not a win-rate
estimate. `CabtEnv(seed=...)` does not expose the engine's `BattleStart` seed,
so full fresh-process replay is not guaranteed. The mirror opponent intentionally
resets its recurrent memory at every opponent decision. The candidate therefore
needs reviewer-gated tournament evaluation before any promotion claim.
