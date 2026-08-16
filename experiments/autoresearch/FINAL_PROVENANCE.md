# Final autoresearch provenance - 2026-08-16

## Release decision

The frozen Stage 4 policy is the only promoted policy and the safe fallback.
The PPO candidates are experimental evidence and are not packaged for
submission.

## Frozen root

| Artifact | SHA-256 |
| --- | --- |
| `experiments/autoresearch/root/stage4_root.pkl` | `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b` |
| `experiments/autoresearch/root/stage4_root_fp32.tar.gz` | `32add97ad0848cc097a983a45a75a935532a807645b9e36d972bd6fee1c49751` |

The root architecture is Stage 4: 1,302,151 parameters, d_model 128, four
attention heads, four layers, FFN 512, and 32 scratch registers. The root is
tracked and must not be overwritten by candidate experiments.

## Best experimental candidate

AR-015 is retained as the best experimental candidate by balanced evidence,
not promoted strength:

| Artifact | SHA-256 |
| --- | --- |
| `experiments/autoresearch/AR-015/candidate.pt` | `62ea62eb7ee5f1baca1f50a5b9713ef83b96dd2f2d0ea707151739058d235ff6` |
| `experiments/autoresearch/AR-015/sample.manifest.json` | `96775c83ec0bac8e99d4f087bd439a7b05c00974fb0251fb702e135cef8cd573` |
| manifest canonical content | `bd0367154fc348ac6b37d8679a1ce1f12587d8142bc447eb0fed874f3733e5b3` |
| `experiments/autoresearch/AR-015/trajectory_bundle.pt.gz` | `2a8926080eba535a57360fd8b6f2d934cd2a293bc10de40c4f210f768f0aff97` |

AR-015 tournament: 17-23-0, 42.5%, with 8-2 versus the frozen root, 6-4
versus random, 2-8 versus first, and 1-9 versus `lb826_alakazam_seok`.

AR-017 is retained as a negative-control scale-up. It produced 12-28-0, 30%,
so it is not a candidate for release.

## Opt-in safety contract

The experimental public-agent path requires both:

```text
PTCG_MODEL_PATH=/absolute/path/to/experiments/autoresearch/AR-015/candidate.pt
PTCG_EXPECTED_MODEL_SHA256=62ea62eb7ee5f1baca1f50a5b9713ef83b96dd2f2d0ea707151739058d235ff6
```

The candidate byte hash is checked before provenance and model loading. The
default path does not enter this gate. Tournament runs must remain
repository-local, use absolute paths and `--no-sweep`, and must not package or
submit AR-015.

## Evidence index

- `experiments/autoresearch/experiment_ledger.jsonl`
- `experiments/autoresearch/STATE_CAPSULE_017.md`
- `experiments/autoresearch/AR-015/tournament_ar015_10.json`
- `experiments/autoresearch/AR-017/tournament_ar017_10.json`
- `experiments/autoresearch/AR-010/tournament_candidate_vs_root.json`
- `experiments/autoresearch/AR-010/tournament_root_random_first_10.json`

## Validation note

The autoresearch focused suites pass (`28 passed` for the final AR-016/AR-013
path and trajectory coverage). The full repository command
`uv run --locked pytest -q` remains blocked during collection by the existing
`scripts/validate/test_bc_train_progress.py` import of the missing
`_standard_microbatch_count` from `scripts/bc/bc_train_mlx.py`; this is outside
the autoresearch changes and is recorded rather than hidden.
