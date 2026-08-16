# State Capsule 015 - AR-014 tournament gate

Captured: 2026-08-16 after the first repository-local candidate tournament.

## Ground truth

The AR-010 candidate was loaded only through the explicit absolute
`PTCG_MODEL_PATH` plus matching expected SHA, with `--no-sweep` and no package
or submission. The frozen-root comparison used `agent/main.py`, whose loaded
parameters were verified tensor-equal to the frozen Stage 4 checkpoint.

| Evaluation | Candidate | Frozen Stage 4 root |
| --- | ---: | ---: |
| random, n=10 | 5-5 | 2-8 |
| first, n=10 | 2-8 | 2-8 |
| lb826_alakazam_seok, n=10 | 2-8 | 2-8 |
| candidate vs frozen root, n=10 | 8-2 candidate | comparison |

The initial n=2 smoke was 0-6 for both candidate and root against random,
first, and lb826. The larger runs supersede that low-sample signal for the
same opponents. No competitive promotion is justified: candidate absolute
strength remains weak and it is not consistently above the public opponent.

## Decision

Keep the candidate as an experimental branch only. Keep Stage 4 as the sole
promoted policy and fallback. The candidate's 8-2 relative result against the
root is enough to justify more compute, but not enough to change the champion.

## Next hypothesis

Increase self-play/PPO sample compute from the two-episode-per-mode probe, with
explicit terminal-return and opponent provenance, then repeat the same
candidate-vs-root and public-opponent tournament gate. Do not reopen MoE or
RoPE-ND without evidence from this simpler path. If the larger candidate does
not improve absolute tournament strength, revert to the frozen root and move
to packaging/provenance near the deadline.

## Evidence

- `experiments/autoresearch/AR-010/tournament_candidate_smoke.json`
- `experiments/autoresearch/AR-010/tournament_root_smoke.json`
- `experiments/autoresearch/AR-010/tournament_candidate_vs_root.json`
- `experiments/autoresearch/AR-010/tournament_candidate_lb826_10.json`
- `experiments/autoresearch/AR-010/tournament_root_lb826_10.json`
- `experiments/autoresearch/AR-010/tournament_candidate_random_first_10.json`
- `experiments/autoresearch/AR-010/tournament_root_random_first_10.json`
- `experiments/autoresearch/AR-013/review.md`
