# State Capsule 016 - AR-015 tournament gate

Captured: 2026-08-16 after the four-episode PPO candidate evaluation.

## Collection

AR-015 doubled the prior probe budget to two `random` and two
`mirror_no_memory` episodes. It produced 405 ordered samples at 73.367
rows/s, with detached recurrent inputs, legal masks, and terminal returns from
-1 to +1. The selected metadata file remained 2026-08-12 with Parquet SHA
`c9e19e462c053c2476502dbaa14c8316ff73972b5bd7089a6531d20e8ce281dc`.

Candidate SHA:
`62ea62eb7ee5f1baca1f50a5b9713ef83b96dd2f2d0ea707151739058d235ff6`.

Bundle SHA:
`2a8926080eba535a57360fd8b6f2d934cd2a293bc10de40c4f210f768f0aff97`.

## Tournament ground truth

The repository-local no-sweep run used absolute paths and the candidate SHA
preflight:

| Opponent | W-L-D | Win rate |
| --- | ---: | ---: |
| lb826_alakazam_seok | 1-9-0 | 10% |
| frozen Stage 4 root | 8-2-0 | 80% |
| random | 6-4-0 | 60% |
| first | 2-8-0 | 20% |
| Total | 17-23-0 | 42.5% |

The candidate is not promoted. The frozen Stage 4 root remains the only
promoted policy and fallback. The relative 8-2 result justifies one bounded
increase in collection compute, while the 1-9 public result prevents
exploitation-only commitment.

## Next control point

Permit up to four games per probe mode with focused validation, collect a
larger candidate without reopening the architecture, and repeat the same
four-opponent gate. If public win rate does not improve, stop PPO exploration,
revert promotion to the root, and package the reproducible root/candidate
provenance for the deadline.
