# State Capsule 017 - exploration closed

Captured: 2026-08-16 after the final bounded PPO tournament.

## AR-017 result

The eight-episode candidate used four `random` and four `mirror_no_memory`
episodes, producing 586 samples at 74.193 rows/s. Terminal returns were mixed
(-1 to +1), and strict provenance/root checks passed.

| Opponent | W-L-D | Win rate |
| --- | ---: | ---: |
| lb826_alakazam_seok | 1-9-0 | 10% |
| frozen Stage 4 root | 6-4-0 | 60% |
| random | 3-7-0 | 30% |
| first | 2-8-0 | 20% |
| Total | 12-28-0 | 30% |

The larger collection was worse than AR-015's 17-23 (42.5%). The candidate
is rejected for promotion and remains negative evidence, not a fallback.

## Final policy state

- Promoted policy: frozen Stage 4 root.
- Frozen root pickle SHA:
  `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- Frozen root FP32 package SHA:
  `32add97ad0848cc097a983a45a75a935532a807645b9e36d972bd6fee1c49751`.
- Best experimental candidate: AR-015, SHA
  `62ea62eb7ee5f1baca1f50a5b9713ef83b96dd2f2d0ea707151739058d235ff6`.
- AR-017 candidate SHA:
  `37790fe1443e0e802abff0a3f252836ca0cc365c410f5e4d90c8a4bb8fa28de2`.

## Termination decision

Stop PPO exploration for the deadline window. Package the frozen root as the
safe fallback, retain AR-015 and AR-017 with their full manifests/bundles and
tournament JSON as provenance, and do not submit the experimental candidate.
RoPE-ND and MoE redesigns remain unjustified by the tournament evidence.
