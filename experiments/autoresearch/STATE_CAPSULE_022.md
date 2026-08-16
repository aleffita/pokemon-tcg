# State Capsule 022 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-16T18:11:52.163720+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `4` exact recurrent sibling groups
  and `8` fibers with effective K
  `[2, 2, 2, 2]`.
- One grouped FP32 policy-only update applied independent group-relative
  terminal credit through future continuation with discount
  `0.97`.
- Candidate: `0fb2fed2282298cb2e1e2f9cf14ca28b101735c5e839f303abba6f9d49da0c1a`; preflight passed.
- Controlled same-deck candidate-vs-root tournament: `13-17-0` in 30 games;
  candidate panel `7-23-0`; frozen-root panel `8-22-0`.
- Reject AR-022 for promotion. Three of four training groups had zero
  variance, so only 220 logical actions received credit; root remains fallback.
  No RoPE-ND, MoE, or historical ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-022/report.md`
- `experiments/autoresearch/AR-022/manifest.json`
- `experiments/autoresearch/AR-022/metrics.json`
- `experiments/autoresearch/AR-022/sample.manifest.json`
- `experiments/autoresearch/AR-022/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-022/candidate.pt`
- `experiments/autoresearch/AR-022/tournament_candidate_vs_root_same_deck_30.json`
- `experiments/autoresearch/AR-022/tournament_candidate_panel_10.json`
- `experiments/autoresearch/AR-022/tournament_root_panel_10.json`

## Metrics

- Collection: `9.909958` s,
  `69.12239241669995` decisions/s.
- Update: `3.4491552088875324` s; one optimizer step.
- Credited logical actions: `220`.
- Parameter L2 delta: `0.009192485631226077`;
  gradient norm `0.8533870577812195`.

## Next control point

Keep the frozen Stage 4 root fallback. The external-deck update failed its gate;
the next hypothesis must improve branch diversity or opponent-policy realism
before more compute is allocated.
