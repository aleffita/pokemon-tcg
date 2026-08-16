# State Capsule 020 - sibling-fiber GRPO micro-update

Captured 2026-08-16T17:47:00.052564+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-020 collected `7` common-base sibling fibers
  (requested K=4) with branch actions
  `[[2, 0, 1, 3], [0, 2, 1]]` and returns `[[-1.0, -1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]]`.
- One FP32 policy-only update applied branch-relative terminal credit through
  the future continuation with discount `0.97`.
- Candidate: `89a70d4eddb3c856d7c4a4e1ad520e2d23bc7230c76b4c10904c45970eeb8637`; preflight passed.
- Candidate-vs-root tournament: `2-8-0` in 10 games; candidate panel:
  `8-22-0` across lb826, random, and first; frozen-root panel: `3-27-0`.
- Direct gate rejected promotion. The candidate and root panels used different
  packaged default decks, so the panel comparison is directional; root remains
  fallback. No RoPE-ND, MoE, or historical ETL/Parquet/packed-data work was run.

## Evidence

- `experiments/autoresearch/AR-020/report.md`
- `experiments/autoresearch/AR-020/manifest.json`
- `experiments/autoresearch/AR-020/metrics.json`
- `experiments/autoresearch/AR-020/sample.manifest.json`
- `experiments/autoresearch/AR-020/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-020/candidate.pt`
- `experiments/autoresearch/AR-020/tournament_candidate_vs_root_10.json`
- `experiments/autoresearch/AR-020/tournament_candidate_panel_10.json`
- `experiments/autoresearch/AR-020/tournament_root_panel_10.json`

## Metrics

- Collection: `8.839274` s, `64.3718009671044` decisions/s.
- Update: `4.025555834174156` s; ratio mean `1.000110149383545`.
- Credit scope: `branch_and_continuation`; credited logical actions `569`.
- Parameter L2 delta: `0.013461024582650779`; gradient norm `0.8922091722488403`.

## Next control point

Keep the frozen Stage 4 root fallback. The next research control point is a
newly bounded hypothesis with its own dynamic-K, multi-deck rollout and the
same tournament gate.
