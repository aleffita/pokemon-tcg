# State Capsule 021 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-16T18:04:32.326997+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `4` exact recurrent sibling groups
  and `14` fibers with effective K
  `[4, 4, 4, 2]`.
- One grouped FP32 policy-only update applied independent group-relative
  terminal credit through future continuation with discount
  `0.97`.
- Candidate: `52702295763ecee036e4f6bfaac6660df6ca5ec1cfca66efab5146ae8b292718`; preflight passed.
- Controlled same-deck candidate-vs-root tournament: `22-8-0` in 30 games;
  candidate panel `8-22-0`; frozen-root panel `7-23-0`.
- Root-relative win is strong but external-panel strength is weak; keep AR-021
  experimental and retain root fallback. No promotion, RoPE-ND, MoE, or
  historical ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-021/report.md`
- `experiments/autoresearch/AR-021/manifest.json`
- `experiments/autoresearch/AR-021/metrics.json`
- `experiments/autoresearch/AR-021/sample.manifest.json`
- `experiments/autoresearch/AR-021/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-021/candidate.pt`
- `experiments/autoresearch/AR-021/tournament_candidate_vs_root_same_deck_30.json`
- `experiments/autoresearch/AR-021/tournament_candidate_panel_10.json`
- `experiments/autoresearch/AR-021/tournament_root_panel_10.json`

## Metrics

- Collection: `15.582338` s,
  `69.24506432020944` decisions/s.
- Update: `8.520291083026677` s; one optimizer step.
- Credited logical actions: `1079`.
- Parameter L2 delta: `0.009213928003272408`;
  gradient norm `0.5238350033760071`.

## Next control point

Train the next bounded grouped sibling-fiber hypothesis with external-opponent
deck strata, then rerun the same controlled gate. Keep root fallback until the
external panel improves, not merely the root-relative score.
