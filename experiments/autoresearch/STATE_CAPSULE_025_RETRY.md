# State Capsule AR-025-retry - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-16T18:36:04.454499+00:00.

## Current state

- This is a bounded retry of AR-025 with two groups per external-policy
  matchup, not a new algorithm.
- Frozen Stage 4 root remains fallback:
  `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- Eight exact recurrent sibling groups and 17 fibers used effective K
  `[2,2,2,2,2,2,2,3]`; 435 logical actions received credit.
- Candidate `2bd20e999284877a75ca7cdfe3f6be7a53af1269deaff7ce81c8d75e7111700b`
  passed preflight.
- Direct gate: `16-14-0` in 30. Six-opponent panel: `7-53-0`, below the
  frozen-root reference `12-48-0`. Candidate rejected for promotion.

## Evidence

- `experiments/autoresearch/AR-025-retry/report.md`
- `experiments/autoresearch/AR-025-retry/manifest.json`
- `experiments/autoresearch/AR-025-retry/metrics.json`
- `experiments/autoresearch/AR-025-retry/sample.manifest.json`
- `experiments/autoresearch/AR-025-retry/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-025-retry/candidate.pt`
- `experiments/autoresearch/AR-025-retry/tournament_candidate_vs_root_same_deck_30.json`
- `experiments/autoresearch/AR-025-retry/tournament_candidate_panel_10.json`
- `experiments/autoresearch/AR-025/tournament_root_panel_10.json`

## Metrics

- Collection: `12.10168` s, `99.15978642394391` decisions/s.
- Update: `6.603137208148837` s; one optimizer step.
- Credited logical actions: `435`; zero-variance groups: `5` of `8`.

## Decision

Retain as scale diagnostic only. Keep frozen Stage 4 as fallback and do not
promote on a root-relative gate without external-panel strength.
