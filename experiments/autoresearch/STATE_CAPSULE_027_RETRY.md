# State Capsule AR-027-retry - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-16.

## Current state

- Frozen Stage 4 root remains fallback:
  `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-027-retry used code commit
  `509b948bee7fb1b43503a900257268d9bd14f848`.
- It collected 16 exact recurrent sibling groups and 41 fibers with effective
  K `[2, 2, 2, 2, 3, 4, 4, 2, 4, 2, 4, 2, 2, 2, 2, 2]`.
- One grouped FP32 policy-only update applied independent group-relative
  terminal credit through future continuation with discount `0.97`.
- Candidate:
  `6db9eb3496ca00f9a70cbb1f8e24027cacb456219b74e355e18c95783981d24c`;
  preflight passed.
- Tournament decision: reject promotion; retain the frozen root fallback.

## Tournament gate

- Candidate vs root, same deck: `13-17-0` in 30.
- Candidate external panel: `9-51-0` in 60.
- Frozen-root external panel: `12-48-0` in 60.
- Candidate per external policy: lb1009 `0-10`, lb945 `0-10`, lb826 `1-9`,
  lb814 `2-8`.

## Evidence

- `experiments/autoresearch/AR-027-retry/report.md`
- `experiments/autoresearch/AR-027-retry/manifest.json`
- `experiments/autoresearch/AR-027-retry/metrics.json`
- `experiments/autoresearch/AR-027-retry/sample.manifest.json`
- `experiments/autoresearch/AR-027-retry/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-027-retry/candidate.pt`
- `experiments/autoresearch/AR-027-retry/tournament_candidate_vs_root_same_deck_30.json`
- `experiments/autoresearch/AR-027-retry/tournament_candidate_panel_10.json`
- `experiments/autoresearch/AR-027-retry/tournament_root_panel_10.json`

## Metrics

- Collection: `30.201511` s, `100.35921560688664` decisions/s.
- Update: `16.868608332937583` s; one optimizer step.
- Logical decisions / substeps: `3031 / 3185`.
- Credited logical actions: `1291`.
- Zero-variance groups: `10 / 16`.
- Parameter L2 delta: `0.009225403082207767`;
  gradient norm `0.4703052341938019`.

## Next control point

Use the opponent-specific losses to guide the deck swarm and the next
autoresearch hypothesis. Do not promote this candidate or alter the root.
