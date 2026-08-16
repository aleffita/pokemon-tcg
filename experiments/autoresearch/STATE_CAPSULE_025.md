# State Capsule 025 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-16T18:35:57.057264+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-025 collected `16` exact recurrent sibling groups
  and `43` fibers with effective K
  `[4, 2, 4, 2, 2, 2, 2, 2, 2, 2, 3, 4, 2, 2, 4, 4]`.
- One grouped FP32 policy-only update applied independent group-relative
  terminal credit through future continuation with discount
  `0.97`.
- Candidate: `93b46cb113c917d4ea12cb25eb0bdcc7ca6ce31fbdd2ad71e6c5e2f31455bb52`; preflight passed.
- The candidate lost the same-deck frozen-root gate `13-17-0` in 30 and the
  six-opponent panel `10-50-0` versus frozen root `12-48-0` in 60. Candidate
  rejected for promotion; frozen Stage 4 remains fallback.
- No RoPE-ND, MoE, or historical ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-025/report.md`
- `experiments/autoresearch/AR-025/manifest.json`
- `experiments/autoresearch/AR-025/metrics.json`
- `experiments/autoresearch/AR-025/sample.manifest.json`
- `experiments/autoresearch/AR-025/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-025/candidate.pt`

## Metrics

- Collection: `27.275887` s,
  `106.83428855216941` decisions/s.
- Update: `66.6324357080739` s; one optimizer step.
- Credited logical actions: `1088`.
- Parameter L2 delta: `0.009227449781494418`;
  gradient norm `0.7253366112709045`.

## Next control point

Do not scale this configuration further: the larger update increased compute
but reduced field strength and left eleven of sixteen groups zero-variance.
Test a targeted credit or branch-selection hypothesis, keeping the same panel
as the promotion gate.

## Tournament evidence

- Candidate vs frozen root, same deck, 30 games: `13-17-0` (43.3%);
  report SHA-256 `e1f456169857d0903661583baf6d59a899f427656eda72258671fb2bd1f6e4ac`.
- Candidate panel, 60 games: `10-50-0` (16.7%);
  report SHA-256 `e6d277a66fa3b1cfb794ffac90e2e862f61735956fe73e781d8b35e6fea2e5c3`.
- Frozen-root panel, 60 games: `12-48-0` (20.0%);
  report SHA-256 `816c701a72ee415cabb702f6791947b696b72bbd2b3a801f4e45b2b5a5dba740`.
