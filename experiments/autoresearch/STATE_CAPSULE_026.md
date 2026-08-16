# State Capsule 026 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-16T18:48:29.906613+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-026 collected `8` exact recurrent sibling groups
  and `20` fibers with effective K
  `[2, 4, 2, 2, 4, 2, 2, 2]`.
- Branch sampling mixed policy probability with uniform legal-action mass at
  `0.5`. One grouped FP32 policy-only update applied independent group-relative
  terminal credit through future continuation with discount
  `0.97`.
- Candidate: `af13a7ece3bca7c42760091b86478458b2e028c9130a6c9776010ede377347c2`; preflight passed.
- The candidate won the direct root gate `21-9-0` and the panel `15-45-0`
  versus frozen root `12-48-0`. Keep as the current experimental direction,
  but do not promote while absolute field strength is low.
- No RoPE-ND, MoE, or historical ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-026/report.md`
- `experiments/autoresearch/AR-026/manifest.json`
- `experiments/autoresearch/AR-026/metrics.json`
- `experiments/autoresearch/AR-026/sample.manifest.json`
- `experiments/autoresearch/AR-026/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-026/candidate.pt`

## Metrics

- Collection: `13.648123` s,
  `105.06939177131154` decisions/s.
- Update: `8.569577292073518` s; one optimizer step.
- Credited logical actions: `601`.
- Parameter L2 delta: `0.00923267879010395`;
  gradient norm `0.9121775031089783`.

## Next control point

Scale the same `uniform_mix=0.5` branch-diversity hypothesis to sixteen
groups, then require a stronger panel before promotion. Frozen Stage 4 remains
fallback.

## Tournament evidence

- Candidate vs frozen root, same deck, 30 games: `21-9-0` (70.0%);
  report SHA-256 `bcc1041c9f1e978745c50b30ec5d6f678eb64a3075616fdb94749d135cb8463a`.
- Candidate panel, 60 games: `15-45-0` (25.0%);
  report SHA-256 `b4d6b1735df149182f1919266fe3cc345f6d02e35f9ac35eb95945887ca7fa7f`.
- Frozen-root panel, 60 games: `12-48-0` (20.0%);
  report SHA-256 `49024a62a7d53f64cc2ec05731d5e1b11de89a62f0a1d1d550579efba57c2bda`.
