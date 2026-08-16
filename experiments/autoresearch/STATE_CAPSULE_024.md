# State Capsule 024 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-16T18:28:32.444817+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-024 collected `4` exact recurrent sibling groups
  and `8` fibers with effective K
  `[2, 2, 2, 2]`.
- One grouped FP32 policy-only update applied independent group-relative
  terminal credit through future continuation with discount
  `0.97`.
- Candidate: `abade9c813286b0480e7fb265cfa659412492dd95bb1aafa21337a839816dcd3`; preflight passed.
- The candidate won the same-deck frozen-root gate `19-11-0` in 30 and
  slightly improved the six-opponent panel `8-52-0` versus frozen root
  `7-53-0` in 60. Absolute strength is low; candidate rejected for promotion
  and frozen Stage 4 remains fallback.
- No RoPE-ND, MoE, or historical ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-024/report.md`
- `experiments/autoresearch/AR-024/manifest.json`
- `experiments/autoresearch/AR-024/metrics.json`
- `experiments/autoresearch/AR-024/sample.manifest.json`
- `experiments/autoresearch/AR-024/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-024/candidate.pt`

## Metrics

- Collection: `5.295967` s,
  `111.59434831059819` decisions/s.
- Update: `2.664541875012219` s; one optimizer step.
- Credited logical actions: `344`.
- Parameter L2 delta: `0.009215549014408107`;
  gradient norm `0.7295721769332886`.

## Next control point

Scale the same four-policy external strata to more independent sibling groups
to address two zero-variance groups. Keep the six-opponent panel as the gate;
do not promote on the one-win aggregate improvement alone.

## Tournament evidence

- Candidate vs frozen root, same deck, 30 games: `19-11-0` (63.3%);
  report SHA-256 `d6d802bbbee1d282efb5a78c8d2871171346b92f3b7254e23ab9e4aca10f6a41`.
- Candidate panel, 60 games: `8-52-0` (13.3%);
  report SHA-256 `9605b84bcdce169335d73bb8288da4524bf2f3ebe726854fa4d8329e3b3caaa0`.
- Frozen-root panel, 60 games: `7-53-0` (11.7%);
  report SHA-256 `2f20385eee72653aa6bfae6fa5cdd4656c35e57dd9fef3a7d3a362bcba6d321a`.
