# State Capsule 060 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T11:23:10.753407+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `56` exact recurrent sibling groups
  and `134` fibers with effective K
  `[3, 2, 2, 3, 4, 2, 2, 2, 2, 2, 2, 3, 4, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 3, 4, 2, 2, 2, 3, 2, 2, 4, 3, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 4]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `e39d262319ab9cc859b40dab2d8666f762fa5f45ce2005b454929429a645011b`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C060/report.md`
- `experiments/autoresearch/AR-038-C060/manifest.json`
- `experiments/autoresearch/AR-038-C060/metrics.json`
- `experiments/autoresearch/AR-038-C060/sample.manifest.json`
- `experiments/autoresearch/AR-038-C060/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C060/candidate.pt`

## Metrics

- Collection: `51.28642` s,
  `148.34336215006047` decisions/s.
- Update: `543.955007707933` s; `3` optimizer steps.
- Credited logical actions: `7608`.
- Parameter L2 delta: `0.0006043268805882794`;
  gradient norm `1.3191030025482178`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
