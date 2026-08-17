# State Capsule 062 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T11:44:00.542854+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `56` exact recurrent sibling groups
  and `122` fibers with effective K
  `[2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `85801ebee23a45276855939dcd960052f1dcb225f16c4430113efe2d5fad99a2`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C062/report.md`
- `experiments/autoresearch/AR-038-C062/manifest.json`
- `experiments/autoresearch/AR-038-C062/metrics.json`
- `experiments/autoresearch/AR-038-C062/sample.manifest.json`
- `experiments/autoresearch/AR-038-C062/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C062/candidate.pt`

## Metrics

- Collection: `49.416` s,
  `141.00696189946223` decisions/s.
- Update: `499.0700469589792` s; `3` optimizer steps.
- Credited logical actions: `6968`.
- Parameter L2 delta: `0.0006042820132908667`;
  gradient norm `1.2944811582565308`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
