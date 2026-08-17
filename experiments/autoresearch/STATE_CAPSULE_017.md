# State Capsule 017 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T03:25:44.629417+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `52` exact recurrent sibling groups
  and `104` fibers with effective K
  `[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `cc64760c5a393bef817e8774a3f4e318a0f05ed90d4983fccf94ede412bc6bb3`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C017/report.md`
- `experiments/autoresearch/AR-038-C017/manifest.json`
- `experiments/autoresearch/AR-038-C017/metrics.json`
- `experiments/autoresearch/AR-038-C017/sample.manifest.json`
- `experiments/autoresearch/AR-038-C017/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C017/candidate.pt`

## Metrics

- Collection: `48.029281` s,
  `130.6286483677962` decisions/s.
- Update: `478.33253849996254` s; `3` optimizer steps.
- Credited logical actions: `6274`.
- Parameter L2 delta: `0.0006045006356399151`;
  gradient norm `2.7391409873962402`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
