# State Capsule 067 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-17T12:35:23.214692+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-021 collected `56` exact recurrent sibling groups
  and `122` fibers with effective K
  `[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]`.
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `0.97`, or emitted a no-op when all groups were
  zero-variance.
- Candidate: `ac4798935317bddce6767683c9a39574cdbbe74da886ce77f4fae62825ebf4db`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-038-C067/report.md`
- `experiments/autoresearch/AR-038-C067/manifest.json`
- `experiments/autoresearch/AR-038-C067/metrics.json`
- `experiments/autoresearch/AR-038-C067/sample.manifest.json`
- `experiments/autoresearch/AR-038-C067/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-038-C067/candidate.pt`

## Metrics

- Collection: `49.613786` s,
  `139.75954089704163` decisions/s.
- Update: `501.24367541703396` s; `3` optimizer steps.
- Credited logical actions: `6934`.
- Parameter L2 delta: `0.0006041144532329378`;
  gradient norm `1.2279994487762451`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
