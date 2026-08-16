# State Capsule 023 - grouped dynamic-K sibling-fiber GRPO

Captured 2026-08-16T18:22:08.167214+00:00.

## Current state

- Frozen Stage 4 root remains fallback: `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- AR-023 collected `4` exact recurrent sibling groups
  and `12` fibers with effective K
  `[2, 4, 4, 2]`.
- One grouped FP32 policy-only update applied independent group-relative
  terminal credit through future continuation with discount
  `0.97`.
- Candidate: `6c064668e3201deb73bb32be415dc73204e9414b5c2b7c4b50ebdec65e579e4a`; preflight passed.
- The candidate won the same-deck frozen-root gate `20-10-0` in 30 games,
  but lost the external panel `7-23-0` versus frozen root `8-22-0`.
  Candidate is rejected for promotion; frozen Stage 4 remains fallback.
- No RoPE-ND, MoE, or historical ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/AR-023/report.md`
- `experiments/autoresearch/AR-023/manifest.json`
- `experiments/autoresearch/AR-023/metrics.json`
- `experiments/autoresearch/AR-023/sample.manifest.json`
- `experiments/autoresearch/AR-023/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-023/candidate.pt`

## Metrics

- Collection: `9.187725` s,
  `109.38507827590689` decisions/s.
- Update: `6.222183666890487` s; one optimizer step.
- Credited logical actions: `663`.
- Parameter L2 delta: `0.009200024094590262`;
  gradient norm `0.5540380477905273`.

## Next control point

Improve external-policy training signal or branch diversity, then test the
next targeted hypothesis against the same panel. Do not promote AR-023 on its
root-relative win alone.

## Tournament evidence

- Candidate vs frozen root, same deck, 30 games: `20-10-0` (66.7%);
  report SHA-256 `e9becd02602e5befaae5ad266bfcbeb6642db780bb2b345a99ee15ec24ad5188`.
- Candidate panel, 30 games: `7-23-0` (23.3%);
  report SHA-256 `871605969e0b3116b87f7b66c99a3867aa4b0520b4e80fd8fd07fc59bf5fb710`.
- Frozen-root panel, 30 games: `8-22-0` (26.7%);
  report SHA-256 `95d181e17939a28935d7605c63b1b6dd7859aaeb874b7e7eb3c3c367cd7e30af`.
