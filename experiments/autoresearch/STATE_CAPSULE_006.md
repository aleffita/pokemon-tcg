# State Capsule 006 - AR-004 reviewed

Captured: 2026-08-16 after the AR-004 reviewer gate.

## Decision

Stage 4 remains the frozen competitive root and only promoted policy. The
packed backend remains opt-in infrastructure. AR-004 closes most runtime
validation holes but is **rework**, not a promotion: production-path resume
integration and several adversarial claims remain incomplete.

## Evidence

- Commit: `f28e13e` (`fix(data): close packed backend provenance gates`).
- Focused tests: 19 passed; log at `experiments/autoresearch/AR-004/logs/tests.log`.
- Reviewer: `experiments/autoresearch/AR-004/review.md`.
- The additional progress test still fails at collection because the preserved
  baseline lacks `_standard_microbatch_count`; this is not an AR-004 change.
- Confirmed runtime checks include all five order columns, boundary and split
  membership, row-order and ordered-source digests, required dedup/TBPTT
  metadata, and explicit failure without TBPTT.

## Open gates

AR-005 must test resume through the actual trainer setup, bind the Stage 4
legacy exception to the approved root artifact rather than only its basename,
correct the inverted-source and seed-variant tests, and exercise packed
`opt_group` relabeling with sequential TBPTT consumption. No packed self-play
or RL run is allowed before these gates are closed.

## Next control point

After AR-005 review, inspect the real self-play/RL boundary and run the smallest
valid on-policy trajectory/log-probability probe, initially from the frozen
Stage 4 root and default Parquet backend unless packed is explicitly promoted.
