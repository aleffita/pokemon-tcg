# AR-003 reviewer report

Captured 2026-08-16. Decision: **keep opt-in / inconclusive**. Preserve
`245ae42`; do not promote packed data yet.

## P1 findings

- The builder and independent parity produce val-first/train-second and
  validate the five-field row key, but the trainer opens only
  `episode_id`, `side` and `step_id` for its runtime metadata. Full
  `decision_id`/`substep` order validation and the val/train boundary are not
  fail-closed in the actual trainer path.
- Checkpoint resume validates architecture and optimizer/scheduler state but
  does not bind the resume to source digest, selection, split, packed digest or
  backend. A later resume with a different corpus could proceed silently.

## P2 findings

- Parity passed 86/86 columns, row order, dtypes, values, labels, masks and
  auxiliary fields with zero mismatches, but the required-column list still
  shares the production helper and there is no independent dedup or full
  `_load_temporal_batch` parity test.
- Five focused tests pass, but adversarial manifest-boundary, source-order,
  dedup, multi-source, resume-identity and packed-without-TBPTT rejection
  cases are not all covered.
- Zero spills is valid for the observed run but not a general pressure proof;
  the baseline benchmark disables the SSD spill directory. Peak MLX memory is
  about 18.9 GiB for both backends and is not RSS.
- The 1/2/3 epoch comparisons are separate process runs reusing one store in
  the candidate scenarios. The ETL cost is charged once per hypothetical
  source-to-ready horizon and must not be summed three times as one campaign.

## Metrics check

The separate load metrics are technically plausible and correctly process
scoped: baseline 9,992,110,080 B RSS and 2,858,248,224 decoded bytes versus
candidate 916,111,360 B RSS and 418,820,760 bytes. The 2-epoch packed result
is faster after ETL, while 1 and 3 epochs are slower; no stable promotion
claim follows.

## Required next gates

1. Make runtime row-order/boundary validation open and verify the complete
   order key.
2. Bind checkpoint resume to source/selection/split/packed/backend identity.
3. Add adversarial manifest, source-order, dedup, multi-source and resume
   tests, then repeat the 2-epoch comparison with process-local training RSS
   and controlled CPU/Metal state.

No model or inference change occurred, so no tournament was required.
