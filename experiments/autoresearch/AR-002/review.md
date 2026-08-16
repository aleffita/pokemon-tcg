# AR-002 reviewer report

Captured 2026-08-16. Decision: **REWORK before multi-day promotion**, not discard.

## Findings

- **P1:** the probe's val/train layout is correct, but the backend does not
  independently validate row-level `(episode_id, side, step_id, decision_id,
  substep)` order. A future interleaved source could shift data relative to
  the trainer's prefix/suffix split.
- **P1:** parity is strong for the probe's 86 columns and records zero
  mismatches, but it derives its required list from the manifest. Add an
  independent trainer-required column contract and an integration test through
  `_load_temporal_batch`/TBPTT.
- **P2:** the clean candidate load log reports 285,835,264 bytes (about 272.6
  MiB), while `load_benchmark.json` combines processes and reports a 7.77 GiB
  high-water mark. The latter is not candidate-specific and must not be used
  for RSS. `psutil.swap_memory()` is global cumulative state, not per-run
  swap. The clean logs are canonical. The report also labels 7,984,250,880
  bytes as 7.98 GiB, although that is 7.44 GiB binary or 7.98 GB decimal.
- **P2:** `parity.py` diverges from `split_episode_ids()` for `max_rows=0`;
  the probe does not exercise this. The manifest records a seed that does not
  currently affect the split.

## Evidence and decision

The data-path gain is causal-compatible and data-identical on the fixed probe:
2.80 GB to 85.2 MB decoded bytes and 1.866 s to 0.162 s clean load, with 86
columns and zero mismatches. The 21.51 s to 20.82 s training difference is
only a one-epoch observation. One epoch including the 1.60 s ETL is 22.42 s
candidate versus 21.51 s baseline; 2/3 epoch values are projections, not
measurements. Keep the backend opt-in, but rework its contract before AR-003.

## Required AR-003 gates

1. Store explicit val rows followed by train rows or store explicit split
   indices, and validate row-level order against the source.
2. Derive parity from an independent required-column set for current trainer
   variants, including aux and TBPTT paths.
3. Reuse `split_episode_ids()` for `max_rows=0` and resolve the seed contract.
4. Reject unsupported multi-source/TBPTT combinations explicitly.
5. Run baseline, ETL and candidate in separate processes, with per-process RSS
   and no global swap counters as per-run metrics.
6. Measure 1, 2 and 3 epochs while reusing one packed store and accounting for
   ETL exactly once.
