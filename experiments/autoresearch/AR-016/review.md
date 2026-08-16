# AR-016 review

## Verdict

**KEEP**

Commit `0445190ecef72cbdb1e0d6330e11d2b7dc46d001` is a bounded collection-
throughput change. It does not alter model weights, architecture, Stage 4 root
selection, opponent semantics, recurrence handling, or tournament code.

## Audit evidence

- Repository identity: project root `pokemon-tcg`, branch `develop`,
  remote `origin=https://github.com/aleffita/pokemon-tcg.git`.
- The commit diff contains only `MAX_GAMES_PER_MODE = 4`, the corresponding
  `run_probe` validation, CLI choices/help, documentation, and focused tests.
- `run_probe` rejects values outside 1-4 before metadata validation, deck/data
  access, card-table construction, or Stage 4 model loading. The parser accepts
  1, 2, 3, and 4 and rejects 5 before probe execution.
- Existing value `2` is explicitly preserved by the parser test; value `4` is
  accepted; value `5` is rejected by both the parser and direct `run_probe`
  validation.
- The loop remains the same two modes, `random` and `mirror_no_memory`, with
  the unchanged seed expressions `seed + mode_index * 100 + game_index` for
  both the environment and policy action generator. The frozen root hash and
  all existing trajectory invariants are unchanged in the commit diff.
- Focused validation passed:

  ```text
  uv run --locked pytest tests/test_trajectory_probe.py tests/test_ar010_candidate_path.py
  28 passed in 1.85s
  ```

- `git diff --check` passed for the working tree and for the commit diff.
- No tournament, package, submission, or binary regeneration was run.

## AR-015 artifact preservation

No `experiments/autoresearch/AR-015/` path is present in the `0445190` diff.
The current AR-015 provenance is internally consistent with the recorded
hashes:

| Artifact | SHA-256 |
| --- | --- |
| `candidate.pt` | `62ea62eb7ee5f1baca1f50a5b9713ef83b96dd2f2d0ea707151739058d235ff6` |
| `trajectory_bundle.pt.gz` | `2a8926080eba535a57360fd8b6f2d934cd2a293bc10de40c4f210f768f0aff97` |

The AR-015 trajectory log and manifest report the same candidate and bundle
hashes, and the working tree shows AR-015 only as an existing untracked
artifact directory; this review made no change to it.

## Scope boundary

This is a code and provenance audit only. It does not claim that increasing the
limit improves competitive strength; that requires a later trajectory run and
tournament gate. No commit was created by this review.
