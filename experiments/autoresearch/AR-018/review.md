# AR-018 reviewer audit

## Initial audit

The first review found one P0: `CabtEnv.reset()` could retry after a
discarded opening in which the mirror had already acted, while the collector
reset the mirror only once before calling `env.reset()`. The review also
requested four-game coverage, an end-to-end learner/behavior logprob check,
and independently recorded mirror terminal evidence.

## Rework and final verdict

**KEEP.** The P0 is closed by `28c2b96`. `CabtEnv` now accepts an explicit
reset hook and invokes it before every battle-start attempt. The stateful
mirror is wired to that hook, so memory, event records, and terminal state are
cleared before every retry. `tests/test_trajectory_probe.py` includes a fake
two-attempt environment proving that the discarded attempt invokes the mirror
under the first reset state and the accepted attempt starts under a fresh
reset state.

The former P1 findings are also closed for this gate:

- the final smoke ran four games and covered both agent sides, `[1, 0, 0, 1]`;
- a copied learner snapshot recomputes each complete logical-action logprob
  from the recorded actions and masked logits, with an importance ratio of one
  against the behavior snapshot;
- `CabtEnv` notifies the mirror of terminal outcomes, and mirror event records
  carry the terminal flag and mirror-perspective reward. The manifest records
  one or more terminal mirror records per game.

## Checks

- `uv run --locked pytest -q tests/test_trajectory_probe.py tests/test_ar010_candidate_path.py`: `32 passed`.
- `py_compile` for the environment, collector, executable probe, and tests: passed.
- `git diff --check`: passed.
- Four-game metadata-bound smoke: passed; 672 logical decisions, 759 records,
  118.369 records/s, 104.801 decisions/s.
- All four games had independent agent and mirror continuity chains, legal
  action checks, composite logprob checks, and symmetric terminal returns.
- Manifest and self-play JSONL SHA-256 values match the recorded files.
- No GRPO, RoPE-ND, tournament, package, ETL, Parquet, or packed-data path was
  run.

No P0 or P1 remains. This is a correctness/throughput gate, not competitive
evidence; Stage 4 remains the only promoted policy and fallback. The next
control point is trajectory-group GRPO.
