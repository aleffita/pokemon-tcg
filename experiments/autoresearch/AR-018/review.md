# AR-018 review - KEEP for the next RL task

## Verdict

**KEEP the implementation as the true recurrent self-play foundation.** The
correctness gate passed and the result is interpretable as a collector smoke.
It is not a competitive promotion and has no tournament evidence.

## Acceptance audit

- The mirror initializes one learned Stage 4 memory per episode.
- Every mirror substep receives a non-null persistent `memory_in`; only the
  final substep output is committed.
- Agent and mirror lanes have independent continuity chains and reset digests.
- `CabtEnv` side-specific tracker, ability, and deck arguments remain the
  encoder context for the mirror.
- Legal action indices and legality checks are recorded for both lanes.
- Substep logprobs remain present and the complete logical decision sum is
  recorded under `logical_action_logprob` and `decision_logprob`.
- The real one-game metadata-bound smoke completed with symmetric terminal
  return signs and no Parquet or packed hot-path use.

## Non-blocking limitations

- The probe is intentionally one game and not a strength evaluation.
- The event JSONL is compact relative to tensor buffers but is still a
  per-substep diagnostic log.
- No batched collection or recurrent gradient update exists yet.

Evidence: `report.md`, `manifest.json`, `logs/selfplay.jsonl`, and
`logs/tests.log`.
