# AR-008 reviewer report

## Decision

**REWORK BEFORE POLICY UPDATE.** No P0 was found. AR-008 is valid smoke
evidence for the trajectory contract, but it is not update-ready or
promotion-grade evidence.

## Findings

- **P1:** the committed JSONL stores action, behavior logprob, value, reward,
  and digests, but not observations, real masks, or raw state. It cannot
  support an offline PPO/GRPO update because current-policy logprobs cannot be
  recomputed. The next update must run in the same process or persist the full
  model input/mask state.
- **P1:** the loader computes the Stage 4 hash after loading but does not
  enforce the expected frozen-root hash as a precondition. The root artifact is
  intentionally external/untracked, so the next probe must bind to its known
  SHA-256 before policy use.
- **P2:** a conflicting `archive_date` is accepted when `current.date` matches
  the explicit date; both date fields must be checked.
- **P2:** legality is runtime-enforced but only a mask digest is persisted,
  preventing independent artifact audit.
- **P2:** `deck_sha256` names a parsed-content digest, not the file-byte digest;
  rename or document the distinction.

## Confirmed

Strict FP32 Stage 4 loading, explicit metadata-date behavior, real CabtEnv
interaction, ordered multi-select substeps, recurrent memory boundaries,
terminal rewards, mirror labeling, Parquet provenance-only handling, and the
day-31 fail-closed path all behave as reported. Six focused tests pass.

## Required next work

Bind the expected Stage 4 root hash, validate both date fields, persist enough
observation/mask state or keep it in-process, and then collect a larger
current-policy sample before any PPO/group-relative update or tournament claim.
