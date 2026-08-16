# State Capsule 014 - AR-013 accepted

Captured: 2026-08-16 after the AR-013 reviewer gate.

## Decision

Stage 4 remains the frozen competitive root and fallback. The AR-010 PPO
candidate is now a reproducible repository-local experiment artifact, but it
is not promoted competitively until it earns that status in tournament ground
truth.

## Durable evidence

- AR-013 commit: `03e8efe`.
- Reviewer: `experiments/autoresearch/AR-013/review.md`, verdict KEEP.
- Focused validation: 26 tests passed; `py_compile` and `git diff --check`
  passed.
- Candidate SHA:
  `e6efe207d4b08dd458b40be14297b142ca2987b2238f97d807b6bf85320c7773`.
- Manifest SHA:
  `b5c273f75c8bee147a16fbde49b8ca5fed0c017fa656b5f29cf1d480db051256`.
- Bundle SHA:
  `f2f6ca653d752d91fc17b61298c3dfa09aac1ad5741f0d4c6b71ba065aedbbbf`.
- Frozen Stage 4 pickle SHA:
  `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- Frozen Stage 4 FP32 package SHA:
  `32add97ad0848cc097a983a45a75a935532a807645b9e36d972bd6fee1c49751`.
- All five receipt-referenced binary artifacts are tracked in Git and were
  verified byte-for-byte against the receipt.

## Runtime gate

- `PTCG_MODEL_PATH` is explicit opt-in and must be absolute.
- `PTCG_EXPECTED_MODEL_SHA256` is required for that opt-in path and is checked
  against candidate bytes before provenance validation and model loading.
- Missing, malformed, mismatched, or tensor-tampered candidate bytes fail
  closed.
- Unset `PTCG_MODEL_PATH` preserves the default public-agent path.

## Open control point

Run only a repository-local tournament with absolute paths, explicit expected
candidate SHA, `--no-sweep`, and no packaging/submission. Compare the candidate
against root-relevant baselines before any keep/revert decision. The frozen
Stage 4 root remains the fallback regardless of the result.
