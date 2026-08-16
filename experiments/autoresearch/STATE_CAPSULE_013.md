# State Capsule 013 - AR-011 reviewed

Captured: 2026-08-16 after the AR-011 reviewer gate.

## Decision

Stage 4 remains the frozen competitive root and only promoted policy. The
regenerated AR-010 candidate is technically runnable, but tournament is still
blocked until candidate-byte integrity is enforced by an explicit expected
SHA receipt and the exact binary hashes are recorded.

## Evidence

- Commit: `8bc38e5`.
- Focused suite: 21 passed.
- Candidate SHA:
  `e6efe207d4b08dd458b40be14297b142ca2987b2238f97d807b6bf85320c7773`.
- Manifest SHA:
  `b5c273f75c8bee147a16fbde49b8ca5fed0c017fa656b5f29cf1d480db051256`.
- Bundle SHA:
  `f2f6ca653d752d91fc17b61298c3dfa09aac1ad5741f0d4c6b71ba065aedbbbf`.
- Review: `experiments/autoresearch/AR-011/review.md`.

## Open gate

AR-012 must enforce the expected candidate SHA only when the opt-in path is
used, then run a repository-local absolute-path no-sweep tournament. Do not
package or submit this candidate.
