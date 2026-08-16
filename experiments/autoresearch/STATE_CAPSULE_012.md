# State Capsule 012 - AR-010 reviewed

Captured: 2026-08-16 after the AR-010 reviewer gate.

## Decision

Stage 4 remains the frozen competitive root and only promoted policy. AR-010
improves first-decision compatibility and provenance checks but is **rework**:
artifact paths can escape the candidate directory and no regenerated candidate
package exists under the new contract.

## Evidence

- Commit: `c6c5695`.
- Focused suite: 15 passed.
- Review: `experiments/autoresearch/AR-010/review.md`.
- The old AR-009 candidate is deliberately rejected by the new validator.

## Open gates

AR-011 must enforce candidate-directory containment and reject traversal,
absolute, and symlink artifacts; then regenerate candidate, manifest, bundle,
report, and logs under AR-010. No tournament may run before this gate.
