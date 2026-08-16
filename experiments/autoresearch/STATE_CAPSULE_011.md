# State Capsule 011 - AR-009 reviewed

Captured: 2026-08-16 after the AR-009 reviewer gate.

## Decision

Stage 4 remains the frozen competitive root and only promoted policy. AR-009
produced a small PPO candidate, but it is **not tournament-ready** because the
opt-in agent fails on its first real decision and candidate provenance is not
enforced.

## Evidence

- Commit: `a55e7a4`.
- Candidate SHA-256:
  `c23ec42ce559db77894e7accd46e131462a276c1092cafac74ec1c66f1291542`.
- 175 samples, one PPO epoch, root KL `0.0005297892`, parameter L2 delta
  `0.0092003815`, focused tests 10 passed.
- Review: `experiments/autoresearch/AR-009/review.md`.

## Open gates

AR-010 must make the first opt-in decision succeed, enforce root/sample/bundle
provenance, and digest complete model inputs. No tournament may run before
these checks pass.

## Next control point

After AR-010 review, run the candidate versus named random/first/public
opponents with deck-conditioned result rows, then make a keep/revert decision
against the frozen Stage 4 root.
