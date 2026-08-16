# AR-011 reviewer report

## Decision

**REWORK BEFORE TOURNAMENT.** No P0 was found. The regenerated package passes
provenance, strict loading, and first-decision checks, but candidate weight
integrity is not enforced by the loader.

## Findings

- **P1:** changing a tensor in `candidate.pt` while preserving embedded
  provenance still passes validation and strict loading. The externally
  reported candidate SHA is not enforced.
- **P1:** candidate, manifest, bundle, and logs are untracked, so a clean
  checkout of the reviewed commit cannot reproduce the exact package.
- **P1 conditional:** the opt-in path is repository-local and imports root
  `scripts`/`rl`; it must not be used as a packaged Kaggle submission.
- **P2:** direct adversarial checks pass for non-regular, non-canonical,
  symlink, and candidate-link cases, but dedicated tests are sparse.
- **P2:** tournament use requires an absolute model path; the relative path
  form fails after tournament changes the agent cwd.

## Confirmed

The 21-test suite reproduces; all regenerated hashes and bundle/manifest
linkage match; old AR-009 is rejected; first `choose()` succeeds; and default
public behavior remains unchanged.

## Required next work

Enforce an expected candidate SHA via an explicit opt-in receipt/preflight,
record the exact binary artifact hashes, and run only a repository-local,
absolute-path, no-sweep tournament without packaging or submission.
