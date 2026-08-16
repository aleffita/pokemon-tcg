# AR-010 reviewer report

## Decision

**REWORK BEFORE TOURNAMENT.** No P0 was found. The first-decision path and
provenance checks are improved, but detached artifact paths remain possible.

## Findings

- **P1:** relative `..` paths and absolute paths are accepted without
  containment checks. A manifest and bundle placed outside the candidate
  directory were accepted by `validate_candidate_provenance`, so the candidate
  package is not yet self-contained.
- **P1:** no AR-010 regenerated candidate, bundle, manifest, report, or test log
  exists. The old AR-009 candidate is intentionally incompatible with the new
  validator because it lacks bundle/content provenance and complete input
  digests.
- **P2:** adversarial tests do not yet cover traversal, absolute paths,
  symlinks, manifest-content tampering, bundle/manifest mismatch, or default
  behavior regression. The fixture uses synthetic rather than real encoder
  inputs.

## Confirmed

The first opt-in `choose()` succeeds, root validation precedes model loading,
default public behavior remains unchanged, complete input-digest logic exists,
and the focused 15-test suite reproduces.

## Required next work

Reject any provenance artifact path that is not a non-symlink regular file
contained in the candidate directory, add adversarial tests, regenerate the
candidate and all linked artifacts under AR-010, and save a complete report and
log before tournament use.
