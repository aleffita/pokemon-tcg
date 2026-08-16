# AR-013 review

## Verdict

**KEEP**. The AR-013 changes close the AR-012 path and SHA-validation findings
within the requested scope. No tournament, package, submission, or external
side effect was executed.

## Evidence

- Repository identity: `/Users/alefita/workdir/pokemon-tcg`, branch `develop`.
- Audited commit: `03e8efe` (`fix(autoresearch): close AR-012 tournament path findings`).
- Focused validation: `uv run --locked pytest -q tests/test_ar010_candidate_path.py tests/test_trajectory_probe.py`
  completed with **26 passed in 1.93s**.
- `git diff --check 03e8efe^ 03e8efe` completed with exit code 0.
- The five receipt-referenced artifacts are ordinary Git blobs (`100644`), not
  LFS pointers, and their working-tree SHA-256 values equal both the receipt
  values and the bytes read from commit `03e8efe`:

  | Artifact | SHA-256 | Receipt/commit match |
  | --- | --- | --- |
  | `experiments/autoresearch/AR-010/candidate.pt` | `e6efe207d4b08dd458b40be14297b142ca2987b2238f97d807b6bf85320c7773` | yes |
  | `experiments/autoresearch/AR-010/sample.manifest.json` | `b5c273f75c8bee147a16fbde49b8ca5fed0c017fa656b5f29cf1d480db051256` | yes |
  | `experiments/autoresearch/AR-010/trajectory_bundle.pt.gz` | `f2f6ca653d752d91fc17b61298c3dfa09aac1ad5741f0d4c6b71ba065aedbbbf` | yes |
  | `experiments/autoresearch/root/stage4_root.pkl` | `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b` | yes |
  | `experiments/autoresearch/root/stage4_root_fp32.tar.gz` | `32add97ad0848cc097a983a45a75a935532a807645b9e36d972bd6fee1c49751` | yes |

- The manifest's canonical content digest, computed by the repository
  validator after removing its self-referential `sha256` field, is
  `915d7314f6356695f38b1b183c54efbb3bd8141ee28f89f981bc8722ccbdad16`, equal
  to the receipt and manifest field.
- Stage 4 remains frozen by the approved pickle and FP32-package digests. The
  same pickle digest is also the runtime `APPROVED_STAGE4_ROOT_SHA256` and is
  repeated in the earlier capsules and trajectory artifacts. The root files
  were first tracked by AR-013, so this is a digest-continuity claim, not a
  claim that a predecessor Git blob was independently byte-compared.

## Opt-in and default behavior

- `public_agents/submissions/latest-submission-300elo/main.py` rejects a
  relative `PTCG_MODEL_PATH` at lines 53-57, before the repository imports and
  before candidate provenance or model loading at lines 93-110 and 139.
- The rejection is conditional on `PTCG_MODEL_PATH` being set. With it unset,
  the existing candidate-search/default path remains active; the default-path
  test also confirms that candidate provenance is not invoked.
- With opt-in enabled, missing, malformed, mismatched, or tensor-tampered
  candidate bytes are fail-closed by the expected SHA-256 preflight. The
  focused suite also executes a real first `choose()` model decision.

## P1/P2 findings

- **P1:** none observed within this review scope.
- **P2:** none blocking. Reproducibility is intentionally bounded: the commit
  makes the candidate, manifest, bundle, frozen Stage 4 files, validation code,
  and receipt reproducible. AR-010 logs and review artifacts remain separate
  working-tree artifacts, and no claim is made that a tournament or packaged
  submission has been reproduced; the receipt explicitly says
  `tournament_status=not_run`.

## Limits and decision

This review did not run the full test suite, tournament, packaging, or
submission, as required. KEEP applies to the AR-013 integrity/path hardening;
it is not a competitive promotion of the PPO candidate and does not alter the
Stage 4 fallback.
