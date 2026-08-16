# AR-012 reviewer report

## Verdict

**REWORK**

The expected-byte preflight is correctly implemented for the opt-in path, but
the candidate package is not durable in the reviewed commit and the tournament
path still relies on an operator-supplied absolute path that the code does not
enforce. These findings should be closed before treating the candidate as a
reproducible tournament artifact.

## Scope and method

- Reviewed commit `5b890b9` and its parent diff.
- Inspected `main.py`, `scripts/rl/ppo_micro_update.py`, the focused tests,
  the AR-010 candidate metadata, the receipt, and the frozen Stage 4 files.
- Ran `uv run --locked pytest -q tests/test_ar010_candidate_path.py
  tests/test_trajectory_probe.py`: **24 passed**.
- Ran a read-only provenance and digest check against the current local files:
  candidate provenance passed and every receipt-listed digest matched.
- Did not edit production code, run a tournament, package, submit, or commit.

## Verified controls

### Expected SHA preflight

`public_agents/submissions/latest-submission-300elo/main.py` invokes
`validate_expected_model_sha256()` at lines 99-102 when and only when the
non-empty `PTCG_MODEL_PATH` opt-in is present. The helper rejects missing or
blank values, rejects non-64-character/non-hex values, hashes the candidate
bytes, and rejects a mismatch. This call precedes
`validate_candidate_provenance()` and the later
`load_inference_checkpoint()` call at line 135.

The default branch does not import or call the candidate gate. The focused
default-path test sets an intentionally unusable expected-SHA variable and
confirms the candidate provenance gate is not entered.

### Focused test coverage

The tests cover all requested positive and negative behaviors:

- embedded tensor tamper with retained provenance and the original expected
  digest;
- missing expected digest;
- 64-hex mismatch;
- default path bypass;
- real opt-in first `choose()` with a model forward.

The implementation also contains a direct malformed-digest check. There is no
dedicated regression test for a malformed non-empty digest, noted below as a
minor coverage gap.

### Receipt and local artifacts

The current local values match `AR-012-candidate-receipt.txt` exactly:

| Artifact | SHA-256 |
| --- | --- |
| AR-010 candidate | `e6efe207d4b08dd458b40be14297b142ca2987b2238f97d807b6bf85320c7773` |
| sample manifest | `b5c273f75c8bee147a16fbde49b8ca5fed0c017fa656b5f29cf1d480db051256` |
| trajectory bundle | `f2f6ca653d752d91fc17b61298c3dfa09aac1ad5741f0d4c6b71ba065aedbbbf` |
| frozen Stage 4 pickle | `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b` |
| frozen Stage 4 FP32 tar | `32add97ad0848cc097a983a45a75a935532a807645b9e36d972bd6fee1c49751` |

The candidate's embedded root, manifest, bundle, and manifest-content hashes
also validate against those local files. The receipt explicitly records the
Stage 4 root/fallback and the repository-local, absolute-model-path,
`--no-sweep`, no-packaging, and no-submission constraints.

## Findings

### P1 - receipt-referenced candidate and frozen root are absent from commit

The reviewed commit contains the receipt and source changes, but does not
contain `experiments/autoresearch/AR-010/candidate.pt`, its manifest or bundle,
or either frozen Stage 4 root file. `git ls-tree -r --name-only 5b890b9`
confirmed that those five binary artifacts are absent, while the files exist
only as current untracked workspace files. The receipt itself says the binary
artifacts are not committed.

This means a clean checkout of `5b890b9` cannot reproduce or verify the exact
candidate referenced by the receipt, and a later local replacement can leave a
valid-looking committed receipt pointing at a different unavailable byte
sequence. The current files are not stale relative to the receipt, but their
provenance is not durable across checkout or workspace loss. This is the same
artifact-persistence class of risk identified before AR-012 and remains open.

### P2 - absolute candidate path is documented but not enforced in code

The receipt requires an absolute `PTCG_MODEL_PATH`, but `main.py` accepts the
raw environment value and checks it with `os.path.isfile()` without requiring
`Path.is_absolute()` or canonicalizing it. The tournament loader
(`scripts/_common.py:102-108`) temporarily changes the working directory to
the agent directory while importing the public agent. Consequently, a
repository-relative candidate path can fail to resolve in the tournament even
when it was valid from the shell's original directory.

The failure is loud rather than a silent fallback, and the receipt gives a
correct operational workaround: use the absolute path from the receipt and
run repository-local with `--no-sweep`. It is nevertheless a path/cwd footgun
at the exact tournament gate and is not protected by the preflight itself.

### P2 - malformed-digest regression is not directly tested

`validate_expected_model_sha256()` correctly rejects malformed values, but the
AR-012 test additions exercise only missing and 64-hex mismatched values. Add
one focused malformed-value case (for example, a short or non-hex string) to
prevent a future change from weakening that stated fail-closed contract.

## Decision

**REWORK before tournament gate.** The byte-integrity logic and current local
receipt are sound, but the P1 durability issue and P2 path/cwd issue must be
resolved or explicitly accepted by a later reviewer. No tournament result is
claimed by AR-012.
