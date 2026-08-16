# Autoresearch candidate provenance contract

The `PTCG_MODEL_PATH` environment variable is an explicit local opt-in. When
it is set, `public_agents/submissions/latest-submission-300elo/main.py`
requires `PTCG_EXPECTED_MODEL_SHA256`, hashes the candidate bytes, and rejects
missing or mismatched expected bytes before validating candidate evidence or
invoking the strict FP32 inference loader. When `PTCG_MODEL_PATH` is unset, the
public agent keeps its existing submission checkpoint search and behavior;
`PTCG_EXPECTED_MODEL_SHA256` is ignored.

An AR candidate payload must contain an `autoresearch` object with:

- `root_sha256`, equal to the approved frozen Stage 4 root SHA-256;
- `sample_manifest_sha256`, the SHA-256 of the sample-manifest file bytes;
- `sample_manifest_content_sha256`, the manifest's canonical-content digest;
- `bundle_sha256`, the SHA-256 of the compressed trajectory-bundle file; and
- `artifacts.sample_manifest` and `artifacts.trajectory_bundle`, set to the
  canonical adjacent filenames `sample.manifest.json` and
  `trajectory_bundle.pt.gz`.

The candidate, manifest, and compressed bundle must be regular files rather
than symlinks. The manifest and compressed bundle must be adjacent to the
candidate, match their recorded file hashes, carry the approved root hash, and
contain identical manifest data. Absolute paths, traversal components, and
other artifact filenames are rejected before any artifact is opened.
The bundle is checked in collection order, including the action and recurrent
state digests plus an ordered digest entry for every model-input tensor. A
changed, missing, stale, or detached artifact is rejected loudly before the
candidate model is loaded.

Deck provenance uses separate names: `deck_content_sha256` hashes the
normalized parsed 60-card list, while `deck_source_file_sha256` hashes the
source file bytes. They are different claims and must not be substituted for
one another.

The root inference contract supplies a disabled `prospective_planner` object
when an older checkpoint omits that field. This preserves the old model path
while allowing the public agent's first `choose()` decision to execute safely.
