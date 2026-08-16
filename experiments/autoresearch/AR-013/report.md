# AR-013 report

## Scope

AR-013 closes the remaining AR-012 review findings before any tournament. The
opt-in public-agent path now rejects a relative `PTCG_MODEL_PATH` immediately
after reading the environment variable, before candidate provenance or model
loading. The default path remains unchanged when `PTCG_MODEL_PATH` is unset.

The expected candidate SHA contract also has a dedicated malformed-value
regression test and remains fail-closed.

## Artifact durability

The approved bytes were not altered. The following receipt-referenced files
are now tracked in Git by the AR-013 commit:

- `experiments/autoresearch/AR-010/candidate.pt`
- `experiments/autoresearch/AR-010/sample.manifest.json`
- `experiments/autoresearch/AR-010/trajectory_bundle.pt.gz`
- `experiments/autoresearch/root/stage4_root.pkl`
- `experiments/autoresearch/root/stage4_root_fp32.tar.gz`

Their SHA-256 values remain those recorded in
`experiments/autoresearch/AR-010/AR-012-candidate-receipt.txt`. Git regular
objects were used because the five files are approximately 21 MB in total,
each is below the repository's normal single-file hosting limit, and no
repository policy excludes these paths. No Git LFS pointer or regenerated
binary was introduced.

## Validation

The focused candidate-path and trajectory-probe suites, Python compilation,
and whitespace checks are recorded in `logs/tests.log`. No tournament,
packaging, or submission was run.
