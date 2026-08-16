# State Capsule AR-028 multi-deck screen

Captured 2026-08-16.

## Current state

- Frozen Stage 4 root remains the production fallback:
  `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.
- Learner strata: `deck_supreme_60.json` (v0) and
  `candidates/deck_v2_anti_control_lock.json` (v2).
- Opponent strata: `lb1009`, `lb945`, `lb826`, and `lb814`, each with its
  shipped deck and external policy.
- Training candidate:
  `experiments/autoresearch/AR-028-multideck-screen2/candidate.pt`.
- Candidate SHA-256:
  `bc07eb8507b86bdadebba1608681335d8dfc48cea5462fc547f725ad1f236300`.

## Collection and update

- 8 independent deck/opponent groups, dynamic K `[2,4,2,2,2,4,3,2]`.
- 21 fibers, 1,285 logical decisions, 1,348 substeps.
- Collection: 11.668447 s, 110.126 decisions/s.
- Update: 4.707561 s, one FP32 grouped optimizer step.
- 7/8 groups were zero-variance; 189 logical actions received credit.
- Root/sample/bundle provenance and candidate preflight passed.

## Tournament gate

Corrected model-loading panel, 5 games per opponent and two learner decks:

- v0: `5-25-0`;
- v2: `5-25-0`;
- combined: `10-50-0` (`16.667%`).
- Lucario (`lb1009` + `lb945`): `0-10` for each learner deck.
- Alakazam (`lb826`): `0-10` for each learner deck.
- Crustle (`lb814`): `5-5` combined.

The candidate is rejected for promotion. The earlier report generated before
checkpoint-path correction loaded the baseline and is invalid for candidate
claims. The root fallback is unchanged.

Evidence:

- `experiments/autoresearch/AR-028-multideck-screen2/manifest.json`
- `experiments/autoresearch/AR-028-multideck-screen2/metrics.json`
- `experiments/autoresearch/AR-028-multideck-screen2/sample.manifest.json`
- `experiments/autoresearch/AR-028-multideck-screen2/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-028-multideck-screen2/candidate.pt`
- `experiments/decks/AR-028-multideck-screen2-candidate-correct-model-panel-5.json`

## Next control point

Run the valid v3 Mimikyu Safeguard deck against the same six-opponent surface,
then decide whether to retrain on v3 or return to a focused Lucario counter.
