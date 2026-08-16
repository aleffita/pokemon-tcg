# AR-027 - grouped dynamic-K sibling-fiber GRPO

Captured on 2026-08-16T19:02:25.577757+00:00 from frozen Stage 4 root `b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b`.

## Result

The collector created multiple exact recurrent sibling bases per matchup.
Each base selected its own effective K from the legal action set, each matchup
was normalized independently, and all groups were combined in one FP32
policy-only optimizer step with terminal credit through discounted future
logical decisions. The frozen root remains the fallback pending tournament.

| Metric | Result |
| --- | ---: |
| Groups / fibers | 16 / 39 |
| Effective K per base | `[4, 4, 2, 2, 2, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2]` |
| Branch policy/uniform mixture | `policy_uniform_mixture` / 0.5 |
| Logical decisions / substeps | 2427 / 2545 |
| Collection seconds / decisions/s | 22.888142 / 106.03744089183888 |
| One grouped optimizer step | 1 |
| Update seconds | 9.161688708001748 |
| Loss / gradient norm | -0.03534653105761369 / 0.6900275349617004 |
| Candidate parameter L2 delta | 0.009215305122152694 |

## Contracts checked

- Every sibling group has one exact simulator snapshot, distinct legal branch
  actions, common branch provenance, and independent recurrent lanes.
- Effective K is dynamic per base: `min(K_max, legal branch actions)`.
- Deck and matchup strata normalize returns independently; no group is centered
  against another matchup's terminal distribution.
- The candidate uses one optimizer step over all groups, while each group's
  sibling-relative credit remains separate.
- All rollouts run to terminal completion and continuation credit uses discount
  `0.97` without duplicating conditional substeps.
- Candidate preflight passed: `True`.

## Limitations and next gate

This is retained as diagnostic output only, not as a strength estimate. The
execution output did not establish a clean provenance boundary relative to the
optimization commit. Do not interpret or promote this candidate; use
AR-027-retry as the authoritative replacement for the same hypothesis.

## Provenance

- Code commit at execution: `509b948bee7fb1b43503a900257268d9bd14f848`
- Candidate SHA-256: `e098fa1dc6747d63e85e25f4cb908001128145fe6e4b05c9aed9fa73470a5fef`
- Sample manifest SHA-256: `667999495f2c36e589728f6b943df8118557b9d557212ef480db79465eb90428`
- Trajectory bundle SHA-256: `b1cb01a0391eed15d8fbf7cbb08d7e7d000b7b45b6d2ccafd1a76ffdd037802e`
- Tournament gate: superseded before interpretation; see
  `experiments/autoresearch/AR-027-retry/`.
