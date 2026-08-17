# Codex -> Antigravity: AR-028 multi-deck handoff

Captured: 2026-08-16
Owner of architecture, training, simulator, tournament execution, and compute: Codex.
Owner of tactical deck analysis and matchup hypotheses: Antigravity.

## Stable file contract

Read-only Codex input for the swarm:

- this file: `.agents/orchestrator/CODEX_TO_ANTIGRAVITY_AR028_MULTIDECK.md`;
- measured panel reports under `experiments/decks/AR-028*.json`;
- `experiments/decks/DIAGNOSIS_AR028_PANEL.md`;
- the latest completed multi-deck artifact directory under
  `experiments/autoresearch/AR-028-multideck/`.

Swarm outputs for Codex consumption:

- complete candidate decks only in `experiments/decks/candidates/`;
- one diagnostic or matchup report per iteration in
  `experiments/decks/diagnostics/` (create the directory if needed);
- do not overwrite prior candidates or alter `agent/deck.csv`/`agent/deck.json`.

Codex will consume files when present. Antigravity may use its own harness-native
cron/file monitor, as already authorized by the operator, but must not launch
training, MPS/Metal work, tournaments, or database writes.

## Measured evidence to use

The first explicit deck panel was weak in absolute strength:

| learner deck | random | first | lb1009 | lb945 | lb826 | lb814 | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0 Deck Supreme 60 | 7-3 | 2-8 | 0-10 | 0-10 | 0-10 | 4-6 | 13-47 |
| v1 anti-Lucario tempo | 2-3 | 0-5 | 0-5 | 1-4 | 0-5 | 0-5 | 3-27 |
| v2 anti-control lock | 0-5 | 2-3 | 0-5 | 0-5 | 1-4 | 3-2 | 6-24 |

The actionable bottleneck is the two Lucario policies (`lb1009` and `lb945`),
not aggregate setup probability. v2 showed a small diagnostic signal against
Crustle and Alakazam, but did not solve Lucario. The repaired v3 is now an exact
60-card candidate with the documented Mimikyu Safeguard hypothesis:

`experiments/decks/candidates/deck_v3_apex_sovereign.json`

It is eligible for Codex screening, but is not yet a promoted deck.

## Requested tactical work

1. Inspect the AR-028 panel and the v3 rationale against actual card/database
   evidence.
2. Concentrate on a robust Lucario line while preserving the v2 control signal.
3. Produce a new exact-60 candidate only when the delta is source-backed and
   the rationale states the expected matchup mechanism.
4. Validate quantity totals and card IDs before emitting the JSON.
5. Put new candidates in `experiments/decks/candidates/` and diagnostics in
   `experiments/decks/diagnostics/`; leave prior files immutable.

Do not prescribe GRPO, RoPE-ND, simulator, tournament, or model architecture
changes from the swarm. Those remain Codex decisions. Return hypotheses and
deck artifacts through the file contract so the compute lane stays unblocked.

## Compute-lane status

Codex attempted AR-028 multi-deck collection with v0 and v2 crossed against
four external policy/deck strata. Collection wrote a 30 MB sample manifest and
16 MB trajectory bundle, but no `candidate.pt`, `manifest.json`, or
`metrics.json` was produced. This is an incomplete run and must not be used as
training or promotion evidence. Codex is repairing the bounded update/compute
path and will rerun it; the swarm should continue tactical analysis in the
meantime without waiting for a chat response.

## Latest valid result

The repaired multi-deck screen completed successfully:

- v0 + v2 learner strata crossed against `lb1009`, `lb945`, `lb826`, and
  `lb814` external policy/deck strata;
- 8 groups, dynamic K `[2,4,2,2,2,4,3,2]`, 21 fibers;
- 1,285 logical decisions at 110.126 decisions/s;
- 4.708 s grouped update, 7/8 zero-variance groups, 189 credited actions;
- candidate SHA `bc07eb8507b86bdadebba1608681335d8dfc48cea5462fc547f725ad1f236300`.

The corrected candidate tournament was `10-50-0` over 60 games:

- v0 `5-25-0`, v2 `5-25-0`;
- Lucario policies combined `0-20`;
- Alakazam combined `0-10`;
- Crustle combined `5-5`.

This candidate is rejected for promotion. The operational conclusion is a
specific Lucario bottleneck, not a general setup failure. Inspect the valid
v3 Mimikyu Safeguard hypothesis against this result and emit a new exact-60
candidate only if the matchup mechanism is source-backed. Codex owns whether
and how that deck is used for GRPO; do not redirect the architecture lane.

The v3 baseline tactical screen is now complete: `3-27-0` over 30 games
(`random 2-3`, `first 0-5`, `lb1009 0-5`, `lb945 0-5`, `lb826 0-5`,
`lb814 1-4`). It did not change the Lucario bottleneck and is rejected as a
deck promotion. Continue investigating a source-backed Lucario counter or a
new exact-60 revision; do not assume v3 should become the GRPO learner deck.

Codex also screened the existing adaptive inference modes on v0/v2 against
the four external policy decks: b1 K=3 scored `7-25-0`, b2 K=3 epsilon 0.05
scored `6-26-0`, and both were `0-16` against Lucario. These are diagnostics,
not promoted paths. Continue deck analysis from the concrete Lucario failure;
Codex is now using v3 + v2 as learner strata in a targeted Lucario GRPO run.
