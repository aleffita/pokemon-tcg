# Codex and Antigravity Continuous Deck Protocol

Status: active until the competition submission is frozen.

This file is the filesystem contract. Antigravity does not wait for Codex,
request confirmation, start training, or modify active autoresearch code.
Codex does not wait for Antigravity. Each GRPO/tournament run consumes one
immutable snapshot of the available deck pool at process start.

## Ownership

- Antigravity owns `experiments/decks/swarm/inbox/` and
  `experiments/decks/swarm/archive/`.
- Codex owns `experiments/decks/swarm/results/` and every
  `experiments/autoresearch/AR-*/` directory.
- `model/results.db` is read-only for Antigravity. It must not hold a write
  transaction or launch tournament, PyTorch, MLX, Metal, MPS, or GPU work.
- Neither side edits or deletes files owned by the other side.

## Antigravity output: candidate deck inbox

Publish each live candidate as:

```text
experiments/decks/swarm/inbox/NNN_short_name.json
```

The file must be a JSON array containing exactly 60 integer card IDs and must
pass `tests/test_deck_m1_validation.py`. Lower numeric prefixes have higher
priority. Use a temporary filename that does not end in `.json`, then rename
it atomically only after validation. Never rewrite a published `.json` in
place.

Keep only the strongest current candidates in `inbox/`. Antigravity may remove
or replace superseded inbox candidates after a newer result snapshot exists.
Move historically useful candidates to `archive/`; Codex copies every consumed
deck into its AR run directory, so removal from the live inbox cannot detach
completed-run provenance.

Do not write monographs or analysis into `inbox/`. Optional analysis belongs in
`archive/NNN_short_name.md` and must use the same stem as its deck.

## Codex output: machine-readable competitive feedback

After every GRPO collection or tournament, Codex writes:

```text
experiments/decks/swarm/results/latest.json
experiments/decks/swarm/results/AR-NNN.json
```

`latest.json` contains:

- run and candidate checkpoint identity;
- the exact consumed deck paths and SHA-256 hashes;
- per-deck, per-opponent, per-seed returns;
- paired inter-deck cohort advantages;
- sibling variance and zero-signal coverage;
- collection/update throughput;
- tournament metrics when available.

The runner may also emit its own training-time turn-zero deck at
`experiments/autoresearch/AR-NNN/decks/generated_turn0.json`. This is the same
model's free deck action, not an Antigravity artifact. It enters the matchup
matrix like every other learner deck. Antigravity should analyze its results
but must not overwrite it; improved human/swarm responses still go to `inbox/`.

At competition inference, `deck.csv` is teacher-forced through the model's
recurrent turn zero before gameplay. If no deck is supplied in a research
harness, the model emits a legal 60-card deck from that same turn-zero head.

Antigravity's five-minute scheduler reads `latest.json` first. If its run ID or
content hash has not changed, it performs no write. If it changed, Antigravity
may additionally inspect the matching `AR-NNN.json`, the referenced AR manifest,
and `model/results.db` read-only, then revise the inbox against observed
weaknesses. It must distinguish GRPO rollout return from tournament win rate.

## Continuous loop

1. Antigravity reads the newest result snapshot and identifies deck-specific
   weaknesses by opponent, opening consistency, prize trade, and legal setup.
2. Antigravity validates and atomically publishes zero or more improved 60-card
   candidates to `inbox/`, then removes or archives superseded inbox entries.
3. The next Codex run automatically snapshots the highest-priority inbox decks,
   combines them with its control decks, and evaluates/trains across paired
   opponent seeds.
4. Codex writes the new compact result snapshot. The loop repeats without any
   message, acknowledgement, scheduler, or blocking action on the Codex side.

## Resource and safety boundaries

- Antigravity's scheduler may run every five minutes, but its cycle must remain
  CPU-light and filesystem/SQLite-read-only outside its owned directories.
- Do not modify a deck while a `.json` publish is visible. Atomic rename is the
  publication boundary.
- Do not place more than eight live candidates in `inbox/`; archive the rest.
- Do not copy a result file back into the inbox and do not infer model promotion
  from training loss alone. Tournament evidence remains the competitive gate.
