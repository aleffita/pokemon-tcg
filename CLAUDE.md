# Pokemon TCG AI Battle - Operational Contract

This file is the agent-facing contract for the current `pokemon-tcg` checkout.
It replaces the old historical handoff. Use live code and current user
instructions over older notes.

## Authority Order

1. Alefita's current request.
2. The live repository state in this checkout.
3. This `CLAUDE.md`.
4. `TASK.md` and `docs/` specifications.
5. Canonical Wikifita pages in `/Users/alefita/Claude/wikifita`.
6. Historical transcripts, previous handoffs, and old archive notes.

Alefita owns the scientific direction. Agents should provide facts,
mechanisms, risks, uncertainty, and options when asked; do not prescribe the
research direction or re-open settled choices.

## Hard Boundaries

- Use `uv run` for Python entry points. Do not use raw `python` or `python3`
  commands in this repository.
- Do not start full training, full dataset rebuilds, full prospective builds,
  long tournaments, polling loops, or background automations unless Alefita
  explicitly asks.
- Agent-safe execution is smoke-sized only, usually through `configs/smoke.json`.
- Do not touch a live build's `data/`, `.work/`, shard, checkpoint, or process
  directories unless that is the explicit task.
- Do not modify `engine/`; it is supplied by the Kaggle environment.
- Do not commit credentials, tokens, private data, or raw sensitive transcripts.
- Keep code identifiers and comments in English.
- User-facing explanations may be in Brazilian Portuguese.
- `AGENTS.md` must be a relative symlink to `CLAUDE.md`. Never edit
  `AGENTS.md` as an independent file.

## Current Architecture

The active line is recurrent behavioral cloning with MLX training and PyTorch
FP16 submission inference.

The current full training config uses:

- `d_model=128`
- `nhead=4`
- `nlayers=4`
- `ff_dim=512`
- `scratch_registers=32`
- `structured=true`
- `tbptt_chunk=64`
- Muon+AdamW optimizer routing
- Prospective V2 enabled
- would-KO enabled

Code defaults in `rl/train_config.py` are fallback defaults, not necessarily
the active full-run contract. Check `configs/train_config.json` and the
checkpoint payload before making current claims.

## MLX Training, PyTorch Inference

Training is MLX. Submission inference is PyTorch FP16.

`scripts/build_submission.py` converts the selected MLX trainer checkpoint into
`model/bc_model/bc_best_torch_fp16.pt` and packages that self-contained
artifact. Runtime settings travel inside the checkpoint, including architecture
schema, encoder schema, would-KO settings, prospective planner settings,
optimizer/scheduler provenance, and training counters.

Submission archives must not depend on `train_config.json`, MLX runtime files,
vendored MLX wheels, network access, Google Drive, or an external service.

The submission package deliberately excludes:

- `rl/policy_mlx.py`
- `rl/prospective_planner_mlx.py`
- transient training configs
- vendored MLX dependencies

## Train Config And Checkpoints

`train_config.json` is a transient session sheet. It can change between runs.
The checkpoint is the durable source for architecture/runtime contracts.

Resume policy:

- `optimizer_state=reset` preserves model weights and starts fresh optimizer
  buffers.
- `optimizer_state=resume` requires a checkpoint with the identical optimizer
  contract.
- `scheduler_state=reset` starts a session-local scheduler phase.
- `scheduler_state=resume` uses the checkpoint scheduler horizon and rejects
  incompatible horizons.

The scheduler counts optimizer updates, not forward passes and not rows.
Warmup is clamped to the active scheduler horizon.

Checkpoints:

- `--out` is the best validation checkpoint path.
- `*_latest.pkl` is the rolling resume checkpoint.
- `checkpoint_every_epochs` writes numbered epoch snapshots.
- the final local epoch is always retained.

## TBPTT Contract

TBPTT is decision-based.

Rows are grouped by `(episode_id, side)` from `episode_meta.npy`, then split by
contiguous `step_id`. A multi-select decision can emit multiple rows, but those
rows remain one engine decision and share the same incoming memory.

The real TBPTT work plan obeys both:

```text
decisions_per_chunk <= tbptt_chunk
rows_per_physical_microbatch <= batch_size
```

Decisions are indivisible. One decision larger than `batch_size` is an error.

Optimizer accounting:

```text
microbatches_per_epoch = len(real_tbptt_plan)
optimizer_steps_per_epoch = ceil(microbatches_per_epoch / accum_steps)
run_optimizer_steps = local_epochs * optimizer_steps_per_epoch
```

The final partial accumulation window is stepped before validation. The trainer
raises if actual microbatches or optimizer steps diverge from the planned
counts.

Progress display is UI accounting. Do not explain a visual progress defect by
changing TBPTT, batch size, optimizer, or scheduler semantics without first
reading the actual Rich task state and iterator counts.

## would-KO

`would-KO` is generated at dataset/build time and recomputed from visible
runtime state when the checkpoint declares it. It is not hidden-deck lookahead.

Dataset manifests must distinguish:

- eligible attack rows;
- computed would-KO rows/options;
- valid zero outcomes;
- simulator failures;
- NaN/inf or out-of-range target defects.

Valid zero would-KO targets are evidence. They are not failures.

## Prospective V2 And Fita GRPO

Prospective V2 is implemented as an offline sidecar plus planner/runtime
scaffolding:

- `prospective_v2/` sidecar arrays and manifest;
- deterministic action coverage shared offline/runtime;
- additive RoPE-ND-style planner coordinates;
- group-relative branch objectives;
- PyTorch runtime reranking fallback.

The current objective can be described as group-relative ranking/distillation
over counterfactual visible-state branches. It is not strict on-policy GRPO.

A strict on-policy GRPO claim would require current-policy sampling, stored
behavior log-probs, update cadence, and on-policy group-relative advantages.
Keep that as research backlog unless the code and data contract actually
change.

## Action Coverage

Prospective branch coverage uses deterministic full-domain enumeration when the
legal domain fits the cap. When it does not fit, it samples evenly across the
lexicographic legal domain with endpoints included.

Current cap:

- `max_branches=64`

Offline builder and runtime must use the same enumeration code and compatible
schema/fingerprint contracts. Do not create a separate approximate runtime
branch generator.

## Data Pipeline

Durable sources are raw replay ZIPs and checkpoints. Encoded BC arrays and
prospective sidecars are reproducible derived artifacts.

The BC builder emits directory-form NPY datasets with:

- `__labels__.npy`
- `__would_ko_meta__.npy`
- `action_mask.npy`
- `episode_meta.npy`
- `dataset_manifest.json`
- feature arrays
- optional `prospective_v2/` sidecar

Shard completion is idempotent: `.done` is written last, and only completed
shards are reused. `.dataset_base_stage.json` allows a build interrupted before
prospective construction to resume from the validated base arrays.

Prospective sidecars validate adapter versions, compact storage version, action
schema, sources, config, BC fingerprint, and the no-hidden-deck/no-synthetic-fill
boundary. `prospective_workers` is operational and intentionally excluded from
the semantic sidecar fingerprint.

## Local Platform Status

`rl/results_db.py` currently implements partial local platform support:

- SQLite schema v2 with exact schema rejection/rebuild behavior;
- local/remote sources;
- teams, cards, exact deck fingerprints, submissions, and submission decks;
- tournaments, matchups, matches, participants, card usage;
- local replay steps/options/events/snapshots/Pokemon-on-field;
- idempotent receipts for tournaments and matches;
- source-separated card/deck Elo initialized from 600.

`scripts/tournament.py` persists aggregate tournaments and normalized local
match replay rows. `scripts/rebuild_db.py` rebuilds a remote-context database
atomically from canonical cards, decks, and replay ZIPs. `scripts/dashboard.py`
is a real Streamlit surface with overview, cards, decks, deck builder, agents,
arena, replays, and config tabs.

Still incomplete relative to `TASK.md`:

- model/model_revision tables;
- generic experiments and anamnese;
- training config/run tables;
- dashboard-editable tournament configs;
- full submission lifecycle/events;
- rating policies, epochs, submission/model Elo, append-only rating events;
- complete official visualizer reconstruction/launch.

## Deck Strategy

The submitted archive contains `deck.csv` at the root. That deck is immutable
for the submitted artifact.

Deck optimization is an outer loop:

```text
candidate decks -> local arena/self-play -> select one deck -> agent/deck.csv -> submission.tar.gz
```

The dashboard can write `agent/deck.csv` locally with a `.bak` backup. That is
pre-submission preparation, not hidden runtime deck selection.

`suggested_deck.csv` is future/planned only. It is not implemented in the
inspected code.

## Evaluation Boundaries

Historical `model/eval_results.txt` tournaments are useful regression context
but depend on older checkpoints, decks, opponents, and dates.

The July 29, 2026 one-epoch prospective run completed `320/320` local games.
The aggregate was about `10.3%` including smoke, but the useful breakdown was:

- `18/20`, `90%` vs smoke;
- `15/300`, `5%` without smoke.

Interpretation: the architecture and MLX-to-PyTorch inference path worked
end-to-end, but the one-epoch policy was undertrained and not submission-ready.

Earlier Kaggle `900-930` ladder/rating references are user-reported historical
provenance. Do not compare them directly to local tournament Elo.

## Agent-Safe Commands

Light checks:

```bash
uv run tcg-train --help
uv run tcg-build --help
uv run tcg-tournament --help
uv run tcg-rebuild-db --help
uv run scripts/bc/build_prospective_groups.py --help
git diff --check
```

Smoke-only work, when explicitly useful:

```bash
uv run tcg-build --smoke
uv run tcg-tournament --smoke --games 1 --no-sweep --note "smoke check"
uv run tcg-rebuild-db --dry-run --target model/checkpoint/smoke/results.db
```

Full-run examples, not agent-safe unless explicitly authorized:

```bash
uv run tcg-build-bc data/bc_data/bc_2026_07_28 data/bc_replay_zip/2026-07-28.zip --config configs/train_config.json
uv run tcg-train data/bc_data/bc_2026_07_28 --config configs/train_config.json --out model/checkpoint/bc_temporal_v2_mlx.pkl
uv run tcg-build --checkpoint model/checkpoint/bc_temporal_v2_mlx.pkl
uv run tcg-tournament --games 20 --no-sweep --note "describe the run"
```

## Documentation Map

- `README.md` - project overview and command map.
- `docs/README.md` - internal documentation map.
- `TASK.md` - exhaustive local platform target contract, not fully complete.
- `/Users/alefita/Claude/wikifita/kaggle/pokemon_tcg_ai_battle.md` - canonical
  Wikifita project hub.
- `/Users/alefita/Claude/wikifita/kaggle/pokemon_tcg_tbptt_training_contract.md`
  - exact TBPTT accounting.
- `/Users/alefita/Claude/wikifita/kaggle/pokemon_tcg_prospective_v2.md` -
  Prospective V2 and Fita GRPO boundary.
- `/Users/alefita/Claude/wikifita/kaggle/pokemon_tcg_data_pipeline.md` - data
  pipeline and sidecar idempotence.
- `/Users/alefita/Claude/wikifita/kaggle/pokemon_tcg_local_platform_status.md`
  - local platform implementation status.

## Before Committing

Always check status and preserve unrelated work:

```bash
git status --short --branch
git diff --check
```

For documentation-only changes, do not run training or rebuild datasets. Use
the light command checks above only when proportional.
