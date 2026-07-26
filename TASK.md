# TASK — MLX migration and minimal recurrence

## Objective

Prepare and implement, in a later coding task, the migration from the Mikaelzinho PyTorch snapshot to the Mikaelzinho MLX port, followed by the low-risk improvements already supported by the current architecture:

```text
MLX semantic corrections
  -> FP16-native trainer
  -> resumable checkpoints and accumulation
  -> exact padding/option compaction
  -> complete inference logs
  -> autoregressive multi-select
  -> persistent scratch registers
  -> sequential metadata and TBPTT
  -> Elo-oriented evaluation
```

This task file is a handoff and execution order. It does not authorize implementing Mamba, Hope, RoPE-ND, EBT, TRM, J-Lens, strategic MoE, PPO, GRPO, GSPO, a world model in inference, symbolic regression, custom Metal, or distributed Mac training.

## Baseline

Use the extracted MLX project and the corresponding Mikaelzinho PyTorch snapshot as the reference pair. Do not use the original author's full PPO repository as the parity target.

```text
d_model              128
attention heads      4
Transformer layers   3
FFN width            512
scratch registers    4
static card features enabled
split policy/value   enabled
structured head      disabled for current BC baseline
max options          192 + SUBMIT
```

Development target: M3 Pro, 24 GB unified memory. The final submission must be self-contained and must not require Drive, network, API calls, or an external teacher during inference.

## Global acceptance checklist

- [ ] No files under `engine/` changed.
- [ ] The MLX code uses one versioned token/architecture schema.
- [ ] Known token-type collisions are removed.
- [ ] Attention mask, MHA bias, padding ID zero, static buffers, and value head are correct.
- [ ] FP16 is native for model/data; loss, reductions, accumulation, and optimizer state use FP32 as specified.
- [ ] Gradient accumulation and scheduler are defined in optimizer-update units.
- [ ] `mx.compile`/`mx.eval` boundaries do not force per-step Python synchronization.
- [ ] Validation uses real cross-entropy.
- [ ] Checkpoints restore model, optimizer, scheduler, config, and trainer state.
- [ ] State/option padding compaction is exact.
- [ ] Complete logs reach inference trackers.
- [ ] Multi-select is autoregressive.
- [ ] Episode metadata is preserved and validation splits by episode.
- [ ] Scratch memory persists only within a match/side and resets correctly.
- [ ] TBPTT preserves order and isolates episodes.
- [ ] Functional tests pass.
- [ ] No PyTorch-vs-MLX benchmark suite was added as a project objective.

## Phase A — canonical MLX contract

- [ ] Add a single architecture/token schema module.
- [ ] Remove duplicated hardcoded token IDs.
- [ ] Fix the `self_bench`/`opp_active`/`opp_bench` collision.
- [ ] Freeze `128/4/3/512/4` for the current baseline.
- [ ] Version token schema and checkpoint configuration.
- [ ] Remove or correct any misleading `--ff` option.
- [ ] Make the loader reject incompatible schemas/configs.

### Acceptance

- [ ] Encoder, policy, trainer, agent, and checkpoint use the same semantic contract.
- [ ] Own and opponent zones remain distinguishable.
- [ ] A checkpoint cannot silently instantiate another architecture.

## Phase B — semantic P0 corrections

- [ ] Replace boolean attention padding mask with additive `0/-inf` mask.
- [ ] Match the reference MHA bias behavior explicitly.
- [ ] Make card/attack ID zero contribute a zero vector.
- [ ] Keep `card_feat` immutable and out of trainable parameters.
- [ ] Keep `atom_support` immutable when present.
- [ ] Make categorical value return `sum(softmax(logits) * support)`.
- [ ] Validate source/target index remapping after compaction.

### Tests

- [ ] Extra right padding does not change real-token outputs.
- [ ] Empty ID lists do not accumulate a learned absence vector.
- [ ] Static arrays are unchanged after an optimizer update.
- [ ] Invalid actions are masked.
- [ ] Categorical value has the expected scalar shape.

## Phase C — FP16 trainer and complete state

- [ ] Remove the FP16-to-NumPy-FP32 batch conversion.
- [ ] Keep model inputs, embeddings, linears, Q/K/V, and residuals in FP16.
- [ ] Compute logits/loss/reductions in FP32.
- [ ] Add gradient accumulation with FP32 accumulators.
- [ ] Clip once after accumulation.
- [ ] Keep clipping graph-safe.
- [ ] Compile forward/loss/backward/clip/update where shapes are stable.
- [ ] Materialize state with `mx.eval` at correct boundaries.
- [ ] Count scheduler steps per optimizer update.
- [ ] Fix total steps across all epochs and accumulation.
- [ ] Replace raw-logit validation loss with stable cross-entropy.
- [ ] Save model, optimizer, config, trainer state, scheduler position, seed, and dataset manifest.

### Acceptance

- [ ] Training resumes without resetting optimizer or schedule.
- [ ] Effective batch size is explicit.
- [ ] Losses and gradients are finite.
- [ ] Validation loss is mathematically valid.

## Phase D — exact data and shape handling

- [ ] Derive slab size from the M3 Pro memory budget.
- [ ] Start with conservative 32k–64k rows and prefetch depth 1.
- [ ] Add finite option buckets: 32/64/128/192.
- [ ] Port exact state compaction.
- [ ] Remap source/target indices after removing state columns.
- [ ] Add `episode_id`, `side`, `step_id`, `decision_id`, `substep`, reset/new-episode, terminal, and reward metadata.
- [ ] Split by episode, never by raw row suffix.
- [ ] Deduplicate episodes during daily ingestion.
- [ ] Validate ranges, NaN/Inf, and label/mask consistency.
- [ ] Keep episode manifest with source/date/deck/submission metadata when available.

### Acceptance

- [ ] No chunk crosses an episode boundary.
- [ ] Padding that can be removed exactly is not processed.
- [ ] Temporal order is preserved inside each episode/side.
- [ ] The loader does not materialize the full corpus.

## Phase E — inference action semantics

- [ ] Pass the full observation, including logs, to tracker, ability tracker, encoder, and memory.
- [ ] Remove the one-pass `topk(count)` implementation.
- [ ] Recompute logits after each selected option.
- [ ] Update `picked` and the action mask after each substep.
- [ ] Respect `min_count`, `max_count`, duplicate prevention, and legal `SUBMIT`.
- [ ] Use exact batch-one compaction in inference.
- [ ] Keep inference inputs in FP16 where appropriate.
- [ ] Validate the bundle without Drive/network/external files.

### Acceptance

- [ ] Training and inference receive equivalent information.
- [ ] Multi-select follows the factorization represented by the dataset.
- [ ] Selected options cannot be selected twice.
- [ ] `SUBMIT` ends only a legal selection.

## Phase F — minimal persistent registers and TBPTT

- [ ] Add `memory_in`/`memory_out` to the model API.
- [ ] Use the scratch-token output as `memory_out`.
- [ ] Use learned scratch parameters only as initial memory when input memory is absent.
- [ ] Store memory per match and side.
- [ ] Reset at match start and never share between sides/matches/processes.
- [ ] Build ordered sequences by `(episode_id, side)`.
- [ ] Add TBPTT chunks of 8/16/32 decisions.
- [ ] Carry memory across chunks.
- [ ] Stop gradients only at chunk boundaries.
- [ ] Accumulate gradients over chunks.
- [ ] Add counterfactual validation pairs with similar local state and different histories.

### Acceptance

- [ ] A new match starts with clean memory.
- [ ] Later decisions receive earlier scratch state.
- [ ] Simultaneous matches have isolated memories.
- [ ] Ordered sequence batching is enforced.
- [ ] Counterfactual histories can produce different memory when relevant.

## Release candidates

### RC1 — corrected MLX

- [ ] Phases A–C complete.
- [ ] Semantic smoke tests pass.
- [ ] New self-contained checkpoint produced.

### RC2 — action semantics

- [ ] Phases D–E complete.
- [ ] Logs are complete in inference.
- [ ] Multi-select autoregressive path passes engine smoke tests.

### RC3 — recurrent registers

- [ ] Phase F complete.
- [ ] Sequential dataset and TBPTT tests pass.
- [ ] Reset/isolation tests pass.

### RC4 — refined corpus

- [ ] Episode deduplication active.
- [ ] Historical/recent mixture defined.
- [ ] Rare actions/matchups tracked.
- [ ] Episode-level holdout active.

## Elo-oriented evaluation record

For every release record:

```text
release id
dataset manifest
architecture config
training state
deck
submission artifact
observed Elo/rating
matchup results
known regressions
```

Primary criteria are Elo, matchup robustness, action legality, long-horizon behavior, and no semantic regressions. Throughput is a means to train the right temporal unit, not the target.

## Implementation guardrails

- [ ] Use `uv run` for Python commands.
- [ ] Keep code identifiers/comments in English and handoff prose in Brazilian Portuguese.
- [ ] Preserve existing checkpoints and data.
- [ ] Do not delete user artifacts.
- [ ] Add a functional test for each semantic change.
- [ ] Update this file only after the corresponding behavior is verified.
