# Pokémon TCG AI Battle — Current MLX Handoff

> **Authoritative scope:** this handoff governs the current implementation task. The original snapshot notes are preserved below as historical inventory and context; when they conflict with this section or the detailed plan at the end of the file, the current handoff wins.

## Repository initialization

This project was extracted from the shared Google Drive artifact `tcg-pokemon-agent-mlx-port.zip` (Drive file ID `1R3wCNKXlnJ5jEHbYtqkjyQ_aOkxvEe_x`, snapshot received 2026-07-25). The repository is intentionally prepared but not yet modified to implement the plan. The initial Git commit must contain the extracted source plus this merged handoff and `TASK.md`.

## Current objective

Migrate the **Mikaelzinho PyTorch snapshot** to the **Mikaelzinho MLX port** with functional parity at the intended architecture, then connect the low-risk capabilities already latent in the dataset and model:

```text
MLX semantic corrections
  -> FP16-native trainer
  -> reliable checkpoints and accumulation
  -> exact padding/option compaction
  -> complete inference logs
  -> autoregressive multi-select
  -> persistent scratch registers
  -> sequential metadata and TBPTT
  -> Elo-oriented evaluation
```

This is not a benchmark project. Do not spend the implementation phase comparing PyTorch speed to MLX speed or constructing a retroactive parity benchmark. MLX is the selected runtime. The required validation is semantic correctness, functional behavior, and improvement toward Elo.

The author-original repository and its full PPO/self-play infrastructure are context only. They are not the comparison target for this port. The concrete baseline is the simplified Mikaelzinho behavioral-cloning implementation and the MLX code included in this archive.

## Explicit non-goals for this phase

Do **not** introduce yet:

- Mamba/Mamba-2;
- Hope/Nested Learning or advanced continual learning;
- RoPE-ND or a new positional geometry;
- Energy-Based Transformer;
- TRM;
- J-Lens/mechanistic interpretability;
- strategic MoE;
- PPO, GRPO or GSPO;
- a world model in the submitted inference path;
- symbolic regression;
- custom Metal kernels;
- multi-Mac distributed training;
- changes to the supplied game engine.

Those are later research directions. This phase exists to make the current agent correct and to activate the recurrence and action semantics it already almost has.

## Development boundary

Use the M3 Pro with 24 GB unified memory as the development target. The M1 Air is not part of the main runtime design for this task. Do not assume Thunderbolt creates one transparent 32 GB MLX heap.

The final artifact must be self-contained. It must not require Google Drive, network access, external APIs, a teacher service, or a remote bucket during arena inference. Development and offline training may use local data and local tools, but the shipped agent must carry what it needs.

## Authority order

When deciding whether a change belongs in this task, use this order:

1. the current user request;
2. this current handoff section;
3. `TASK.md` and its acceptance criteria;
4. the actual source snapshot;
5. the historical inventory below.

## Imported historical inventory

The remainder of the original `CLAUDE.md` is intentionally preserved. It records the source archive's directory structure, known status, datasets, checkpoints, engine restrictions, and local workflow. It is useful for orientation, but statements such as the old branch labels, old hardware assumptions, future PPO plans, and direct `python3` examples are historical. Use `uv run` for Python and follow the current plan below.


## Projeto

Agente de RL para a Kaggle Pokemon TCG AI Battle Challenge.
Pipeline planejado: Behavioral Cloning (BC) → PPO self-play → deck finetune.
Referência: repo do #1 do leaderboard (Majkel) que usa BC + PPO + Self-Play (porém com 4x H200).

## Branches

- **main** — BC pipeline completo (PyTorch), submissions na leaderboard
- **mlx-port** — Port do modelo e treino pra MLX (Apple Silicon nativo, sem bugs MPS)

## Estrutura

```
tcg-pokemon-agent/
├── agent/                  # Nosso agente (main.py + deck.csv) — submission
├── rl/                     # Pipeline de RL / Imitation Learning
│   ├── encoder/            # Encoding do estado do jogo (portado do repo do #1)
│   │   ├── enc_constants.py    # Shape/layout constants (token model + mlp)
│   │   ├── card_features.py    # Static per-card features from EN_Card_Data.csv
│   │   ├── encoding.py         # obs dict → numpy arrays (TokenEncoder, GameTracker)
│   │   ├── attack_data.py      # Per-attack properties (damage, cost, effects) — gerado
│   │   ├── buff_data.py        # Transient turn-scoped buff tables (defense/offense)
│   │   ├── option_dedup.py     # Action-space dedup (collapse interchangeable options)
│   │   └── effect_data.py      # Attack/ability/trainer effect multi-hots
│   ├── deck/               # Deck definitions (decks.py, decks_generated.py, decks_kaggle.py)
│   ├── env/                # Environment wrappers (env.py, vec_env.py) — pra PPO
│   ├── policy.py           # TokenTransformer model (pointer-scoring Transformer)
│   ├── lr_schedule.py      # LR schedule (warmup + cosine/linear decay)
│   ├── train_ppo.py        # PPO self-play trainer (futuro)
│   └── search_agent.py     # Search/MCTS agent (needed for would_ko sim + validate_dedup)
├── model/                  # Checkpoints e modelos treinados
│   ├── checkpoint/         # Checkpoints intermediários (bc_train --out default)
│   └── bc_model/           # Melhor modelo BC final (copiado automaticamente)
├── cg/                     # SDK do Kaggle (ctypes wrapper p/ engine C++)
│   ├── sim.py              # Carrega lib nativa (dylib/so/dll) + ctypes structs
│   ├── game.py             # battle_start/select/finish API
│   ├── api.py              # Observation dataclasses (to_observation_class)
│   ├── utils.py            # Dict → dataclass conversion helpers
│   ├── libcg.dylib         # macOS native lib
│   ├── libcg.so            # Linux x86_64 native lib
│   ├── libcg-arm64.so      # Linux ARM64 native lib
│   └── cg.dll              # Windows native lib
├── engine/                 # C++ source (read-only, fornecido pelo Kaggle)
│   └── ptcgProgram22/      # Headers do game engine (~44 arquivos .h/.cpp)
├── data/
│   ├── manifest.csv            # URLs dos datasets diários de replay
│   ├── replay/                 # Replay samples (ex: 85966927.json)
│   ├── bc_replay_zip/          # Zips de replay do Kaggle (input do pipeline BC)
│   │   ├── 2026-07-16.zip     # 4760 episodes, ~698MB
│   │   └── 2026-07-21.zip     # ~5000 episodes, ~700MB (novo)
│   └── bc_data/                # Output do pipeline BC (.npy dir)
│       └── bc_2026_07_21/     # 801,865 rows, masked_rate=0.0
├── public_agents/          # Agentes públicos pra benchmark
│   ├── lb1009_mega_lucario_ex_islet/
│   ├── lb945_multiply_ivan/
│   ├── lb826_alakazam_seok/
│   ├── lb814_crustle_emre/
│   ├── lb798_lucario_pilkwang/
│   ├── starters/           # Starters do staff (lb600_dragapult, lb600_mega_lucario, etc.)
│   └── submissions/        # Submissions salvas pra benchmark
│       └── lb881_alakazam_v1/submission.tar.gz
├── scripts/
│   ├── bc/                     # BC pipeline
│   │   ├── bc_train.py             # BC trainer (TokenTransformer, CPU-only on M1)
│   │   ├── build_bc_dataset.py     # Single-replay builder (offline, streaming)
│   │   ├── build_bc_from_zips.py   # Zip builder (batch streaming, checkpoint resume)
│   │   ├── debug_nan.py            # NaN diagnostic (single step, data/model/grad check)
│   │   └── debug_nan_steps.py      # NaN diagnostic (multi-step, finds exact NaN step)
│   ├── deck_builder/           # Visual deck builder
│   │   ├── extract_card_images.py  # Extract card images from PDF
│   │   ├── build_deck_tool.py      # Generate HTML deck builder tool
│   │   ├── card_pages.json         # Card ID → PDF page mapping
│   │   ├── card_images/            # Extracted card images (1267 JPGs)
│   │   └── deck_builder.html       # Visual deck builder (generated)
│   ├── _common.py              # Shared helpers (load_agent, make_env, load_submission)
│   ├── evaluate.py             # 1v1 eval (win rate vs random/first)
│   ├── run_battle.py           # Single battle + HTML replay
│   ├── tournament.py           # Mini-torneio vs múltiplos oponentes (--note flag)
│   ├── build_submission.py     # Empacota + valida submission.tar.gz
│   ├── play_test.py            # Smoke test do SDK (random vs random)
│   └── validate/               # Validação do encoding
├── EN_Card_Data.csv        # Card data completo (all cards, attacks, costs)
├── pyproject.toml          # uv project (Python 3.13, numpy)
└── uv.lock                 # Lock file do uv
```

## Referência externa (read-only)

`~/Workspaces/poke-rl-ref/` — repo clonado do Majkel (#1 leaderboard, branch encoding-overhaul)

- `rl/` — encoding, card_features, policy, training
- `scripts/` — build_bc, evaluate, build_submission
- `native_encode/` — Cython fast path
- `HANDOFF.md` — playbook completo do #1

## Status do projeto

### Encoder (validado ✅)
TokenEncoder converte obs dict → arrays numpy. 46 feature keys, N_STATE_TOKENS=337, MAX_OPTIONS=192.

### Datasets
| Dataset | Rows | Source | Status |
|---|---|---|---|
| bc_2026_07_16 | 788,369 | 4760 eps, dia 16 | Treinado (checkpoint v1) |
| bc_2026_07_21 | 801,865 | ~5000 eps, dia 21 | Em treino (continual learning) |

### BC Model
- **Arquitetura**: d128, L3, h4, static, split-heads — **1,090,947 params** (PyTorch)
- **Checkpoint v1**: 3 epochs (dia16), val_acc=0.6882, equiv=0.7036, top3=0.9121
- **Checkpoint v2**: em treino (dia21, continual learning do v1)

### MLX Port (funcional — branch mlx-port)
- **Modelo**: `rl/policy_mlx.py` — TokenTransformerMLX (1,202,352 params)
- **Encoder**: `rl/encoder/` — reutilizado (numpy, sem mudanças)
- **Trainer**: `scripts/bc/bc_train_mlx.py` — MLX Metal GPU, fp16 dataset, checkpoint resume
- **Dataset fp16**: `data/bc_data/bc_2026_07_21_fp16/` (18.9 GB, -40.6% vs float32)
- **Status**: Treinando, val_acc=0.6947 (epoch1, fp16)
- **Velocidade**: ~2.7h/epoch (batch 128, fp16, MLX Metal GPU)
- **Notas**: PyTorch MPS tem NaN com Transformers — MLX funciona sem problemas
- **Análise**: `data/reports/architecture_analysis.md`, `data/reports/mlx_research.md`

### Torneio (v1, 20 jogos cada)
| Oponente | LB Score | Win Rate |
|---|---|---|
| random | — | 100% |
| first | — | 100% |
| lb1009_mega_lucario_ex_islet | 1009 | 95% |
| lb945_multiply_ivan | 945 | 85% |
| lb826_alakazam_seok | 826 | 100% |
| lb814_crustle_emre | 814 | 100% |
| lb798_lucario_pilkwang | 798 | 90% |
| starters/lb600_mega_lucario_ex | 600 | 95% |
| starters/lb600_dragapult_ex | 600 | 65% |
| starters/lb526_iono | 526 | 55% |
| starters/lb510_mega_abomasnow_ex | 510 | 90% |
| sub/lb881_alakazam_v1 | 881 | 65% |
| **OVERALL** | — | **82.5%** |

### Submissions no Kaggle
| Sub | Data | Epochs | Dataset | LB Score |
|---|---|---|---|---|
| v1 | 2026-07-21 | 2 | dia 16 | 889.7 |
| v2 | 2026-07-21 | 3 | dia 16 | ~873 |
| v3 | 2026-07-22 | — | dia 16 + deck Yushin | pendente |

### Deck Builder (novo ✅)
Tool visual HTML pra montar/visualizar decks:
- `scripts/deck_builder/extract_card_images.py` — extrai imagens do PDF
- `scripts/deck_builder/build_deck_tool.py` — gera HTML interativo
- Suporta: busca, filtros, load/export CSV, View Deck

### Estratégia de dados
- Datasets encodados no Google Drive (1TB disponível)
- M1 com ~50GB livres — trocar datasets conforme necessário
- Continual learning: checkpoint preserva conhecimento dos dados antigos
- Zips pequenos (~700MB/dia) — manter vários

### Próximos passos
1. Avaliar treino v2 (continual learning, dia21) — em andamento
2. Rebuildar submission com modelo atualizado
3. Testar decks alternativos (Marnie's Impidimp, Team Rocket, Garchomp)
4. PPO self-play (futuro)

## Known issues
- **PyTorch MPS + Transformer = NaN**: MPS tem bug numérico com attention/layer norm. CPU funciona. MLX Metal funciona sem problemas — usar `bc_train_mlx.py`.
- **MLX mx.savez + model.update**: flatten/unflatten de params causa mismatch de nomes. Usar pickle (.pkl) pra checkpoints.
- **MLX bf16**: `mx.set_default_dtype` não existe. Converter arrays manualmente.
- **Engine crasha no sandbox Linux**: `libcg.dylib` é Mach-O. Funciona no Mac local.
- **OOM com multiprocessing**: batch streaming resolve (500 eps/batch + checkpoint resume).
- **Kaggle __file__**: `__file__` não é definido no sandbox — usar sys.path lookup.
- **bc_train --epochs**: precisa ser MAIOR que o epoch do checkpoint, senão não treina.

## Regras importantes

- **NÃO modificar** nada em `engine/` (read-only, fornecido pelo Kaggle)
- **Código em inglês** (variáveis, funções, comentários)
- **Explicações em português** brasileiro informal

## Hardware

- Dev local: MacBook Air M1, 8 GB RAM unificada, CPU (MPS bugado pra Transformers)
- Kaggle: 12h CPU/dia, 30h GPU/mês (P4/P100)
- Competição: submissions rodam CPU-only, GPU-P4x2, ou GPU-P100
- Google Drive: 1TB (backup de datasets, Colab plugin)

## Ambiente Python

- **Runtime**: Python 3.13.5 via uv (`.venv/bin/python`)
- Dependências: numpy, PyTorch (CPU), kaggle-environments (GitHub master)
- `PYTHONPATH=.` necessário pra rodar scripts

## Comandos úteis

```bash
# Build BC dataset (de zip)
PYTHONPATH=. python3 scripts/bc/build_bc_from_zips.py data/bc_data/bc_novo data/bc_replay_zip/2026-07-XX.zip

# Treinar BC (novo treino)
PYTHONPATH=. python3 -u scripts/bc/bc_train.py data/bc_data/bc_2026_07_21 --d-model 128 --static --split-heads --epochs 8 --batch 128

# Treinar BC (continual learning — epochs > checkpoint epoch!)
PYTHONPATH=. python3 -u scripts/bc/bc_train.py data/bc_data/bc_2026_07_21 --d-model 128 --static --split-heads --epochs 8 --resume model/checkpoint/bc_best.pt --batch 128

# Eval 1v1
PYTHONPATH=. python3 scripts/evaluate.py -n 50

# Torneio (com nota)
PYTHONPATH=. python3 scripts/tournament.py --games 20 --note "descrição da run"

# Torneio contra submission salva
PYTHONPATH=. python3 scripts/evaluate.py agent/main.py public_agents/submissions/lb881_alakazam_v1/submission.tar.gz -n 20

# Build + validar submission
PYTHONPATH=. python3 scripts/build_submission.py

# Deck builder
python3 scripts/deck_builder/extract_card_images.py
python3 scripts/deck_builder/build_deck_tool.py
open scripts/deck_builder/deck_builder.html

## Current architectural contract

Freeze this contract while implementing the migration:

```text
d_model              = 128
attention heads      = 4
Transformer layers   = 3
FFN width            = 512 (4 * d_model)
scratch registers    = 4
static card features = enabled
split policy/value   = enabled
structured head      = disabled for the current BC baseline
max options          = 192 plus SUBMIT where applicable
```

The model is an entity/action Transformer with structured option scoring. It is not a causal language decoder. The potential sequence is approximately:

```text
[CLS] [SELECT_TYPE] [SELECT_CONTEXT]
[state entity tokens]
[scratch tokens]
[option tokens]
```

The state has roughly 337 potential positions. There are up to 192 option positions plus `SUBMIT`. Padding slots are capacity in the storage format, not necessarily real entities or actions.

### One canonical token schema

Create one versioned architecture specification shared by encoder, MLX policy, trainer, loader, and agent. It must distinguish at least:

```text
CLS, SELECT_TYPE, SELECT_CONTEXT,
SELF_DECK, OPP_DECK,
SELF_PRIZE, OPP_PRIZE,
SELF_HAND, OPP_HAND,
SELF_DISCARD, OPP_DISCARD,
STADIUM,
SELF_ACTIVE, SELF_BENCH,
OPP_ACTIVE, OPP_BENCH,
EFFECT, OPTION, SCRATCH
```

Do not keep independent hardcoded ID maps. The known MLX audit found semantic collisions in the unit token mapping; own bench, opponent active, and opponent bench must not share a type by accident.

### Padding IDs

Card and attack ID zero is padding, including empty positions inside pre-evolution, tool, energy, and attack bags. It must contribute a zero vector, not a learned “absence” embedding copied multiple times. Apply an explicit `ids != 0` mask to the embedding output or otherwise reproduce the PyTorch `padding_idx=0` semantics.

### Config versioning

Every checkpoint must record and validate `architecture_version`, dimensions, scratch count, static/split-head flags, structured-head flag, option capacity, dtype, and token schema version. Never silently load a checkpoint with a different ontology.

## P0 semantic corrections

These are correctness fixes, not optional performance work.

### Additive attention mask

The MLX port currently builds a boolean padding mask and passes it to multi-head attention. The intended mask is additive:

```text
0       for an available key
-inf    for a padded or blocked key
```

Use the actual installed MLX API contract and a dtype-compatible large negative value. Add a functional test proving that right-padding does not change the outputs of real tokens.

### MHA bias

The PyTorch reference uses bias in the attention projections. Instantiate the MLX attention explicitly with the matching bias behavior; do not inherit an MLX default that silently changes the model.

### Static tables

Keep `card_feat` and `atom_support` immutable. They are domain data/buffer-like inputs, not trainable parameters. Learn a projection of the static features if needed, but do not update the table itself or allocate optimizer state for it.

### Categorical value head

When categorical value is enabled, return the expected scalar:

```text
p = softmax(atom_logits)
value = sum(p_i * atom_support_i)
```

Do not return atom logits as a scalar value.

## FP16-native trainer

Treat FP16 as the correct representation for the current workload, not as a benchmark experiment. The current port writes FP16 data but converts it back to NumPy FP32 before creating MLX arrays; remove that round trip.

Use this dtype contract:

```text
IDs/labels/positions       int32
masks                      bool or uint8
numeric features           float16
embeddings/linear/QKV      float16
residual stream            float16
logits for loss            float32
loss reductions            float32
accumulated gradients      preferably float32
Adam state                 float32
metrics                    float32 or host
```

Do not add dynamic loss scaling until there is evidence of underflow. First use FP16 model/data with FP32 loss, reductions, accumulation, and optimizer state; check finiteness.

### Gradient accumulation

Implement accumulation in the first functional trainer. For `K` microbatches, accumulate in FP32 and normalize by the real example count, including an incomplete final microbatch. Clip once after accumulation, then perform one optimizer step.

The sequence is:

```text
forward -> FP32 CE -> backward -> accumulate
repeat K times
normalize -> clip -> optimizer update
increment optimizer step -> scheduler update -> mx.eval
```

The scheduler counts optimizer updates, not forwards.

### Compile/eval boundaries

Keep forward, loss, backward, clipping, and optimizer update in the MLX graph where shapes are stable. Do not convert gradient norms to Python `float` each step. Use an MLX graph-safe clipping operation. Materialize the model/optimizer state explicitly with `mx.eval` at update and buffer-reuse boundaries.

### Scheduler and validation

Compute total steps across every epoch and accumulation step:

```text
total_optimizer_steps = epochs * ceil(microbatches / accumulation_steps)
```

Restore scheduler position on resume. The known port bug computes steps for one epoch while the global counter crosses all epochs.

Validation must use true cross-entropy:

```text
CE = -(logit[label] - logsumexp(logits))
```

The legacy `log(raw_logit)` calculation is not a valid loss and must not select checkpoints.

### Complete checkpoints

Save model parameters, optimizer state, architecture config, trainer state, scheduler/global step, seed, and dataset manifest. Resume must restore all of them, not just model weights.

## Exact padding and memory handling

The purpose of compaction is to remove exact zeros and inaccessible slots without approximating the model.

### Option buckets

Compute the largest real option index in a batch and round to a finite set such as:

```text
32, 64, 128, 192
```

Keep a finite set of compiled shapes. Preserve `SUBMIT`, action masks, and option indexing.

### State compaction

Remove a state column only when it is padded for every row in the batch. Always retain tokens needed by CLS, selection/context, scratch, value, and option source/target references. Remap source/target indices after compaction using the existing reference logic.

### Slabs

The archive default of 262,144 rows is too aggressive for the M3 Pro. `opt_attr` is about 13,824 bytes per row in FP16, so one such slab is roughly 3.6 GB before other columns, model state, optimizer, gradients, activations, and macOS. With a prefetched slab, two can coexist.

Use a conservative, manifest-derived slab policy, initially around 32k–64k rows on the M3 Pro, with prefetch depth 1. Treat the slab as an I/O unit, not a fixed memory claim.

## Inference correctness before architecture changes

The dataset builder uses observations and incremental logs, but the current agent reduces the live input to `logs=[]`. This creates different belief states in training and inference. The same complete observation must flow to the game tracker, ability tracker, encoder, and persistent memory.

Change the conceptual API from:

```python
choose(select, current)
```

to:

```python
choose(obs)
```

Do not discard revealed cards, serials, zone moves, attacks, or effects contained in logs.

## Autoregressive multi-select

The dataset teaches a sequence:

```text
state, picked=[]       -> option_1
state, picked=[1]      -> option_2
state, picked=[1, 2]   -> SUBMIT
```

The current `topk(count)` inference is not the same policy. Implement a loop that chooses one legal option, updates `picked`, rebuilds the option mask/tokens, and runs the next substep until legal `SUBMIT` or `max_count`. Respect minimum count, maximum count, and duplicate prevention. Recompute logits after each selected option.

## Sequential metadata and persistent scratch

The builder is temporally aware while reading replays, but the saved rows are independent. Add sidecars or an equivalent schema containing:

```text
episode_id
side
step_id
decision_id
substep
new_episode/reset
terminal
reward
```

Split by episode, never by a raw row suffix. Chunks must not cross episodes; daily ingestion must deduplicate episodes.

The minimum recurrent change uses the existing four scratch registers:

```text
J_0_in  = learned_initial_registers
J_{t+1} = scratch_slice(model_output_t)
```

Expose `memory_in` and `memory_out`. Store memory per match and per side. Reset at a new match, never share it across sides or processes, and do not add a new recurrent cell in this phase.

## TBPTT

Train sequences grouped by `(episode_id, side)` in ordered chunks of 8, 16, or 32 decisions. Carry the register state between chunks and apply `stop_gradient` only at the chunk boundary. Mask padded timesteps and normalize loss by real decisions. Gradient accumulation operates over chunks.

## Functional tests

### Model

- token types match the canonical schema;
- ID zero contributes no embedding;
- static card/support tables are unchanged after an update;
- padding does not affect real-token outputs;
- invalid options are masked;
- loss and gradients are finite;
- categorical value returns a scalar expectation;
- checkpoint round-trip restores config and state.

### Data

- every label is legal under its action mask;
- episode and step metadata are consistent;
- chunks do not cross episodes;
- reset flags are exact;
- validation shares no episode with training;
- multi-select substeps preserve order.

### Multi-select

- selected options disappear from the mask;
- min/max counts are enforced;
- `SUBMIT` is only accepted when legal;
- forward count follows the number of substeps.

### Recurrence

- initial memory is deterministic/resettable;
- memory persists between decisions;
- match and side memories are isolated;
- `stop_gradient` only occurs at TBPTT boundaries;
- a new match does not inherit old memory.

### Submission

- full logs are passed to the tracker;
- the bundle runs without external files or network;
- the CPU path remains available as a fallback;
- no engine files were modified.

## Implementation phases

### Phase A — canonical MLX contract

Centralize architecture/token schema, fix IDs, freeze `128/4/3/512/4`, version checkpoints, and remove misleading configuration flags.

**Exit:** encoder, policy, trainer, agent, and checkpoint share one semantic contract.

### Phase B — semantic P0 fixes

Fix additive mask, MHA bias, padding ID zero, static buffers, categorical value, and config loading.

**Exit:** valid MLX forward with no known semantic collisions or accidental trainable static data.

### Phase C — FP16 trainer

FP16 end-to-end, FP32 loss/reductions/accumulation, accumulation, post-accumulation clipping, optimizer-step scheduler, compile/eval boundaries, validation CE, complete checkpoints.

**Exit:** resumable, finite, effective-batch-controlled training.

### Phase D — data and shapes

Adaptive slabs, exact state/option compaction, finite compiled buckets, sequential metadata, episode split, deduplication and validation.

**Exit:** no unnecessary padding work and enough metadata to preserve trajectories.

### Phase E — inference semantics

Complete logs, autoregressive multi-select, batch-one compaction, FP16 input path, self-contained bundle.

**Exit:** training and inference see the same information and action factorization.

### Phase F — minimal recurrence

Memory API, per-match/per-side registers, sequence batcher, TBPTT, chunk accumulation, and counterfactual validation.

**Exit:** the agent is no longer single-shot between decisions without introducing Mamba, Hope, TRM, EBT, or another new paradigm.

## Release candidates oriented to Elo

### RC1 — corrected MLX

Semantic contract, mask, embeddings, static tables, FP16, valid loss, complete checkpoint.

### RC2 — action semantics

Exact compaction, complete logs, autoregressive multi-select, same current model otherwise.

### RC3 — recurrent registers

Persistent memory, sequential dataset, TBPTT, reset/isolation tests.

### RC4 — refined corpus

Episode deduplication, historical/recent mixture, rare matchups/actions, episode holdout and manifest.

Record for every release:

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

Elo, matchup robustness, action legality, long-horizon behavior, and absence of semantic regressions are the evaluation criteria. Throughput is only an enabler for the correct training unit; it is not the target.

## Repository rules

- Do not modify `engine/`.
- Do not add credentials, tokens, or private data to Git.
- Use `uv run` for Python commands.
- Keep code identifiers and comments in English.
- Keep explanations and handoff in Brazilian Portuguese.
- Preserve existing checkpoints/data while implementing.
- Add a functional test for every semantic change.
- Update `TASK.md` only when work is actually verified.
```
