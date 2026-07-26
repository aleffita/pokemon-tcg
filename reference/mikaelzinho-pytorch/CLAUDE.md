# PTCG AI Battle Challenge — Agent Workspace

## Projeto

Agente de RL para a Kaggle Pokemon TCG AI Battle Challenge.
Pipeline planejado: Behavioral Cloning (BC) → PPO self-play → deck finetune.
Referência: repo do #1 do leaderboard (Majkel) que usa BC + PPO + Self-Play (porém com 4x H200).

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
- **Arquitetura**: d128, L3, h4, static, split-heads — **1,090,947 params**
- **Checkpoint v1**: 3 epochs (dia16), val_acc=0.6882, equiv=0.7036, top3=0.9121
- **Checkpoint v2**: em treino (dia21, continual learning do v1)

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
- **PyTorch MPS + Transformer = NaN**: MPS tem bug numérico com attention/layer norm. CPU funciona.
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
```
