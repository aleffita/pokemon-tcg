---
name: ptcg-results-api
description: Regras e APIs estritas para extração de métricas de torneio, Elo e partidas do banco de dados (model/results.db) no projeto Pokemon TCG.
---

# Diretrizes de Extração de Dados e Telemetria (Pokémon TCG)

## 1. Schema do Banco de Dados Relacional (`results.db` — Versão 2.0.0)
Toda extração deve obedecer estritamente à estrutura normalizada documentada em `docs/database_schema.md`:

- `matches` (Atômica, 139.783 linhas): `id`, `source` (`'remote'` ou `'local'`), `day_id`, `our_agent_id`, `opp_agent_id`, `matchup_id`, `game_index`, `our_agent`, `our_deck_id`, `opp_agent`, `opp_deck_id`, `our_side`, `result` (`1`: Vitória, `0`: Empate, `-1`: Derrota), `external_episode_id`, `source_observation_digest`, `archive_date`, `archive_member`, `n_steps`, `created_at`.
- `tournaments` (Agregada): `id`, `timestamp`, `agent`, `games_per_opp`, `total_w`, `total_l`, `total_d`, `win_rate`, `total_time_s`, `created_at`.
- `matchups` (Blocos de Oponente): `id`, `tournament_id`, `opponent`, `w`, `l`, `d`, `win_rate`, `lb_score`.
- `match_participants` & `match_card_usage`: Mapeamento granular de cartas e baralhos por participante.
- `deck_elo_daily` (Elo por Deck): `deck_id`, `day_id`, `source`, `elo`, `games_played`, `wins`, `losses`, `draws`, `computed_at`.
- `agent_elo_daily` (Elo por Agente): `agent_id`, `day_id`, `source`, `elo`, `games_played`, `wins`, `losses`, `draws`, `computed_at`.
- `seasons` (Controle de Época): `id`, `name`, `is_active`, `created_at`.

## 2. A Filosofia do "Deck Sweep" (Proibição de Agregação Cega)
O desempenho da rede neural na etapa atual da pesquisa depende massivamente do Baralho Piloto (Archetype Saliency). O modelo varia de 12.9% a 27.9% de Win Rate puramente baseado no baralho que tem nas mãos.

**Regra Crítica:** 
NUNCA entregue ou analise relatórios de *Overall Win Rate* sem desmembrar a métrica baseada no Deck que o agente utilizou. A agregação cega mascara o viés comportamental e destrói o diagnóstico empírico. Todo cruzamento DEVE explicitar:
- O agente em teste
- O torneio analisado
- O *Deck* piloto (ex: #633 Yan vs #251 Default)
- O *Deck* oponente (matriz assimétrica top-5)

## 3. Inicialização e Métodos da API Nativa (`rl/results_db.py`)
Sempre instancie `ResultsDB` para operações transacionais e consultas analíticas:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

from rl.results_db import ResultsDB
db = ResultsDB("model/results.db")
```

### 3.1. Sincronização de Leaderboard com TTL de 28h
Nunca execute chamadas brutas à API do Kaggle dentro de loops:
```python
# Verifica mtime de data/kaggle_leaderboard.csv e só bate na API se expirado (>28h)
df = db.sync_kaggle_leaderboard(ttl_hours=28)
```

### 3.2. Extração de Agregados de Torneio
```python
runs = db.get_all_runs()
# r['agent'] (nome do agente)
# r['overall']['w'], r['overall']['wr'] (Overall descompactado)
# r['matchups'] (Matriz contra oponentes específicos)
```

### 3.3. Extração de Elo Invariante Amortecido MD10 ($R_{\text{invariante}}$)
Nunca utilize o `elo_raw` para validar a força global do modelo. Sempre invoque a formulação invariante:

```python
inv_metrics = db.get_invariant_deck_elo(deck_id, source="local")
print(f"Elo Invariante: {inv_metrics['elo_invariant']:.1f} (MD10 Status: {inv_metrics['md10_complete']})")
```
