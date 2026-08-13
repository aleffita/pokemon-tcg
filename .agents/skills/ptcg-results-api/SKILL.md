---
name: ptcg-results-api
description: Regras e APIs estritas para extração de métricas de torneio, Elo e partidas do banco de dados (model/results.db) no projeto Pokemon TCG.
---

# Diretrizes de Extração de Dados (Pokémon TCG)

## 1. Schema do Banco de Dados (results.db)
Embora a API deva ser priorizada, o conhecimento do schema é vital para cruzamentos empíricos complexos via SQL (pandas/sqlite3). As principais tabelas são:

- `matches` (Atômica): `id`, `source`, `our_agent`, `our_deck_id`, `opp_agent`, `opp_deck_id`, `result` (1, 0, -1), `n_steps`, `created_at`.
- `tournaments` (Agregada): `id`, `timestamp`, `agent`, `games_per_opp`, `total_w`, `total_l`, `total_d`, `win_rate`.
- `matchups` (Blocos de Oponente): `id`, `tournament_id`, `opponent`, `w`, `l`, `d`, `win_rate`.
- `deck_elo_daily` (Elo): `deck_id`, `day_id`, `source`, `elo`, `games_played`, `wins`, `losses`.
- `agent_elo_daily` (Elo): `agent_name`, `day_id`, `source`, `elo`, `games_played`, `wins`, `losses`.
- `seasons` (Controle): `id`, `name`, `is_active`.

## 2. A Filosofia do "Deck Sweep" (Proibição de Agregação Cega)
O desempenho da rede neural na etapa atual da pesquisa **depende massivamente do Baralho Piloto (Archetype Overfitting)**. O modelo não é um jogador genérico homogêneo; ele varia de 0% a 30% de Win Rate puramente baseado no baralho que tem nas mãos.

**Regra Crítica:** 
**NUNCA** entregue ou analise relatórios de *Overall Win Rate* aos cientistas sem desmembrar a métrica baseada no Deck que o agente utilizou. A agregação cega mascara o viés comportamental do agente e destrói o diagnóstico empírico. Todo cruzamento DEVE ser feito explicitando:
- O agente em teste
- O torneio analisado
- O *Deck* piloto
- O *Deck* oponente (se possível)

## 3. Inicialização da API Nativa
Apesar de conhecer o schema, sempre instancie a classe `ResultsDB` (`rl/results_db.py`) quando a extração comportar seus métodos:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

from rl.results_db import ResultsDB
db = ResultsDB("model/results.db")
```

## 4. Extração de Agregados de Torneio (API)
Utilize a função interna para resgatar a matriz consolidada:
```python
runs = db.get_all_runs()
# Retorna uma lista com os sweeps organizados:
# r['agent'] (nome do agente)
# r['overall']['w'] e r['overall']['wr'] (Cuidado com agregação cega aqui!)
# r['matchups'] (Matriz exata contra oponentes específicos, vital para isolar os baralhos)
```

## 5. Extração de Elo Diário via SQL e a Régua Invariante
O sistema local de ranqueamento computa o Elo no formato *Standard* (zero-sum). Em pools locais e fechados, os valores irão estacionar ao redor do ponto de equilíbrio (ex: 800) independentemente da força real, impedindo a comparação com o *Leaderboard* remoto (1200+).
- **NUNCA** apresente ou analise o `elo_raw` do banco para validar a força global do modelo.
- **SEMPRE** utilize o método de translação `get_invariant_deck_elo()` ao extrair ranqueamentos:

```python
from rl.results_db import ResultsDB
db = ResultsDB("model/results.db")
# Aplicará a Inversão Asintótica de Bradley-Terry + Calibração Abeliana Softmax
metrics = db.get_invariant_deck_elo(deck_id)
print(metrics["elo_invariant"])
```
