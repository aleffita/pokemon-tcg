# Architectural Blueprint: ETL Pipeline Integrity & Data Auditing

## 1. A Premissa da Falha Silenciosa (Database Trust Bias)
Assumir um banco de dados relacional compilado (`results.db`) como a representação integral de um *dataset* distribuído em rede é uma vulnerabilidade de arquitetura. Durante a operação, o SQLite reportava que a data mais recente armazenada (`archive_date`) correspondia exatamente ao último ZIP contido no disco físico (`2026-08-12.zip`). 

Isso mascarou uma profunda fratura de ETL (Extract, Transform, Load): não faltava um dia inteiro na base; faltava integridade intradiária. Ao executar um script de auditoria física para mergulhar nos 31 pacotes ZIP e computar nó por nó cada instância `.json`, os resultados expuseram a realidade:
- **Total Físico de Replays (JSONs)**: 138.138
- **Total Ingerido pelo Banco**: 133.170
- **Déficit (Fragmentação Silenciosa)**: 4.968 partidas órfãs no limbo do disco.

Estas partidas nunca chegaram à tabela `matches` devido a interrupções de rotina ou falhas silenciosas de ingestão passadas.

## 2. A Topologia Real de Ingestão (Desconstrução dos Entrypoints)
A documentação de CLI (textos de `--help`) e nomes abstratos de scripts geram *wishful thinking*. A análise profunda da Árvore Sintática (AST) do projeto definiu o mapa exato para a manutenção segura de dados em tempos de estrangulamento de rede (*Kaggle Rate Limits*).

### A) A Falsa Promessa Offline: `rebuild_db.py`
O script nativo `tcg-rebuild-db` alega operar em isolamento local absoluto (gerando uma *staging area* e promovendo atomicamente). Entretanto, na linha `308`, ele importa diretamente e autentica `KaggleApi`, acionando `competition_leaderboard_download`.
Além de realizar chamadas de rede indesejadas (que causam o *crash* do sistema sob restrição de *Rate Limit*), ele destrói e remonta iterativamente o banco de dados inteiro. Não é uma ferramenta de sincronização; é um motor de aniquilação e recriação estrutural.

### B) O Vetor Agressivo de I/O: `data_manager.py`
Acionado por `tcg-data`, este é puramente o encarregado da *pipeline* de *download*. Faz chamadas sequenciais brutas à API do Kaggle. Altamente reativo a flutuações de rede e bloqueios do sistema anti-abuso. Executá-lo quando já possuímos a base bruta é um desperdício imperdoável de tokens de API.

### C) O Motor de Ingestão Delta Idempotente: `build_card_stats.py`
Contra-intuitivamente, a função responsável por sincronizar incrementalmente os blocos que faltam (*sync* puro) reside no script que teoricamente computa as estatísticas de cartas (`tcg-build-card-stats`).
A função `populate_replays()` é estritamente **Idempotente** e isolada da rede. Ela varre todos os 138 mil arquivos JSON localmente e confronta seus `external_episode_id` contra o banco. Se detectar redundância, *skippa* o loop. Se detectar uma falha na tabela (como os 4.968 nós faltantes identificados no déficit), insere cirurgicamente os blocos ausentes sem corromper a malha analítica pré-existente e sem emitir um único `HTTP GET` pro Kaggle.

## 3. Postura Operacional (The Zero-Trust Model)
Toda futura avaliação volumétrica ou normalização heurística está subordinada à Auditoria Física. Scripts de manipulação de disco (`tcg-*`) nunca devem ser engatilhados tendo por base argumentos extraídos por `argparse` em terminais isolados, mas sim por leitura direta do backend para mapeamento de dependências de internet. O SQLite é espelho derivado; o `.json` é a verdade primária.
