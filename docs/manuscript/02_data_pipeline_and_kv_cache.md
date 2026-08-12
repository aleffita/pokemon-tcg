# Capítulo 2: Engenharia de Limpeza de Dados e Otimização Unificada (KV Cache)

O processo de ingestão de dados brutos do ambiente Kaggle Environments apresenta vulnerabilidades críticas de estado. Partidas incompletas, desconexões e estados corrompidos injetam ruído fatal na rede atencional. Para mitigar isso, o modelo arquitetural emprega o `build_bc_dataset.py` como um funil rigoroso de purificação semântica antes de qualquer serialização Parquet.

## 2.1. O Processo de Triagem e Data Cleaning
O método `rows_from_episode` implementa as primeiras barreiras físicas:
1. **Filtro de Desfechos Degenerados:** Replays cuja soma final resulta em empate (`rewards[0] == rewards[1]`), ausência de recompensa ou deformação nos dicionários locais são abortados imediatamente. O classificador exige dados estritamente binários na resolução final para que o *Reward Signal* não seja contaminado por indefinições termodinâmicas (abandonos de partida).
2. **Deflexão Off-By-One:** O Kaggle Environments possui um *delay* de índice onde a ação efetuada pelo Agente para um observável em $t$ é registrada no nó $t+1$. A arquitetura desvia e realinha os ponteiros garantindo que a máscara de opções (`action_mask`) da observação pareie cirurgicamente com o rótulo (`label`) da decisão validada, filtrando ruído e eliminando o erro histórico de pareamento incorreto que gerava $\sim 18\%$ de perda de validade técnica.

## 2.2. Telescoping Backward Rewards: O Cálculo Autoritário de Prêmios
Durante a decodificação da partida, o *pipeline* não confia na simulação intrínseca de tensores para contar Prêmios tomados (o que poderia desincronizar em casos de erro do observável). Em vez disso, o `_decision_prize_states` varre a lista crua de jogadores e audita os dicionários originais no log físico.

Para cada decisão isolada, o `_compute_aux_targets` injeta metadados exatos baseados na delta de alteração daquele exato ponto. Há uma distinção crucial na topologia dos tensores auxiliares:
- **`aux_prize_delta` e `aux_ko`:** Atuam como alvos *Locais-Para-O-Turno*. A rede avalia: "De agora até o final deste mesmo turno, quantos prêmios colherei subtraindo os prêmios colhidos pelo oponente?". Eles se repetem para múltiplas ações em uma mesma cascata de combos, pois o alvo tático do combo inteiro culmina naquele prêmio.
- **`aux_return` e `reward`:** Para evitar dupla-contagem letal na regressão (que causaria colapso estelar inflando artificialmente o *Reward*), o sistema calcula uma verdadeira transição em janela deslizante (Delta de transição de decisão para a *próxima* decisão válida). Assim, o valor acumulado no decorrer da partida (*Return* progressivo) é matematicamente limpo:

$$ \text{Reward}_{t} = \frac{(\text{Prêmios\_Me\_}_{t} - \text{Prêmios\_Me\_}_{t+1}) - (\text{Prêmios\_Opp\_}_{t} - \text{Prêmios\_Opp\_}_{t+1})}{6.0} + \text{Term}_{bonus} $$

Onde $\text{Term}_{bonus}$ adiciona $+1.0$ (Vitória) ou $-1.0$ (Derrota) apenas na borda terminal.

## 2.3. O Desafio da Memória Unificada e o Parquet KV Cache
Treinar uma matriz de tensores 128-dim com *Truncated Backpropagation Through Time* (TBPTT) sobre blocos de milhares de episódios no hardware Apple Silicon (M3 Pro 24GB Unified Memory) precipita uma explosão letal de *I/O*. Em abordagens ingênuas, reler o disco para construir lotes espaciais (onde os passos de $t$ a $t+N$ precisam estar encadeados) força o SO a realizar *Thrashing* de SSD (Paginação de Memória Excessiva).

A solução magistral implementada no `bc_train_mlx.py` foi o **`_ParquetRowGroupCache`**. 
Esta classe emula com exatidão um *KV Cache* hierárquico usado em inferências gigantescas de *Large Language Models* (LLMs), mas aplicado à leitura do Dataset:
- **Zonas Termodinâmicas:** O cache mantém Dicionários de Blocos segmentados em *Hot Zone* (itens correntemente usados pelo *worker* iterador) e *Transient Zone* (itens recém saídos, em LRU). 
- **Blocos Row-Group:** Como o formato `.parquet` arquiva matrizes verticalmente (Colunar) agrupadas em sub-lotes (*Row-Groups*), o cache lê um bloco inteiro, decodifica para arrays NumPy de alta densidade e o fixa no *Hot Zone*.
- **O Eviction Tier (Spill):** Para evitar congelamento de *Threads*, quando o limite massivo (10GB) é atingido, o Cache despeja dados para um disco transitório `.cache_spill/` e avança o LRU de forma granular.

Essa estrutura permite que as *Lanes* do Dataloader requisitem linhas adjacentes em $O(1)$ sem engasgos de disco, possibilitando que o vetor `memory_in` (Scratch Registers) deslize pelo mini-lote conectando organicamente cada nó da partida sem pausas. Sem o `_ParquetRowGroupCache`, treinar o TBPTT na velocidade de milhares de exemplos por minuto no M3 Pro seria fisicamente impossível.
