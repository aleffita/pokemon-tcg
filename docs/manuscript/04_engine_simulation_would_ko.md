# Capítulo 4: Oráculos Termodinâmicos e a Matriz Simulatória (`would_ko`)

Uma das maiores armadilhas no *design* de agentes de *Reinforcement Learning* para domínios regidos por leis aritméticas explícitas (como acúmulo de dano e cálculos de pontos de vida HP) é forçar a arquitetura atencional a "deduzir" matemática. Empregar matrizes de projeção para resolver aritmética básica consome bilhões de *flops* latentes que poderiam estar sendo investidos em raciocínio tático de longo prazo.

Para erradicar a dependência da rede atuar como uma calculadora defeituosa, a infraestrutura introduziu a delegação heurística através da matriz `would_ko`.

## 4.1. O Conflito do Não-Determinismo e o Motor de Inferência
No Pokémon TCG, ataques sofrem metamorfoses complexas entre o Atacante e o Defensor. Dano Base, Fraqueza, Resistência, Ferramentas de imunidade transitória e, criticamente, ataques dependentes de Rolagens Estocásticas (Moedas).

A predição analítica (se $\text{Dano\_Base} > \text{HP}_{\text{Opp}}$) é fatalmente insuficiente. A solução instaurada opera delegando o cálculo ao próprio motor C++ interno (`engine`) antes da passagem da matriz. 
Quando o observável identifica Opções Legais do tipo "Ataque", a diretiva `bc_would_ko=True` aciona o método `annotate_would_ko_with_audit` (via `search_agent.py`).

## 4.2. A Auditoria Monte Carlo $N=10$
Ao invés de uma aproximação rasa, o *pipeline* gera um micro-universo determinístico. O motor submete o ataque a **10 Variações Monte Carlo** (`WK_NVAR = 10`), executando o ataque virtualmente 10 vezes em clones do estado do jogo (absorvendo e resolvendo todos os lances de moeda e regras de tabuleiro).

O resultado converge em uma auditoria estrita que expõe 3 matrizes numéricas (heurísticas injetadas de volta no vetor):
1. **`would_ko` (Letalidade Absoluta):** Probabilidade empírica (0.0 a 1.0) de o ataque remover o Defensor em $t$.
2. **`would_ko_prizes` (Lucro Esperado):** Quantidade de prêmios calculada após o *resolve* da morte (distingue abater um Básico vs uma VMAX/EX de 3 prêmios).
3. **`would_ko_win` (Terminação Imediata):** Binário $1/0$, sinalizando se o vetor final daquele ataque encerra a árvore do jogo no ato.

## 4.3. Fusão Tensor-Semântica no Pointer-Network
Essa tríade de oráculos matemáticos não é meramente anexada. Ela é projetada fisicamente no espaço da Ação (`opt_attr_proj` nas camadas Pointer-Network). 

A matriz final de Opção que a *Attention Head* avalia não precisa aprender o que é fraqueza. Ela lê um tensor que diz *"A Opção X tem 100% de chance de matar e garantir 2 prêmios"*. 
A rede Neural passa então a operar em Nível Macro: ela decide se é melhor matar agora e ficar vulnerável no próximo turno, ou sacrificar uma peça menor para segurar o tabuleiro. A abstração de cálculos de micro-estado permitiu à IA ascender à estratégia macroeconômica da partida.
