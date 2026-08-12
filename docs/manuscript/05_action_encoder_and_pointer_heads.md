# Capítulo 5: O Sistema Atencional Pointer-Network e a Compactação MLX

Em ambientes estáticos como Xadrez ou Go, a cabeça de decisão (Policy Head) atua como um `Softmax` sobre um vetor de tamanho imutável. No Pokémon TCG, o espaço fractal de possibilidades dita que cada instante $t$ apresenta um conjunto diferente de Opções Legais. O sistema adota a geometria *Pointer-Network* (Redes de Ponteiros), onde o modelo constrói fisicamente o vetor da Opção e o avalia dinamicamente contra o estado global.

## 5.1. A Anatomia do Construto de Opção (`_opt_stream`)
Ao invés de ler uma ID estéril "Ação #409", o modelo forja uma identidade semântica 128-dim para o movimento legal somando suas partes constituintes:
- **As Pontes Topológicas (`opt_src_proj` e `opt_tgt_proj`):** A rede busca a coordenada espacial exata do Agente causador (Source) e o Agente afetado (Target). Se a ação for ligar energia a um benched, o *Source* carrega o tensor da carta de Energia, o *Target* carrega o Pokémon Benched.
- **Identidade de Ação (`opt_verb_emb`):** Um dicionário embutido ($16 \times 128$) codificando o Verbo semântico ("Atacar", "Jogar Ferramenta", "Evoluir", "Recuar").
- **A Resolução de Colisões (`attack_emb`):** Para ataques idênticos no Dano mas distintos nos Efeitos Ocultos, a abstração falha. A arquitetura concatena a ID do ataque canônico (`MAX_ATTACK, 128`) garantindo que ataques nominais idênticos sejam distinguidos fisicamente.
A soma destas matrizes gera um hiper-tensor que é avaliado individualmente pela cabeça de Decisão (`opt_head`).

## 5.2. O Algoritmo de Sub-Máscaras e Compactação Otimizada (`_OPT_BUCKETS`)
O Flash Attention atinge sua entropia mínima de processamento com tensores densos (quadrados, sem espaços vazios). Porém, durante as partidas, uma árvore de decisão pode ter 5 opções legais, enquanto outra exige 150 opções (ex: procurar no Deck de 60 cartas).

Submeter um tensor $(B, 192, 128)$ preenchido com $90\%$ de máscara vazia (*Padding*) estrangula a *Memory Bandwidth* do Apple Silicon. O núcleo MLX introduz uma compactação adaptativa:
1. O *Encoder* C++ varre as opções e condensa as repetidas via `_dedup_group` (apagando duplicatas legais inúteis, como jogar cópias idênticas da mesma energia no mesmo alvo, preservando apenas o representante canônico).
2. O Motor de Inferência implementa Buckets restritos: `(32, 64, 128, 192)`. Ele aloca o *Flash Attention* na **menor máscara que caiba nas ações legais**.
Esta dinâmica economizou $\sim 47\%$ de *VRAM* durante os microbatches espaciais no M3 Pro, destravando a viabilidade computacional do TBPTT para $B=128, T=16$.

## 5.3. Cisão de Decisão (Ações Explícitas vs. SUBMIT)
Um lapso grave em motores convencionais é considerar "Passar o Turno" como uma ação adjacente aos ataques. Estruturalmente, finalizar uma sub-ação não pertence ao mesmo espaço semântico que invocar cartas. 
A geometria MLX usa `split_heads = True`: 
- A classe de `SUBMIT_ACTION` não compõe o `_opt_stream`. Ela nasce do próprio Vórtice Global da Mesa (`cls_out`). O modelo compara se o estado atual do *board* já alcançou o pico de otimização termodinâmica; se sim, ele aciona o `submit_tok`, encerrando a árvore de combos e transferindo a prioridade para o oponente.
