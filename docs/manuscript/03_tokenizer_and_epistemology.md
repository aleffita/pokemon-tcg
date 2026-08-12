# Capítulo 3: Abolição Espacial, Tokenizer e Sinalização Epistemológica

A modelagem de um ambiente de cartas competitivo impõe um desafio geométrico severo: diferentemente de um texto estruturado de Linguagem Natural, a mesa de um jogo de Pokémon TCG não obedece a uma lógica adjacente. A segunda carta da sua mão não interage dimensionalmente mais com a primeira carta da mão do que com a quinta. São estruturas independentes, agrupadas por "Zonas".

## 3.1. A Invariância de Permutação e a Morte do RoPE
Modelos modernos como LLaMA 3 e arquiteturas GPT dependem vitalmente de matrizes Posicionais Rotacionais (RoPE) para injetar entropia linear (quem vem antes de quem). 

Na arquitetura instanciada no `policy_mlx.py`, **o RoPE foi obliterado.** O vetor recebe a posição da sequência como um tensor mudo. Em seu lugar, a topologia invoca uma projeção discreta de **19 Embeddings Espaciais (Type Embeddings)**. 
Cada carta recebe um tensor associativo dependendo puramente da região onde ela existe na matriz física do ambiente (`T_SELF_HAND`, `T_OPP_ACTIVE`, `T_STADIUM`, `T_SELF_BENCH_1...5`, `T_OPP_DISCARD`, etc.). Esta fundação é estritamente **Set-Based** (Baseada em Conjuntos). A cabeça de Atenção interage rotacionalmente buscando alinhamentos no hiperplano espacial, de forma completamente agnóstica à posição bruta no array numérico $N \times 128$.

## 3.2. A Engenharia de Vórtices (Agregação de _unit_stream_)
Um Pokémon na Arena não é apenas uma Carta. Ele é uma Carta Base, possivelmente ancorando Pré-Evoluções, uma Ferramenta anexa, Múltiplas Energias, e estados transitórios (Dano Acumulado, Envenenamento).

Se o *Tokenizer* isolasse essas entidades em $10$ tokens distintos, o processamento de Relevância Cruzada (*Cross-Attention*) de um ataque exigiria mapeamento hiper-distribuído, exaurindo a capacidade atencional das 4 Camadas $D=128$.
A resolução se dá no `_unit_stream`. O *TokenEncoder* condensa essas instâncias numa Equação Aditiva Pura:
$$ \mathcal{U}_{\text{Vortex}} = \text{Base}_{\text{emb}} + \sum_{i=1}^{n} \text{PreEvo}_{\text{emb}} + \text{Tool}_{\text{emb}} + \sum_{j=1}^{e} \text{Energy}_{\text{emb}} + \text{UnitProj}(\text{Dmg, Status}) $$
Esta redução espacial transforma um Pokémon completo e complexo em um **Único Token Denso**. O *Attention Head* passa a enxergar uma Unidade Bélica coesa, e não peças de Lego espalhadas.

## 3.3. Sinalização Epistemológica e o *GameTracker* Bayesiano
O coração intelectual da vantagem do nosso sistema repousa em sua resolução de **Informação Imperfeita**. O oponente retém dados ocultos, logo, o modelo não enxerga a mesa inteira. 

O `GameTracker` age como um simulador dedutivo assíncrono. Em vez de simplesmente ocultar o que não se vê, ele projeta sombras atencionais (Embeddings Aditivos) que sinalizam níveis de Certeza à rede.

1. **`drawable_emb` e `opp_drawable_emb`:** O motor confronta a lista de cartas revelada pelo *Deck* com a presença pública na mesa/descarte. Quando uma cópia de uma carta some (ou nunca foi vista), o vetor `drawable_emb` é acoplado aos *tokens* de Baralho correspondentes. A rede não sabe que a carta está na mão, mas a Atenção recebe o vetor: *"É matematicamente possível que o inimigo compre ou já tenha esta carta"*.
2. **`hand_certain_emb` e Quebras de Cegueira:** Quando o ambiente reporta o movimento imperfeito de cartas `Type 7 (MoveCardReverse)`—ou seja, o oponente transferiu algo da zona pública para a mão ou comprou sem revelar a face—o *Tracker* contabiliza uma fratura informacional (*blind exit*). Se o inimigo exibe uma carta cujo rastro de entrada foi inequivocamente mapeado, o *Token* na mão ganha o tensor **`hand_certain_emb`**. Quando o *blind exit* polui a zona, o tensor é removido. Assim, a matriz de Atenção pode segregar fisicamente "Certezadas Táticas" de "Possibilidades Assumidas", protegendo o núcleo decisório de blefes não-intencionais (Alucinações Táticas).
