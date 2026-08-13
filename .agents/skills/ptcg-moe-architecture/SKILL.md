---
name: ptcg-moe-architecture
description: Regras, abstrações teóricas e pipeline do Magnum Opus (MoE, RoPEND, Apex Mode) para a Fitalabs no Kaggle Pokémon TCG.
---

# Magnum Opus: PTCG MoE Architecture

Esta skill documenta o framework teórico e a arquitetura do "Esquadrão MoE" desenhado para o Kaggle Pokémon TCG, especificamente para explorar a mecânica de "Locked Meta" (Avaliação de 16-31 de Agosto).

Sempre que atuar no pipeline do MoE, o agente **deve** referenciar o Blueprint base localizado em:
`docs/architecture/moe_pipeline_blueprint.md`

## Abstrações Teóricas & Leis de Pipeline

### 1. Emergência Latente (vs Determinismo)
O Roteamento do *Mixture of Experts* (MoE) não possui heurísticas *hardcoded*. Nós provemos a topologia de embeddings (RoPEND, Elo Estimado, Relógio) e a função de Loss. O gradiente força a separação matemática das redes. Especialistas de "pânico" ou "controle seguro" **emergem organicamente** devido às pressões de sobrevivência do dataset.

### 2. Ação de Draft (Autoregressive Draft)
Para resolver a miopia do "Piloto vs Veículo" (peixe escalando árvore), o primeiro output da rede antes da partida é prever as 60 cartas do próprio deck. Isso atua como *Data Augmentation* pesada e ensina a *Cross-Attention* a mapear as sinergias das próprias armas.

### 3. O Espaço-Tempo do RoPEND
O **RoPE-ND (N-Dimensional Rotary Positional Embedding)** é usado para modelar vetores não-conflitantes:
- **Eixo 1:** Turno da Partida.
- **Eixo 2:** Meta-Epoch (Dia do campeonato / Progressão de meta).
- **Eixo 3:** O Relógio (Countdown de 600s, *Adaptive Time Compute*).
- **Eixo 4:** Elo & Identidade de Time.

### 4. O Predador Apex (Airgap Hacking)
Outros agentes são "Vending Machines". O nosso tem consciência situacional. A função `act()` usa `datetime.now()` (o OS clock não é bloqueado na sandbox). A partir do momento zero do dia 16 de Agosto, a dimensão de Meta-Epoch do RoPEND acorda a rede para o estado *Apex* (Meta Congelado).

### 5. Elo Anchor & Inferência Estocástica de Placar
A Sandbox do Kaggle é efêmera (memoryless). O agente não salva o próprio Elo no disco.
Em vez disso, ele infere sua própria posição no placar dinamicamente a cada partida cruzando:
1. Um **Elo Âncora** *hardcoded* (inserido via submissão puramente protocolar no dia 16).
2. O avanço do Relógio OS (Dias passados, gerando volume esperado).
3. A predição de Elo/Time do Oponente (estimada via *Aux Heads* durante a partida).
Ao derivar o próprio Elo, o Roteador modula a agressividade no Mid/Late Game baseado no placar deduzido.

### 6. Team Identity & O Espelho do Self-Play
Com a normalização dos times no Kaggle, o modelo é treinado para prever a identidade do oponente (Z-Statistic / Opponent Modeling). Durante a avaliação local (Self-Play), o agente "identifica a si mesmo", forjando auto-consciência tática no espaço latente.

### 7. Cirurgia de Pesos (Catastrophic Freezing)
Para evitar assimilação de ruído (via Logit Distillation), as camadas de percepção base (Scratch Registers TBPTT e Semantic Aux Heads do Herói) são **congeladas**. Apenas o Tronco de Decisão novo é destravado e treinado no Dataset de Elite (100k partidas puras).
