---
name: ptcg-moe-architecture
description: Regras, abstrações teóricas e pipeline do Magnum Opus (MoE, RoPEND, Apex Mode) para a Fitalabs no Kaggle Pokémon TCG.
---

# Magnum Opus: PTCG MoE Architecture

Esta skill documenta o framework teórico e a arquitetura do "Esquadrão MoE" desenhado para o Kaggle Pokémon TCG, especificamente para explorar a mecânica de "Locked Meta" (Avaliação de 16-31 de Agosto).

Sempre que atuar no pipeline do MoE, o agente **deve** referenciar o Blueprint base localizado em:
`docs/architecture/moe_pipeline_blueprint.md`
As matemáticas fundamentais estão nas documentações anexas (`01_ropend_theory.md` e `02_stochastic_elo_inference.md`).

## Abstrações Teóricas & Leis de Pipeline

### 1. Emergência Latente (vs Determinismo)
O Roteamento do *Mixture of Experts* (MoE) não possui heurísticas *hardcoded*. Nós provemos a topologia de embeddings (RoPEND, Elo Estimado, Relógio) e a função de Loss. O gradiente força a separação matemática das redes. Especialistas de "pânico" ou "controle seguro" **emergem organicamente** devido às pressões de sobrevivência do dataset.

### 2. Ação de Draft (Autoregressive Draft)
O primeiro output da rede antes da partida é prever as 60 cartas do próprio deck. Isso atua como *Data Augmentation* pesada e ensina a *Cross-Attention* a mapear as sinergias das próprias armas.

### 3. O Espaço-Tempo do RoPEND
O **RoPE-ND** modela vetores não-conflitantes:
- **Eixo 1:** Turno da Partida.
- **Eixo 2:** Meta-Epoch (Dia do campeonato / Progressão de meta).
- **Eixo 3:** O Relógio (Countdown de 600s, *Adaptive Time Compute*).
- **Eixo 4:** Elo & Identidade de Time.

### 4. O Predador Apex (Airgap Hacking)
A função `act()` usa `datetime.now()`. A partir de 16 de Agosto, a dimensão de Meta-Epoch do RoPEND acorda a rede para o estado *Apex* (Meta Congelado).

### 5. Elo Anchor & Inferência Estocástica de Placar
A Sandbox é efêmera. O agente infere sua própria posição no placar cruzando um Elo Âncora *hardcoded*, o Relógio OS (Dias passados), e a predição de Elo do Oponente. O Roteador modula a agressividade no Mid/Late Game baseado no placar estocástico.

### 6. Team Identity & O Espelho do Self-Play
Com a normalização dos times no Kaggle, o modelo é treinado para prever a identidade do oponente. No Self-Play local, o agente "identifica a si mesmo", forjando auto-consciência.

### 7. Treinamento do Modelo Base (Rejeição do Legado)
O modelo "Herói" é de uma arquitetura legada incompatível. Não faremos *Logit Distillation* ou *Surgical Freezing* nele. A esteira correta exige o treinamento de um **Novo Modelo Base** na nova arquitetura (partindo do zero ou do Curriculum V1 Stage 4), que após validado, servirá de alicerce para a expansão do MoE.
