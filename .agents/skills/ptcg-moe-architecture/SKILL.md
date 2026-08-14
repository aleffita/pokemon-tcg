---
name: ptcg-moe-architecture
description: Regras, abstrações teóricas e pipeline do Magnum Opus (MoE, RoPEND, Apex Mode) para a Fitalabs no Kaggle Pokémon TCG.
---

# Magnum Opus: PTCG MoE Architecture & Apex Mode

Esta skill documenta o framework teórico e a arquitetura do "Esquadrão MoE" desenhado para o Kaggle Pokémon TCG, especificamente para explorar a mecânica de "Locked Meta" (Avaliação de 16-31 de Agosto de 2026).

Sempre que atuar no pipeline do MoE, referencie os blueprints mestres:
- `docs/architecture/moe_pipeline_blueprint.md`
- `docs/architecture/01_ropend_theory.md`
- `docs/architecture/02_stochastic_elo_inference.md`
- `docs/neural_engine_and_tokenization_spec.md`

---

## 1. O Paradoxo do Piloto vs. Veículo
O modelo neural não é avaliado no vácuo. Um modelo com alta acurácia de validação (~78%) mas baixo win rate (~17%) não é falho: é um mestre piloto atrelado a um veículo (baralho) com teto termodinâmico inferior. A arquitetura MoE concede auto-consciência tática ao modelo para adaptar seu estilo de pilotagem ao baralho e ao oponente.

## 2. O Espaço-Tempo do RoPEND (4D Rotary Positional Embeddings)
O RoPE-ND particiona a dimensão de embedding $D=128$ em 4 sub-vetores ortogonais de 32 dimensões:
- **Eixo 1 ($c_1$):** Turno / Passo da partida ($0 \dots 200$).
- **Eixo 2 ($c_2$):** Meta-Epoch (Dia do campeonato / Progressão de meta).
- **Eixo 3 ($c_3$):** Relógio de Urgência (Countdown normalizado de 600s).
- **Eixo 4 ($c_4$):** Elo Contínuo & Identidade de Time.

## 3. Autoregressive Draft (Modelagem de Sinergia do Veículo)
Antes de emitir a primeira ação no passo 0, a rede executa uma passada auto-regressiva de atenção sobre a lista de 60 cartas do próprio baralho. Isso ensina a Cross-Attention a mapear as sinergias internas do veículo antes do combate.

## 4. O Predador Apex (Airgap Activation)
Durante a inferência em produção (`act()`), o modelo consulta `datetime.now(timezone.utc)`. 
A partir de **16 de Agosto de 2026**, a ativação do token *Apex Mode* altera a distribuição do roteador MoE de exploração para contra-ataque predatório determinístico no Meta Congelado.

## 5. Inferência Estocástica de Elo em Sandboxes Efêmeras
A sandbox do Kaggle não possui persistência em disco entre partidas. O agente deriva sua posição no ranking cruzando:
1. O Elo Âncora de 16 de Agosto ($R_0$)
2. Dias decorridos ($\Delta T$)
3. A predição de Elo do oponente via cabeça auxiliar ($\hat{R}_{\text{opp}}$)

## 6. Transição de Modelo Base
O modelo legado "Herói" não deve sofrer *Logit Distillation* direta. A esteira correta exige o treinamento de um **Novo Modelo Base V2** (ou refinamento a partir do Stage 4 FP32) na topologia RoPEND, servindo de alicerce para a expansão dos especialistas MoE e alinhamento por GRPO.
