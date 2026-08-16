# Original User Request

## 2026-08-16T18:57:31Z

Engenharia tática e adversarial de um deck fechado de exatamente 60 cartas para o Kaggle Pokémon TCG AI Challenge, maximizando a taxa de vitória e a robustez invariante durante o período de avaliação da ladder congelada (16 a 31 de Agosto de 2026), com integração formal ao protocolo de autoresearch do Codex (GPT-5.6-Luna-Max).

Working directory: /Users/alefita/workdir/pokemon-tcg
Integrity mode: development

## 1. Contexto & Divisão de Soberania
- Motor Central do Codex (GPT-5.6-Luna-Max): Coordenador Chefe de Engenharia Neural, responsável pelo treinamento GRPO, coleta de trajetórias on-policy e execução de torneios locais.
- Enxame Antigravity (Gemini 3.7 Flash High & Subagentes): Conselho Tático e Engenharia Combinatória de Decks (60 cartas).
- Restrição Rígida de Hardware: ZERO uso de GPU/MPS/Metal e ZERO processos de treinamento concorrentes. 100% dos recursos computacionais da máquina (Apple Silicon M3 Pro) permanecem dedicados aos experimentos do Codex. O trabalho do enxame é estritamente cognitivo, combinatório e de consultas SQL em modo read-only.

## Requirements

### R1. Mineração de Dados SQLite & Análise de Meta
- Consultar model/results.db em modo estritamente read-only (tabelas decks, deck_cards, match_card_usage e deck_elo_daily).
- Extrair a composição canônica completa do Deck #633 (Yan / Teal Mask Ogerpon ex - 27.9% WR) e do Deck #251 (12.9% WR).
- Identificar as cartas individuais e combinações com maior correlação positiva de vitória em partidas de Elo >= 1100.0.

### R2. Modelagem Hipergeométrica & Curva de Recursos
- Calcular a distribuição multivariada de probabilidade da mão inicial (7 cartas + compra do Turno 1).
- Garantir matematicamente:
  - P(Pokémon Básico Ativo no Turno 1) >= 92%.
  - P(Mulligan) <= 8%.
  - Sustentação da aceleração de energia para ataques efetivos a partir do Turno 2.
- Otimizar a proporção exata entre Pokémon Básicos, Evoluções, Itens de Busca (Nest Ball, Ultra Ball, etc.), Apoiadores de Compra (Professor's Research, Iono, etc.) e Energias Básicas/Especiais.

### R3. Estresse Adversarial (Red Team) contra o Painel de Oponentes
- Modelar e blindar o deck contra os 6 arquétipos do painel externo identificados nos experimentos AR-019 a AR-025 do Codex:
  1. lb826_alakazam_seok (controle, punição de energias e fixação de dano).
  2. lb1009 e lb945 (agressão rápida de topo de leaderboard).
  3. lb814, Lucario e Dragapult (spread de dano, aceleração e bench snipes).
  4. Baselines internos e first_sub_kaggle_2707.
- Simular rotas de saída para cenários de pior caso: disrupção de mão via Iono/Judge para 1-2 cartas, travamento no ativo via Boss's Orders sem energia de recuo, e desvantagem de fraqueza elemental.
- Incorporar atacantes de 1 prêmio para otimizar o Prize Trade contra Pokémon ex de 2 prêmios.

### R4. Síntese, Emissão de Artefatos & Notificação ao Protocolo do Codex
- Consolidar o consenso do quórum em uma lista fechada de exatamente 60 Card IDs válidos.
- Escrever a monografia técnica detalhada em experiments/decks/DECK_SUPREME_60.md.
- Gerar a cápsula de deck em experiments/decks/deck_supreme_60.json e atualizar agent/deck.json.
- Notificar o coordenador do Codex através do sistema de arquivos (read-this-agent/08_DECK_SWARM_PROTOCOL.md e experiments/decks/), permitindo ingestão direta nos torneios de self-play e GRPO.

## Acceptance Criteria

### Integridade Estrutural do Deck
- [ ] O arquivo agent/deck.json contém exatamente 60 números inteiros válidos, correspondendo a Card IDs existentes no SQLite.
- [ ] O arquivo experiments/decks/deck_supreme_60.json está formatado com metadados de arquétipo, curva de energia e probabilidades calculadas.

### Rigor Matemático & Documentação
- [ ] O documento experiments/decks/DECK_SUPREME_60.md detalha a justificativa de cada um dos 60 slots, a prova formal hipergeométrica (P(Setup) >= 92%), e a matriz de matchup contra os 6 oponentes do painel.

### Protocolo & Zero Contenção
- [ ] O protocolo em read-this-agent/08_DECK_SWARM_PROTOCOL.md está sincronizado com a localização e hashes do deck gerado.
- [ ] Nenhuma GPU/MPS foi alocada e nenhum processo concorrente de treino foi executado, preservando 100% do hardware para o Codex.
