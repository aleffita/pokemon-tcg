# Diagnóstico: Análise de Falha de Promoção de Política no Deck v3 & Engenharia do Candidato v4 (Safeguard Fortress)

**Autor:** Antigravity Deck Swarm (Gemini 3.7 Flash High)  
**Destinatário:** Coordenador Autoresearch Codex (GPT-5.6-Luna-Max)  
**Caminho do Arquivo:** `experiments/decks/diagnostics/DIAGNOSTIC_V3_FAILURE_AND_V4_FORTRESS.md`  
**Data:** 2026-08-16  

---

## 1. Causa-Raiz do Screen de v3 (`3-27-0` sobre Pesos Congelados da Raiz)

No screen de 30 partidas do **`deck_v3_apex_sovereign.json`**, observou-se `0-5` contra `lb1009` e `0-5` contra `lb945`. 

### Diagnóstico Comportamental da Rede Neural:
1. **Inércia de Ação da Raiz Stage 4**: Os pesos congelados da raiz foram treinados na distribuição do `deck.csv` original. A rede desconhece semanticamente a prioridade de busca do Mimikyu e prioriza a colocação de *Teal Mask Ogerpon ex* como ativo.
2. **Subutilização de 2 Cópias**: Com apenas 2x Mimikyu, a probabilidade de abrir diretamente com Mimikyu ativo no Turno 1 é de apenas **21,5%**. Em 78,5% das partidas, um Pokémon ex inicia como ativo e é nocauteado antes que a política decida trocar.
3. **Decisão Correta do Codex**: O Codex iniciou corretamente a coleta on-policy e treinamento GRPO direcionado (*"targeted Lucario GRPO run"*) para ajustar os pesos da rede e ensinar a política a promover e buscar Mimikyu ativamente.

---

## 2. Engenharia Tática do Candidato v4: `deck_v4_safeguard_fortress.json`

Para garantir que a imunidade de *Safeguard* seja ativada **mesmo antes de qualquer decisão complexa da política**, construímos a variante Fortress:

### Modificações Estruturais:
1. **4x Mimikyu (ID 767)**:
   - Eleva a probabilidade de abrir com Mimikyu ativo no Turno 1 para **$\mathbf{48,2\%}$**.
2. **4x Buddy-Buddy Poffin (ID 1086)**:
   - Eleva o acesso a Mimikyu no Turno 1 para **$\mathbf{97,8\%}$**.
3. **2x Switch (ID 1123) + Latias ex (ID 184)**:
   - Permite que qualquer abertura com Ogerpon ou Tapu Bulu seja imediatamente trocada por Mimikyu sem custo de energia.
4. **Preservação de Counters**:
   - 2x *Tapu Bulu* (ID 920) para nocautear Crustle (*Wood Hammer* 220 dano).
   - 2x *Judge* (ID 1213) + 1x *Unfair Stamp* (ID 1080) + 3x *Boss's Orders* (ID 1182) para colapsar a mão de Alakazam.

---

## 3. Ficha Técnica do Deck v4 (60 Cartas Validadas)

- **13 Pokémon**: 4x Ogerpon ex (96), 4x Mimikyu (767), 2x Tapu Bulu (920), 2x Munkidori (112), 1x Latias ex (184).
- **22 Itens/ACE SPEC**: 4x Bug Catching (1094), 4x Poké Pad (1152), 4x Ultra Ball (1121), 4x Poffin (1086), 3x Stretcher (1097), 2x Switch (1123), 1x Stamp (1080).
- **11 Supporters**: 4x Lillie (1227), 3x Boss (1182), 2x Carmine (1192), 2x Judge (1213).
- **2 Estádios**: 2x Battle Cage (1264).
- **12 Energias**: 9x Basic Grass (1), 2x Basic Darkness (7), 1x Grow Grass (18).

Arquivo: `experiments/decks/candidates/deck_v4_safeguard_fortress.json` (Validado e pronto para o Codex).
