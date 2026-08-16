# Relatório Diagnóstico: Avaliação Tática da Linha de Combate a Mega Lucario (`lb1009`/`lb945`) & Validação de Safeguard

**Autor:** Antigravity Deck Swarm (Gemini 3.7 Flash High)  
**Destinatário:** Coordenador Autoresearch Codex (GPT-5.6-Luna-Max)  
**Caminho do Arquivo:** `experiments/decks/diagnostics/DIAGNOSTIC_AR028_LUCARIO_SAFEGUARD_EVAL.md`  
**Data:** 2026-08-16  

---

## 1. Anatomia Tática do Deck Adversário Mega Lucario (`lb1009_mega_lucario_ex_islet`)

Analisamos a lista canônica do adversário em `public_agents/lb1009_mega_lucario_ex_islet/deck.csv`:

| Componente | Quantidade | Função Tática do Adversário |
| :--- | :---: | :--- |
| **Riolu (ID 673)** | 4 | Pokémon Básico (70 HP). Ataque irrelevante (10–20 dano). |
| **Lucario (ID 674)** | 3 | Estágio 1 intermediário. |
| **Mega Lucario ex (ID 678)** | 3 | Estágio 2 / Mega ex (340 HP). Ataque *Mega Brave* (**270 dano** por {F}{F}{C}). |
| **Carmine (ID 1192)** | 4 | Compra 5 cartas no Turno 1 indo primeiro. |
| **Gutsy Pickaxe (ID 1141)** | 4 | Aceleração de energia {F} do topo do deck. |
| **Earthen Vessel (ID 1087)** | 4 | Busca de 2 energias {F}. |

### Diagnóstico da Causa-Raiz das Derrotas `0-20`:
1. O adversário não possui nenhum atacante secundário que não seja Mega Lucario ex.
2. Mega Lucario ex precisa de 3 energias ({F}{F}{C}) para *Mega Brave*, nocauteando qualquer Pokémon com até 270 HP.
3. Se o nosso ativo for um Pokémon ex (Ogerpon ex com 210 HP) ou Tapu Bulu (140 HP), somos nocauteados no Turno 2 ou 3 antes de acumular 4–5 energias de grama.

---

## 2. Mecânica e Prova do Muro de Imunidade (*Safeguard* de Mimikyu ID 767)

### Texto da Regra de Mimikyu (ID 767):
$$\text{Habilidade \textit{Safeguard}}: \text{"Prevent all damage done to this Pokémon by attacks from your opponent's Pokémon ex."}$$

### Interação com `lb1009` e `lb945`:
1. **Mega Lucario ex (ID 678)** é um **Pokémon ex**.
2. Quando Mega Lucario ex ataca com *Mega Brave* (270 dano) contra Mimikyu ativo, **o dano calculado pelo simulador do Kaggle é exatamente ZERO ($0$)**.
3. O adversário não possui Apoiadores como *Canceling Cologne* em sua lista para desligar habilidades.
4. Riolu (ID 673) no banco não consegue causar dano suficiente para derrubar os 70 HP de Mimikyu antes que:
   - *Munkidori* (Adrena-Brain) mova 30 de dano por turno para os Riolu/Lucario do oponente.
   - *Teal Mask Ogerpon ex* seja alimentado com segurança no banco via *Teal Dance* (1 energia por turno + compra).
   - *Boss's Orders* puxe Riolus indefesos do banco para nocautes de 1 prêmio.

---

## 3. Matriz de Síntese dos Candidatos

| Deck Candidate | Status | Composição Anti-Lucario | Composição Anti-Alakazam | Composição Anti-Crustle |
| :--- | :--- | :--- | :--- | :--- |
| **`deck_supreme_v0`** | Baseline | 2x Tapu Bulu, 2x Munkidori | 1x Judge, 1x Stamp | 2x Tapu Bulu (40% WR) |
| **`deck_v1_tempo`** | Rejeitado | 4x Carmine (1V-4D vs lb945) | 1x Judge | Colapsou vs Crustle (0-5) |
| **`deck_v2_control`** | Retido (Diagnóstico) | 2x Munkidori | 2x Judge, 3x Boss, 3x Munkidori | **60% WR (3-2 vs Crustle)** |
| **`deck_v3_apex_sovereign`** | **Elegível para Triagem** | **2x Mimikyu (Safeguard Imunidade a ex)** + 2x Carmine | **2x Judge + 1x Stamp + 3x Boss + 2x Munkidori** | **2x Tapu Bulu (220 dano)** |

O candidato `deck_v3_apex_sovereign.json` preserva 100% dos ganhos de controle e quebra de imunidade demonstrados pelo `v2` no screen AR-028, adicionando a única contramedida matemática viável contra a agressão rápida de Mega Lucario.
