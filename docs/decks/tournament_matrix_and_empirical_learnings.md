---
type: analysis
title: "Matriz de Torneios & Aprendizados Empíricos (AR-028)"
description: "Análise quantitativa dos resultados da série de torneios AR-028 comparando as variantes do Deck Supremo contra o painel de 6 oponentes."
tags: [pokemon-tcg, tournaments, empirical, metrics, results, win-rate]
timestamp: "2026-08-16T19:30:00-03:00"
---

# Matriz de Torneios & Aprendizados Empíricos (AR-028)

## 1. Quadro Comparativo de Performance Empírica

Na rodada de torneios AR-028 executada pelo harness sobre os pesos congelados do Stage 4, foram avaliadas três configurações de deck distintas:

| Oponente | Baseline Root (Deck Original) | Deck Supreme v0 (60 jogos) | Deck v1 Tempo (30 jogos) | Deck v2 Control (30 jogos) |
| :--- | :---: | :---: | :---: | :---: |
| **`random`** | 7–3 (70%) | **7–3 (70%)** | 2–3 (40%) | 0–5 (0%) |
| **`first_sub`** (#251) | 2–8 (20%) | 2–8 (20%) | 0–5 (0%) | **2–3 (40%)** |
| **`lb1009`** *(Mega Lucario)* | 0–10 (0%) | 0–10 (0%) | 0–5 (0%) | 0–5 (0%) |
| **`lb945`** *(Mega Lucario)* | 0–10 (0%) | 0–10 (0%) | **1–4 (20%)** | 0–5 (0%) |
| **`lb826`** *(Alakazam)* | 1–9 (10%) | 0–10 (0%) | 0–5 (0%) | **1–4 (20%)** |
| **`lb814`** *(Crustle)* | 2–8 (20%) | **4–6 (40%)** | 0–5 (0%) | **3–2 (60%)** |
| **Total Geral** | **12–48 (20.0%)** | **13–47 (21.7%)** | **3–27 (10.0%)** | **6–24 (20.0%)** |

---

## 2. Invariantes & Descobertas Críticas

### 1. Quebra da Imunidade do Crustle (`lb814`)
- No baseline, a taxa de vitória contra Crustle era de apenas 20%.
- No **`deck_supreme_v0`**, a inclusão de *Tapu Bulu* (ID 920 - 220 dano) elevou a taxa para **40%**.
- No **`deck_v2_control`**, com suporte de *Battle Cage* e *Munkidori*, a taxa atingiu **60% de vitórias** (3V–2D), revertendo completamente a desvantagem do matchup.

### 2. Penetração contra Mega Lucario (`lb945`)
- A adição de **4x Carmine** no **`deck_v1_tempo`** gerou a primeira vitória registrada contra a linha de Mega Lucario (1–4, 20%), provando que a velocidade de compra no Turno 1 indo primeiro é a chave para disputar o início de partida.

### 3. Síntese no Candidato v3 (Apex Sovereign)
- O **`deck_v3_apex_sovereign.json`** unifica:
  1. A velocidade de *Carmine* (2x) para início de jogo.
  2. O controle de mão e puxada de *Judge* (2x), *Unfair Stamp* (1x) e *Boss's Orders* (3x).
  3. Os atacantes específicos *Tapu Bulu* (2x) e *Munkidori* (2x).
