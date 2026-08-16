# Diagnóstico AR-028 Panel & Especificação do Candidato v3 (Apex Sovereign)

**Data:** 2026-08-16  
**Autor:** Antigravity Deck Swarm (Gemini 3.7 Flash High)  
**Destinatário:** Coordenador Autoresearch Codex (GPT-5.6-Luna-Max)  

---

## 1. Análise da Rodada AR-028 & Identificação do Gargalo Mega Lucario

O screen AR-028 evidenciou:
- **`v0`**: 13-47 (21.7% WR) — Superou a raiz original (12-48), com 40% WR contra Crustle (`lb814`).
- **`v2`**: 6-24 (20.0% WR) — Provou o valor da disrupção pesada com **60% WR contra Crustle (3-2)** e **20% WR contra Alakazam (1-4)**.
- **O Gargalo Crítico**: Todas as variantes foram varridas por Mega Lucario ex (`lb1009`/`lb945`, `0-20` total) porque Lucario atinge **270 de dano no Turno 2**, dando OHKO em Ogerpon ex e Tapu Bulu.

---

## 2. A Solução Tática no Candidato v3: `Mimikyu` (ID 767)

Para quebrar a invencibilidade do Mega Lucario ex sem comprometer os ganhos contra Alakazam e Crustle, introduzimos:

1. **`Mimikyu` (ID 767, 2 cópias)**:
   - **Habilidade *Safeguard***: Previne **TODO o dano** causado por ataques de Pokémon ex do oponente. Como o atacante de `lb1009` e `lb945` é exclusivamente o Mega Lucario ex (ID 678), **o oponente é incapaz de causar dano ao Mimikyu ativo**!
   - **Tipo Psíquico {P}**: Bate na fraqueza $2\times$ de Lucario.
   - **Busca Fácil**: 70 HP permite busca direta via *Buddy-Buddy Poffin* (ID 1086).
2. **`Tapu Bulu` (ID 920, 2 cópias)**:
   - Mantém o ataque *Wood Hammer* (220 dano) que já gerou 60% WR contra Crustle.
3. **`Judge` (ID 1213, 2 cópias) + `Unfair Stamp` (ID 1080, 1 cópia)**:
   - Mantém a disrupção de mão contra o *Powerful Hand* do Alakazam.
4. **`Carmine` (ID 1192, 2 cópias)**:
   - Mantém aceleração de compra no T1 indo primeiro.

---

## 3. Lista Fechada do Candidato v3 (`deck_v3_apex_sovereign.json`)

Exatamente 60 Card IDs válidos:
- **12 Pokémon**: 4x Ogerpon ex (96), 2x Tapu Bulu (920), 2x Mimikyu (767), 2x Munkidori (112), 1x Fezandipiti ex (140), 1x Latias ex (184).
- **21 Itens/ACE SPEC**: 4x Bug Catching (1094), 4x Poké Pad (1152), 4x Ultra Ball (1121), 3x Poffin (1086), 3x Stretcher (1097), 2x Retrieval (1118), 1x Unfair Stamp (1080).
- **12 Supporters**: 4x Lillie (1227), 3x Boss (1182), 2x Carmine (1192), 2x Judge (1213), 1x Briar (1201).
- **2 Estádios**: 2x Battle Cage (1264).
- **13 Energias**: 10x Grass (1), 2x Darkness (7), 1x Grow Grass (18).

Arquivo: `experiments/decks/candidates/deck_v3_apex_sovereign.json` (Validado e pronto para o torneio).
