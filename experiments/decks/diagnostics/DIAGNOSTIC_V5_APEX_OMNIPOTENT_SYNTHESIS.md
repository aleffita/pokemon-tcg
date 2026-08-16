# Síntese Adversarial & Prova Hipergeométrica: Candidato Deck v5 (Apex Omnipotent)

**Data:** 2026-08-16  
**Autores:** Antigravity Quorum (Red Team Architect + Hypergeometric Mathematician)  
**Destinatário:** Coordenador Autoresearch Codex (GPT-5.6-Luna-Max)  
**Caminho do Arquivo:** `experiments/decks/diagnostics/DIAGNOSTIC_V5_APEX_OMNIPOTENT_SYNTHESIS.md`  

---

## 1. Fundamentação Matemática Hipergeométrica Multivariada

Calculamos as probabilidades exatas do sorteio sem reposição ($n=7$, $N=60$):

$$\Omega = \binom{60}{7} = 386.206.920 \text{ mãos possíveis}$$

### Métricas Comparativas de Consistência (v3 vs v4 vs v5):

| Métrica Hipergeométrica | Deck v3 (2 Mimikyu) | Deck v4 (4 Mimikyu) | Deck v5 Apex Omnipotent (3 Mimikyu + Iron Leaves) |
| :--- | :---: | :---: | :---: |
| **Total de Pokémon Básicos ($B$)** | 12 | 13 | **13** |
| **$P(\text{Mulligan Inicial})$** | 19,06% | 16,28% | **16,28%** |
| **$P(\text{Setup Válido com até 1 Mulligan})$** | 96,37% | 97,35% | **$\mathbf{97,35\%}$** |
| **$P(\text{Mimikyu Ativo na Abertura})$** | 27,36% | 47,72% | **$\mathbf{38,24\%}$** |
| **$P(\text{Acesso Direto a Mimikyu no Turno 1})$** | 83,17% | 90,23% | **$\mathbf{90,10\%}$** |
| **$P(\text{Acesso Composto a Energia de Grama T1})$** | 81,98% | 78,70% | **$\mathbf{82,32\%}$** |
| **$P(\text{Acesso a Mecanismos de Busca T1})$** | 86,85% | 88,96% | **$\mathbf{88,96\%}$** |

---

## 2. Inovações Táticas do Candidato v5 (`deck_v5_apex_omnipotent.json`)

1. **`Iron Leaves ex` (ID 75, 1 cópia)**:
   * **Habilidade *Rapid Vernier***: Ao descer da mão para o banco, move-se imediatamente para o ativo e absorve todas as energias necessárias de outros Pokémon em campo, desferindo **180 de dano instantâneo** (*Prism Edge*) sem gastar cartas de troca ou ligar energias da mão.
2. **`Mimikyu` (ID 767, 3 cópias)**:
   * Ponto de equilíbrio ideal: garante **90,1% de acesso a *Safeguard*** no Turno 1 contra Mega Lucario ex (`lb1009`/`lb945`), liberando slot para *Iron Leaves ex*.
3. **`Tapu Bulu` (ID 920, 2 cópias)**:
   * *Wood Hammer* (220 dano) mantém a aniquilação garantida de *Crustle* (`lb814`).
4. **`Unfair Stamp` (ID 1080) + `Judge` (ID 1213, 2 cópias)**:
   * Disrupção de mão irreversível contra *Alakazam* (`lb826`).
5. **`Buddy-Buddy Poffin` (ID 1086, 4 cópias) + `Bug Catching Set` (ID 1094, 4 cópias)**:
   * Teto máximo de consistência de busca no formato.

---

## 3. Composição Itemizada das 60 Cartas Validadas

- **13 Pokémon**: 4x Ogerpon ex (96), 3x Mimikyu (767), 2x Tapu Bulu (920), 2x Munkidori (112), 1x Iron Leaves ex (75), 1x Latias ex (184).
- **22 Itens/ACE SPEC**: 4x Bug Catching (1094), 4x Poké Pad (1152), 4x Ultra Ball (1121), 4x Poffin (1086), 3x Stretcher (1097), 2x Switch (1123), 1x Unfair Stamp (1080).
- **11 Supporters**: 4x Lillie (1227), 3x Boss (1182), 2x Carmine (1192), 2x Judge (1213).
- **2 Estádios**: 2x Battle Cage (1264).
- **12 Energias**: 9x Basic Grass (1), 2x Basic Darkness (7), 1x Grow Grass (18).

Arquivo: `experiments/decks/candidates/deck_v5_apex_omnipotent.json` (100% verificado no SQLite e pronto para o Codex).
