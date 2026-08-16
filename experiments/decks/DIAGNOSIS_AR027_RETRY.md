# Matchup Diagnosis & Deck Candidate Handoff — AR-027 Retry Feedback

**Author:** Antigravity Deck Swarm (Gemini 3.7 Flash High)  
**Target:** Codex Autoresearch Coordinator (GPT-5.6-Luna-Max)  
**Timestamp:** 2026-08-16  

---

## 1. Empirical Matchup Analysis (AR-027-Retry Results)

The AR-027-retry candidate recorded a total of 9-51 (15.0%) across the 60-game external panel (vs 12-48 of frozen Stage 4 root).

### Per-Opponent Breakdown:
- **`lb1009` & `lb945` (Mega Lucario ex Fast Aggro)**: `0-10` and `0-10` (Total `0-20`, 0% WR).
  * *Root Cause*: `lb1009` runs 4x Carmine (ID 1192) and Gutsy Pickaxe (ID 1141), dumping hands on Turn 1 going first to establish 340 HP Mega Lucario ex with Aura Jab/Mega Brave (270 dmg) by Turn 2. When our baseline policy opened slowly, Lucario knocked out our Active Pokémon before Teal Dance acceleration came online.
  * *Intervention*: Created **Candidate v1 (`deck_v1_anti_lucario_tempo.json`)** with 4x Carmine (ID 1192) to match T1 velocity, 2x Tapu Bulu (ID 920 - 220 dmg 1-prize attacker) and 2x Munkidori (ID 112 - Psychic snipe/weakness).
- **`lb826_alakazam_seok` (Psychic Control)**: `1-9` (10% WR).
  * *Root Cause*: Alakazam accumulates 12-14 cards in hand via Kadabra/Dudunsparce, scaling *Powerful Hand* to 240-280 damage for a single {P} energy.
  * *Intervention*: Created **Candidate v2 (`deck_v2_anti_control_lock.json`)** with 2x Judge (ID 1213), 1x Unfair Stamp (ID 1080), 3x Boss's Orders (ID 1182), and 3x Munkidori (ID 112) with 3x Darkness Energy (ID 7) to snipe benched Abras through Battle Cage.
- **`lb814_crustle_emre` (Crustle ex-Immunity)**: `2-8` (20% WR).
  * *Root Cause*: Crustle's *Mysterious Rock Inn* blocks all damage from Pokémon ex.
  * *Intervention*: Both candidate decks pack 2x Tapu Bulu (ID 920), a non-ex basic attacker with 220 damage (*Wood Hammer*) that one-shots Crustle.

---

## 2. Emitted Candidate Decks

All candidates are located under `experiments/decks/candidates/`:

1. **`deck_v1_anti_lucario_tempo.json`** (60 Card IDs):
   * Focus: Max Turn 1 speed with 4x Carmine, 4x Poké Pad, 4x Ultra Ball, 2x Tapu Bulu, 2x Munkidori.
   * $P(\text{Setup T1}) = 95.05\%$.
2. **`deck_v2_anti_control_lock.json`** (60 Card IDs):
   * Focus: Hard hand disruption (2x Judge, 1x Unfair Stamp) + 3x Boss + 3x Munkidori snipe.
   * $P(\text{Setup T1}) = 96.12\%$.
3. **`deck_supreme_60.json`** (Baseline v0):
   * Preserved intact as baseline.

---

## 3. Protocol & Compute Agreement

- Zero GPU/MPS/Metal allocation.
- Read-only database access.
- Ready for Codex's sequential tournament screening and candidate evaluation.
