# Handoff Report — Milestone 1 Worker (Deck Supreme 60)

**Agent ID:** `worker_m1`  
**Role:** Implementer / QA / Specialist  
**Milestone:** Milestone 1 — Optimal 60-Card Deck Construction & Verification  
**Generated At:** 2026-08-16T19:05:30Z  

---

## 1. Observation

Direct empirical observations and verified metrics from the implementation:

### 1.1 Deliverable Files Created & Verified
1. **`agent/deck.json`**:
   - Array of exactly 60 integer Card IDs:
     `[96, 96, 96, 96, 920, 920, 112, 112, 140, 184, 235, 1094, 1094, 1094, 1094, 1152, 1152, 1152, 1152, 1121, 1121, 1121, 1121, 1086, 1086, 1086, 1097, 1097, 1097, 1118, 1118, 1123, 1123, 1127, 1080, 1227, 1227, 1227, 1227, 1182, 1182, 1192, 1192, 1213, 1201, 1264, 1264, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 7, 18]`
   - Every single integer Card ID was validated against `model/results.db` in table `cards`.

2. **`experiments/decks/deck_supreme_60.json`**:
   - `deck_name`: `"Deck Supreme 60 — Teal Mask Ogerpon ex / Turbo Acceleration & Psychic Counter Hybrid"`
   - `archetype`: `"Teal Mask Ogerpon ex / Grass Turbo Ramp / Anti-Meta Control"`
   - `card_count`: `60`
   - `card_list`: Array of 24 distinct card entries (summing to 60) with fields `id`, `name`, `category`, `stage`, `type`, `hp`, `rule`, `quantity`, and `role`.
   - `energy_curve`: Detailed breakdown (10 Basic {G}, 2 Basic {D}, 1 Grow Grass {G}) with turn-by-turn attachment expectations.
   - `hypergeometric_probabilities`: Exact float, percentage, and rational arithmetic fractions for all required opening hand events.
   - `matchup_profiles`: Comprehensive Red Team counter-strategies against the 6 panel archetypes (`lb826_alakazam_seok`, `lb1009_945_mega_lucario_ex`, `lb814_600_dragapult_crustle`, `first_sub_kaggle_2707`, `lb510_mega_abomasnow`, `deck_633_baseline_yan`).

3. **`tests/test_deck_m1_validation.py`**:
   - Automated unit test suite executed via `uv run pytest tests/test_deck_m1_validation.py` (100% PASS in 0.01s).

---

### 1.2 Quantitative Deck Composition & Verification Table

| Slot # | Card ID | Card Name | Category / Stage | Type / HP | Rule Box | Qty | Role in Deck Supreme 60 |
| :---: | :---: | :--- | :--- | :--- | :--- | :---: | :--- |
| 1-4 | **96** | Teal Mask Ogerpon ex | Basic Pokémon | {G} / 210 HP | Pokémon ex | **4** | Primary Attacker & *Teal Dance* Grass ramp/draw engine |
| 5-6 | **920** | Tapu Bulu | Basic Pokémon | {G} / 140 HP | None | **2** | Single-prize heavy nuke (*Wood Hammer* 220 dmg; bypasses ex-immunity) |
| 7-8 | **112** | Munkidori | Basic Pokémon | {P} / 110 HP | None | **2** | Psychic presence & *Adrena-Brain* 30 damage counter sniping |
| 9 | **140** | Fezandipiti ex | Basic Pokémon | {D} / 210 HP | Pokémon ex | **1** | Disruption recovery (*Flip the Script* draws 3 after KO) |
| 10 | **184** | Latias ex | Basic Pokémon | {P} / 210 HP | Pokémon ex | **1** | Universal mobility (*Skyliner* gives 0 retreat to all Basics) |
| 11 | **235** | Budew | Basic Pokémon | {G} / 30 HP | None | **1** | Early setup pivot & Poffin/Bug Catching Set target |
| 12-15 | **1094** | Bug Catching Set | Item | Trainer | None | **4** | Top-7 search for up to 2 Grass Pokémon and/or Basic Grass Energy |
| 16-19 | **1152** | Poké Pad | Item | Trainer | None | **4** | Deep item/trainer digging and recursion |
| 20-23 | **1121** | Ultra Ball | Item | Trainer | None | **4** | Universal Pokémon search & hand discard outlet |
| 24-26 | **1086** | Buddy-Buddy Poffin | Item | Trainer | None | **3** | Direct benching of low-HP Basic Pokémon |
| 27-29 | **1097** | Night Stretcher | Item | Trainer | None | **3** | Targeted recovery of 1 Pokémon or 1 Basic Energy to hand |
| 30-31 | **1118** | Energy Retrieval | Item | Trainer | None | **2** | Recovery of 2 Basic Energies to hand to fuel *Teal Dance* |
| 32-33 | **1123** | Switch | Item | Trainer | None | **2** | Active repositioning, condition clear, and mobility safety |
| 34 | **1127** | Tera Orb | Item | Trainer | None | **1** | Zero-discard search for Teal Mask Ogerpon ex |
| 35 | **1080** | Unfair Stamp | Item | Trainer | **ACE SPEC** | **1** | Disruption hand reset (Opponent to 2, Self to 5 after KO) |
| 36-39 | **1227** | Lillie's Determination | Supporter | Trainer | None | **4** | Premier draw engine (draw 6, or 8 when trailing) |
| 40-41 | **1182** | Boss’s Orders | Supporter | Trainer | None | **2** | Tactical gust to drag key targets/high-retreat liabilities |
| 42-43 | **1192** | Carmine | Supporter | Trainer | None | **2** | Turn 1 going-first cycle & discard accelerator |
| 44 | **1213** | Judge | Supporter | Trainer | None | **1** | Symmetrical hand reset to 4 cards vs Alakazam hand scaling |
| 45 | **1201** | Briar | Supporter | Trainer | None | **1** | Endgame prize acceleration (+1 prize on Tera attack KO) |
| 46-47 | **1264** | Battle Cage | Stadium | Trainer | None | **2** | Prevents damage counter placement on Benched Pokémon (anti-Dragapult) |
| 48-57 | **1** | Basic {G} Energy | Basic Energy | {G} | None | **10** | Core fuel for manual attachments and *Teal Dance* |
| 58-59 | **7** | Basic {D} Energy | Basic Energy | {D} | None | **2** | Activates Munkidori *Adrena-Brain* ability |
| 60 | **18** | Grow Grass Energy | Special Energy | {G} | None | **1** | {G} Energy + 20 HP resilience buff for Grass Pokémon |
| **TOTAL** | | | | | | **60** | **100% Rules Compliant** |

---

### 1.3 Exact Hypergeometric Probability Proofs ($N=60, n=7$)

- **Basic Pokémon Count ($K_b = 11$):**
  - $P(\text{Setup } n=7) = 1 - \frac{\binom{49}{7}}{\binom{60}{7}} = \frac{1137524}{1462905} \approx \mathbf{77.7579\%}$
  - $P(\text{Mulligan } n=7) = \frac{\binom{49}{7}}{\binom{60}{7}} = \frac{325381}{1462905} \approx \mathbf{22.2421\%}$
  - $P(\text{Setup within 1 Mulligan}) = 1 - [P(\text{Mulligan})]^2 = \frac{2034218243864}{2140091039025} \approx \mathbf{95.0529\%} \ge \mathbf{92.0\%}$
  - $P(\text{Mulligan within 1 Mulligan}) = [P(\text{Mulligan})]^2 = \frac{105872795161}{2140091039025} \approx \mathbf{4.9471\%} \le \mathbf{8.0\%}$

- **Energy Availability ($K_e = 13$):**
  - $P(\text{T1 Energy } n=7) = 1 - \frac{\binom{47}{7}}{\binom{60}{7}} = \frac{9797437}{11703240} \approx \mathbf{83.7156\%}$
  - $P(\text{T1 Energy with Natural Draw } n=8) = 1 - \frac{\binom{47}{8}}{\binom{60}{8}} = \frac{13600990}{15506793} \approx \mathbf{87.7099\%}$

- **Search Engine Access ($K_{\text{eng}} = 22$ cards):**
  - $P(\text{T1 Engine Access } n=7) = 1 - \frac{\binom{38}{7}}{\binom{60}{7}} = \frac{74479}{76995} \approx \mathbf{96.7323\%}$

---

## 2. Logic Chain

1. **Overcoming Historical Baseline Weaknesses (Deck #633 & Deck #251):**
   - *Observation 1:* Deck #633 achieved 27.9% WR running only 5 Basic Pokémon, producing a 52.54% mulligan rate and giving opponents bonus cards in >50% of games.
   - *Observation 2:* Deck #251 relied on a 4-4-4 Stage 2 Alakazam line with fragile 50 HP Abras, collapsing against Turn 2 aggro decks before evolving.
   - *Deduction:* Deck Supreme 60 establishes 11 robust Basic Pokémon ($K_b = 11$), slashing the mulligan rate to 4.95% within 1 mulligan, while running exclusively Basic attackers and high-tempo single-prize nukes (Tapu Bulu 140 HP, 220 dmg) that require zero evolution turns.

2. **Countering the Top Meta Threats (Psychic Exploitation & Damage Shielding):**
   - *Observation 3:* The #1 leaderboard deck `lb1009_mega_lucario_ex_islet` (340 HP) has 2x weakness to Psychic ({P}).
   - *Observation 4:* `lb600_dragapult_ex` spreads 60 damage counters on bench via *Phantom Dive*, and `lb814_crustle_emre` relies on *Mysterious Rock Inn* ex-immunity.
   - *Deduction:*
     - Munkidori (ID 112) + Darkness Energy (ID 7) and Latias ex (ID 184) inject Psychic attack presence and precision 30-damage sniping (*Adrena-Brain*) directly onto Lucario.
     - Battle Cage (ID 1264 x2) completely shuts down Dragapult's 60-counter bench spread.
     - Tapu Bulu (ID 920) is a non-ex single-prize attacker with 220 damage (*Wood Hammer*), directly bypassing Crustle's immunity wall for clean OHKOs.

3. **Disruption and Mobility Hardening:**
   - *Observation 5:* Opponent control engines (`lb826_alakazam_seok`, `first_sub_kaggle_2707`) rely on hand accumulation and retreat stalling with Nighttime Mine / Boss's Orders.
   - *Deduction:*
     - Judge (ID 1213) and Unfair Stamp (ID 1080) reset opponent hands to 4 and 2 cards, disabling *Powerful Hand* damage scaling.
     - Latias ex (ID 184) *Skyliner* grants 0 retreat cost to all Basic Pokémon, completely nullifying retreat trap locks.

---

## 3. Caveats

- **No Caveats:** All 60 card IDs physically exist in `model/results.db`, all deckbuilding rules are 100% satisfied, all hypergeometric proofs are mathematically exact, and all tests pass with zero GPU/MPS resource contention.

---

## 4. Conclusion

Milestone 1 deliverables have been fully constructed and validated:
- `agent/deck.json` is ready for direct ingestion by Codex self-play and evaluation pipelines.
- `experiments/decks/deck_supreme_60.json` provides the complete architectural capsule with exact energy curves, hypergeometric probabilities, and 6 matchup profiles.
- `tests/test_deck_m1_validation.py` provides automated CI validation.

---

## 5. Verification Method

To independently verify the deliverables and assertions:

```bash
# 1. Run the automated pytest validation suite
uv run pytest tests/test_deck_m1_validation.py -v

# 2. Verify agent/deck.json length and database validity
uv run python -c "
import json, sqlite3
deck = json.load(open('agent/deck.json'))
c = sqlite3.connect('file:model/results.db?mode=ro', uri=True).cursor()
assert len(deck) == 60, f'Length is {len(deck)}'
for cid in deck:
    assert c.execute('SELECT id FROM cards WHERE id=?', (cid,)).fetchone() is not None
print('Verified: 60/60 cards valid in model/results.db')
"

# 3. Verify experiments/decks/deck_supreme_60.json capsule integrity
uv run python -c "
import json
capsule = json.load(open('experiments/decks/deck_supreme_60.json'))
assert capsule['card_count'] == 60
assert len(capsule['card_list']) == 24
assert len(capsule['matchup_profiles']) == 6
print('Verified: deck_supreme_60.json capsule complete and intact')
"
```
