# Opponent Panel Technical Survey & Adversarial Specification Report

**Working Directory**: `/Users/alefita/workdir/pokemon-tcg/.agents/survey_miner_2/`  
**Date**: 2026-08-16  
**Investigator**: Opponent Panel Miner (Antigravity Cognitive Swarm)  
**Target Milestone**: M1 / M2 Tactical 60-Card Deck Engineering & Codex Autoresearch Integration

---

## 1. Observation

Direct empirical extraction from `model/results.db` (Schema 2.0.0, 139,783 matches), Codex autoresearch logs (`experiments/autoresearch/AR-019` through `AR-027`), and public agent deck files (`public_agents/`) reveals six distinct operational opponent archetypes in the competitive meta:

### Summary of Discovered Features & Archetypes

```
## Features Discovered
| # | Category | Archetype / Opponent | Key Cards & Mechanics | HP Range | Energy Curve | Primary Weakness | Discovered Via |
|---|----------|----------------------|-----------------------|----------|--------------|------------------|----------------|
| 1 | Control / Disruption | **lb826_alakazam_seok** (Variants: lb881, lb966, lb1004) | Alakazam [743] (Powerful Hand: 20x hand size), Kadabra [742], Dawn [1231], Hilda [1225], Enhanced Hammer [1081], Battle Cage [1264], Telepath Energy [19] | 50 - 140 HP (Single-Prize) | 4 Telepath {P}, 1 Enriching {C}, 2 Basic {P} (Total: 7) | Darkness ({D}) | `public_agents/lb826_alakazam_seok/deck.csv` & SQLite `decks` |
| 2 | Fast Burst Aggro | **lb1009_mega_lucario_ex_islet** (Variant: lb945) | Mega Lucario ex [678] (340 HP, Mega Brave 270 dmg, Aura Jab 130 + 3 discard attach), Carmine [1192] (T1 first draw), Fighting Gong [1142], Premium Power Pro [1141], Hariyama [674] (Heave-Ho Catcher) | 80 - 340 HP (ex / Mega) | 14 Basic {F} (Total: 14) | Psychic ({P}) | `public_agents/lb1009_mega_lucario_ex_islet/deck.csv` & AR-024 |
| 3 | Spread / Sniper / Wall | **lb600_dragapult_ex** / **lb814_crustle_emre** / **lb798** | Dragapult ex [121] (Phantom Dive: 200 + 60 bench counters, Crispin [1198], Unfair Stamp [1080], Fezandipiti ex [140]) / Crustle [345] (Mysterious Rock Inn: immune to ex, 270 HP with Cape/Grow Grass) | 70 - 320 HP (ex & Stage 1) | Dragapult: 4 {R}, 4 {P}; Crustle: 19 {G}, 4 Grow {G}, 4 Spiky {C}, 4 Mist {C} | Dragapult: None; Crustle: Fire ({R}) | `public_agents/starters/lb600_dragapult_ex/` & `public_agents/lb814_crustle_emre/` |
| 4 | 1-Prize Attrition Baseline | **first_sub_kaggle_2707** / **fitalabs_hero_deck251** | Alakazam [743] + Dudunsparce [66] engine, Buddy-Buddy Poffin [1086], Rare Candy [1079], Dawn [1231], Nighttime Mine [1266], Xerosic's Machinations [1197] | 50 - 140 HP (Single-Prize) | 4 Telepath {P}, 1 Enriching {C}, 2-3 Basic {P} (Total: 7-8) | Darkness ({D}) | `public_agents/submissions/` & SQLite Deck #251 |
| 5 | Superheavy Tank Ramp | **lb510_mega_abomasnow_ex** / **lb526_iono** | Mega Abomasnow ex [723] (350 HP, Hammer-lanche 100x discard {W}, 34 Basic {W} Energy, Precious Trolley [1126]) / Iono's Bellibolt ex [269] (280 HP, 22 {L} Energy, Canari [1233]) | 90 - 350 HP (Mega / ex) | Abomasnow: 34 Basic {W}; Bellibolt: 22 Basic {L} | Abomasnow: Metal ({M}); Bellibolt: Fighting ({F}) | `public_agents/starters/` |
| 6 | Turbo Energy Acceleration | **Deck #633 Yan (Teal Mask Ogerpon ex)** / **Deck #440** | Teal Mask Ogerpon ex [96] (Teal Dance: attach {G} from hand & draw 1; Myriad Leaf Shower: 30 + 30 per energy attached to both Actives), Tapu Bulu [920] (220 dmg 1-prize nuke), Bug Catching Set [1094], Judge [1213] | 140 - 210 HP (Basic ex) | 17 Basic {G}, 2 Grow Grass {G} (Total: 19) | Fire ({R}) | SQLite `decks` ID=633 (27.9% WR baseline) |
```

---

## 2. Logic Chain: Detailed Archetype Profiles

### Archetype 1: Control & Energy Punishment (`lb826_alakazam_seok`)
- **Deck Structure**: 18 Pokémon, 35 Trainers, 7 Energies (Total: 60 cards).
- **Core Engine**:
  - **4x Abra [741]** (50 HP, {P}): T1 search target with Buddy-Buddy Poffin [1086] or Telepath Psychic Energy [19].
  - **4x Kadabra [742]** (80 HP, {P}): `Psychic Draw` draws 2 cards upon evolution from hand.
  - **3x-4x Alakazam [743]** (140 HP, {P}): `Psychic Draw` draws 3 cards upon evolution. Attack `Powerful Hand` [{P}]: 20 damage per card in hand. With hand size 10–14, deals 200–280 damage for a single energy.
  - **3x Dunsparce [305] / 2x Dudunsparce [66]**: `Run Away Draw` draws 3 cards and shuffles Dudunsparce into deck, refreshing draw cycles.
  - **1x Fezandipiti ex [140]**: `Flip the Script` draws 3 cards if a Pokémon was KO'd on opponent's last turn.
  - **1x Genesect [142]**: `ACE Nullifier` locks opponent out of playing ACE SPEC cards if Genesect holds a Tool.
  - **1x Shaymin [343]**: `Flower Curtain` protects all non-Rule Box benched Pokémon from attack damage.
  - **4x Battle Cage [1264]**: Blocks placement of damage counters on benched Pokémon.
  - **3x Enhanced Hammer [1081]**: Discards opponent's Special Energy.
  - **4x Dawn [1231] & 4x Hilda [1225]**: Guaranteed deterministic evolution search.
- **Opening Lines**:
  - T1: Bench Abra + Dunsparce via Buddy-Buddy Poffin or Telepath Energy.
  - T2: Evolve Kadabra (draw 2), evolve Alakazam (draw 3), activate Dudunsparce (draw 3). Hand expands to 12+ cards; `Powerful Hand` executes OHKO for 1 energy.
- **Vulnerabilities**:
  - **Hand Disruption**: Judge [1213], Iono, or Unfair Stamp [1080] resets hand to 2-4 cards, collapsing Alakazam's attack damage to 40-80.
  - **Darkness Weakness**: {D} Pokémon deal 2x damage.
  - **Fragile Basics**: 50 HP Abra is easily sniped before evolving.

---

### Archetype 2: Top Leaderboard Fast Aggro (`lb1009_mega_lucario_ex_islet` & `lb945`)
- **Deck Structure**: 17 Pokémon, 29 Trainers, 14 Energies (Total: 60 cards).
- **Core Engine**:
  - **4x Riolu [677]** (80 HP, {F}) & **4x Mega Lucario ex [678]** (340 HP, {F}, Stage 1 Mega ex).
    - Attack 1 `Aura Jab` [{F}, 130 dmg]: Recovers and attaches 3 Basic {F} Energies from discard pile to bench.
    - Attack 2 `Mega Brave` [{F}{F}, 270 dmg]: Massive 270 base damage; boosted to 300+ with Premium Power Pro [1141]. Cannot be used consecutively.
  - **2x Makuhita [673] & 2x Hariyama [674]**: `Heave-Ho Catcher` forces opponent's benched Pokémon to Active upon evolution (built-in Boss's Orders).
  - **2x Lunatone [675] & 3x Solrock [676]**: `Lunar Cycle` discards {F} to draw 3 cards; Solrock attacks for 70 dmg for {F}.
  - **4x Carmine [1192]**: Playable on Turn 1 going first to discard hand and draw 5.
  - **4x Fighting Gong [1142]**: Searches Basic {F} Energy or Basic {F} Pokémon.
  - **1x Hero's Cape [1159]**: Mega Lucario reaches 440 HP.
- **Opening Lines**:
  - T1 (Going 1st): Carmine [1192] to cycle cards into discard and setup Riolu/Makuhita.
  - T2: Evolve Mega Lucario ex, attach 2nd energy, execute `Mega Brave` for 270-300 damage OHKO.
- **Vulnerabilities**:
  - **Psychic Weakness ({P})**: All core attackers take 2x from Psychic. Alakazam dealing 170 base damage reaches 340 damage, instantly OHKOing Mega Lucario ex.
  - **Consecutive Attack Lock**: `Mega Brave` cannot be used two turns in a row. If trapped in active without Switch, damage stalls.
  - **2-Prize Target**: Losing 2 Mega Lucarios yields 4-6 prizes.

---

### Archetype 3: Spread Damage & Immunity Wall (`lb600_dragapult_ex` & `lb814_crustle_emre`)
- **Variant 3A: Dragapult ex [121]**:
  - **3x Dragapult ex [121]** (320 HP, Dragon, No Weakness, Tera):
    - `Phantom Dive` [{R}{P}, 200 dmg + 60 damage counters placed on bench].
  - **4x Drakloak [120]**: `Recon Directive` look top 2, take 1. (4 Drakloaks = +4 cards/turn).
  - **4x Crispin [1198]**: Accelerates {R} and {P} basic energies in a single turn.
  - **1x Unfair Stamp [1080]**: Disruption hand reset (Opponent to 2, User to 5).
  - **1x Latias ex [184]**: `Skyliner` grants free retreat to all Basic Pokémon.
  - **2x Team Rocket's Watchtower [1256]**: Disables Colorless Pokémon abilities (neutralizes Dudunsparce).
- **Variant 3B: Crustle [345] (`lb814_crustle_emre`)**:
  - **4x Crustle [345]** (150 HP, {G}): `Mysterious Rock Inn` completely negates all damage from opponent's Pokémon ex.
  - Boosted to 270 HP with Grow Grass Energy [18] (+20 HP) and Hero's Cape [1159] (+100 HP).
  - Healed via 4x Cook [1212] (heal 70) and 4x Jumbo Ice Cream [1147] (heal 80).
- **Vulnerabilities**:
  - **Dragapult**: Blocked by `Battle Cage [1264]` (blocks bench counters) and `Shaymin [343]` (Flower Curtain). Requires dual energy colors ({R} + {P}).
  - **Crustle**: Vulnerable to Fire ({R}) and single-prize non-ex attackers (Alakazam, Tapu Bulu bypass its immunity).

---

### Archetype 4: Internal Baselines & `first_sub_kaggle_2707` (Deck #251)
- **Deck Structure**: 18 Pokémon, 34 Trainers, 8 Energies.
- **Roster**: 4-4-4 Alakazam line [741, 742, 743], 3-2 Dunsparce/Dudunsparce [65, 66], 1 Fezandipiti ex [140], 1 Shaymin [343].
- **Trainers**: 4 Poffin, 4 Dawn, 4 Hilda, 4 Poké Pad, 4 Enhanced Hammer, 3 Xerosic's Machinations, 3 Rare Candy, 2 Nighttime Mine.
- **Weakness**: Slow initial setup tempo; Nighttime Mine increases retreat costs symmetrically; vulnerable to hand disruption.

---

### Archetype 5: Superheavy Tank Ramp (`lb510_mega_abomasnow_ex` & `lb526_iono`)
- **Deck Structure**:
  - `lb510`: 10 Pokémon (4-4 Snover/Mega Abomasnow ex [723], 2 Kyogre [721]), 16 Trainers, 34 Basic {W} Energy.
  - `lb526`: 15 Pokémon (Iono's Bellibolt ex [269], Kilowattrel [271], Voltorb, Tadbulb, Wattrel), 23 Trainers, 22 Basic {L} Energy.
- **Mechanics**:
  - Mega Abomasnow ex: 350 HP. `Hammer-lanche` [{W}{W}] discards top 6 cards; deals 100 damage per discarded {W} energy (average 300-400 damage per swing).
  - Precious Trolley [1126] fills entire bench on T1.
- **Vulnerabilities**:
  - **Self-Mill Deckout**: Rapidly burns through 6 cards per attack.
  - **Metal Weakness ({M})** for Abomasnow, **Fighting Weakness ({F})** for Bellibolt.
  - **Retreat Lock**: Retreat Cost 4 on Abomasnow traps it if Surfing Beach [1262] is discarded or replaced.

---

### Archetype 6: High Win-Rate Turbo Acceleration (Deck #633 Yan / Teal Mask Ogerpon ex)
- **Deck Structure**: 5 Pokémon (4 Teal Mask Ogerpon ex [96], 1 Tapu Bulu [920]), 36 Trainers, 19 Energies (17 Basic {G}, 2 Grow Grass [18]).
- **Performance in SQLite**: **27.9% Win Rate** across 500+ match tournaments (highest recorded baseline in `results.db`).
- **Mechanics**:
  - **Teal Mask Ogerpon ex [96]** (210 HP, Basic Tera): `Teal Dance` attaches {G} from hand to self and draws 1 card. Multiple Ogerpons on bench provide exponential ramp + card draw per turn.
  - Attack `Myriad Leaf Shower` [{G}{G}{G}]: 30 + 30 per energy attached to both active Pokémon (easily reaches 180-240+ damage).
  - **Tapu Bulu [920]** (140 HP single-prize): `Wood Hammer` [{G}{G}●●, 220 dmg] acts as a 1-prize knockout weapon against opposing ex.
  - **4x Judge [1213]**: Resets both players' hands to 4 cards, dismantling opponent's hand scaling.
  - **4x Bug Catching Set [1094] & 4x Tera Orb [1127]**: Near 100% opening consistency.
- **Vulnerabilities**:
  - **Fire Weakness ({R})**: 2x damage from Fire attackers.
  - **2-Prize Vulnerability**: 210 HP Ogerpon is within OHKO range for boosted attackers.

---

## 3. Disruption Scenarios & Red Team Blindspots

```
## Edge Cases & Worst-Case Disruption Scenarios
| # | Disruption Vector | Trigger Mechanism | Severe Impact on Deck | Required Red Team Counter-Measure |
|---|-------------------|-------------------|-----------------------|-----------------------------------|
| 1 | **Hand Disruption (Iono / Judge / Unfair Stamp)** | Opponent plays Judge [1213] (hand to 4) or Unfair Stamp [1080] (hand to 2 after KO) | Hand-scaling attackers (Alakazam Powerful Hand) collapse from 280 dmg to 40-80 dmg, missing KOs and losing tempo. | 1. Passive on-board draw abilities (`Fezandipiti ex [140]` draws 3 upon KO; `Drakloak [120]` Recon Directive; `Teal Mask Ogerpon ex [96]` Teal Dance).<br>2. `Enriching Energy [13]` (draws 4 on attach).<br>3. Hand-independent secondary attackers (`Tapu Bulu [920]`). |
| 2 | **Active Stall & Trapping (Boss's Orders + Retreat Lock)** | Opponent plays Boss's Orders [1182] or evolves Hariyama [674] (Heave-Ho Catcher) dragging high-retreat bench Pokémon (Dudunsparce [66] Ret 3, Tapu Bulu [920] Ret 3, Mega Abomasnow Ret 4) into active, amplified by Nighttime Mine [1266] (+1 retreat) and Energy Hammers [1081, 1120]. | Active Pokémon is immobilized; turn wasted; energy stripped; opponent free to build bench sweeper. | 1. Include `Switch [1123]` (2x minimum).<br>2. Deploy `Latias ex [184]` (`Skyliner` ability gives all Basics 0 retreat cost).<br>3. Keep Dudunsparce in hand until ready to evolve and immediately use `Run Away Draw` (self-shuffles into deck). |
| 3 | **Prize Trade Disadvantage (2-Prize ex vs 1-Prize Attrition)** | Opponent runs single-prize attackers (Alakazam [743], Crustle [345], Tapu Bulu [920]) against our 2-prize Pokémon ex (Mega Lucario ex, Dragapult ex, Ogerpon ex). | 2-prize deck must score 6 separate knockouts, whereas 1-prize deck only needs 3 knockouts (2+2+2) to win the match. | 1. Integrate 1-prize sub-attackers (`Tapu Bulu [920]`, `Shaymin [343]`).<br>2. HP enhancement (`Hero's Cape [1159]` +100 HP, `Grow Grass Energy [18]` +20 HP) forcing 2-hit KOs on our ex.<br>3. Include `Briar [1201]` to steal an extra prize on the game-winning turn. |
```

---

## 4. Caveats

1. **Frozen Ladder Metadata Variance**: Specific deck lists submitted by external teams on Kaggle may feature minor tech slot variations (e.g. 1x Switch vs 1x Super Rod), but the core archetypes remain bound by the 6 extracted functional categories.
2. **Read-Only SQLite Constraints**: Analysis respected zero-GPU/Metal constraints and read-only connection contracts on `model/results.db`.

---

## 5. Conclusion

To maximize invariant win rate during the frozen ladder period (August 16–31, 2026), the Antigravity Swarm's 60-card master deck (`experiments/decks/DECK_SUPREME_60.md`) must be engineered around:
1. **Psychic ({P}) Offensive Capability**: To exploit the universal {P} weakness of the top-ranked `lb1009` / `lb945` Mega Lucario ex aggro engine.
2. **Grass ({G}) / Turbo Acceleration Hybrid**: Ingesting the high-WR mechanics of Deck #633 (Teal Mask Ogerpon ex + Bug Catching Set + Teal Dance) combined with single-prize nukes (`Tapu Bulu [920]`).
3. **Disruption Resistance**: Dedicated anti-stall switches (`Switch [1123]`, `Latias ex [184]`), hand-recovery engines (`Fezandipiti ex [140]`, `Enriching Energy [13]`), and bench protection (`Battle Cage [1264]`, `Shaymin [343]`).

---

## 6. Verification Method

To independently verify the data and findings in this report:
1. **Inspect Archetypes JSON**:
   ```bash
   uv run python -c "import json; d=json.load(open('.agents/survey_miner_2/panels_compiled.json')); print(list(d.keys()))"
   ```
2. **Verify SQLite Deck Statistics & Archetypes**:
   ```bash
   uv run python -c "import sqlite3; c=sqlite3.connect('file:model/results.db?mode=ro', uri=True).cursor(); c.execute('SELECT id, name, archetype FROM decks WHERE id IN (633, 251, 440)'); print(c.fetchall())"
   ```
3. **Verify Public Deck Files**:
   ```bash
   head -n 20 public_agents/lb1009_mega_lucario_ex_islet/deck.csv
   head -n 20 public_agents/lb826_alakazam_seok/deck.csv
   ```
