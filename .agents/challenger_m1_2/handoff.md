# Handoff Report — Challenger 2 (Milestone 1)

## 1. Observation

- **Environment & SQLite Target**:
  - SQLite database: `model/results.db` (read-only mode `file:model/results.db?mode=ro`).
  - Total cards indexed in `cards` table: 1,267.
  - Hardware / Compute: 100% CPU execution via `uv run python`. Zero GPU/MPS/Metal allocation.

- **Files Inspected & Queried**:
  - `agent/deck.json` (60 integer card IDs).
  - `experiments/decks/deck_supreme_60.json` (24 unique card entries, summed quantity: 60).
  - `docs/database_schema.md` (verified `cards` table schema).
  - `scratch/validate_m1_deck.py` (executed empirical cross-validation harness).

- **Execution Command & Results**:
  ```bash
  uv run python scratch/validate_m1_deck.py
  ```
  ```text
  SQLITE CROSS-VALIDATION & CARD INTEGRITY AUDIT — MILESTONE 1 (CHALLENGER 2)
  ==========================================================================================
  [*] Loaded agent/deck.json with 60 card IDs.
  [*] Loaded experiments/decks/deck_supreme_60.json with 24 unique card definitions.
  [*] SQLite connection established (URI: file:model/results.db?mode=ro)
  [*] Total cards indexed in SQLite results.db: 1267

  ==========================================================================================
  TEST 1: agent/deck.json Count & Type Validation
  ==========================================================================================
  Deck card count: 60 (Required: 60)
  All IDs integer: True
  -> TEST 1 PASSED: Exactly 60 integer Card IDs in agent/deck.json.

  ==========================================================================================
  TEST 2: Relational Foreign Key Integrity (Every ID exists in `cards` table)
  ==========================================================================================
  Unique Card IDs in deck: 24
  Missing in SQLite:       []
  -> TEST 2 PASSED: 100% of Card IDs resolve to valid records in model/results.db.

  ==========================================================================================
  TEST 3: Quantity & ID Parity between agent/deck.json and deck_supreme_60.json
  ==========================================================================================
  Summed quantity in deck_supreme_60.json: 60 (Required: 60)
  Unique cards in deck_supreme_60.json:   24
  Unique cards in agent/deck.json:        24
  -> TEST 3 PASSED: Exact 1-to-1 card ID and quantity parity verified across both artifacts.

  ==========================================================================================
  TEST 4: Field-by-Field Metadata Parity (SQLite vs deck_supreme_60.json)
  ==========================================================================================
  ID    | Card Name                | DB Stage        | DB Type | DB HP | DB Rule      | Qty | Parity
  ------------------------------------------------------------------------------------------
  96    | Teal Mask Ogerpon ex     | Basic Pokémon   | {G}     | 210   | Pokémon ex   | 4   | VERIFIED
  920   | Tapu Bulu                | Basic Pokémon   | {G}     | 140   | None         | 2   | VERIFIED
  112   | Munkidori                | Basic Pokémon   | {P}     | 110   | None         | 2   | VERIFIED
  140   | Fezandipiti ex           | Basic Pokémon   | {D}     | 210   | Pokémon ex   | 1   | VERIFIED
  184   | Latias ex                | Basic Pokémon   | {P}     | 210   | Pokémon ex   | 1   | VERIFIED
  235   | Budew                    | Basic Pokémon   | {G}     | 30    | None         | 1   | VERIFIED
  1094  | Bug Catching Set         | Item            | None    | None  | None         | 4   | VERIFIED
  1152  | Poké Pad                 | Item            | None    | None  | None         | 4   | VERIFIED
  1121  | Ultra Ball               | Item            | None    | None  | None         | 4   | VERIFIED
  1086  | Buddy-Buddy Poffin       | Item            | None    | None  | None         | 3   | VERIFIED
  1097  | Night Stretcher          | Item            | None    | None  | None         | 3   | VERIFIED
  1118  | Energy Retrieval         | Item            | None    | None  | None         | 2   | VERIFIED
  1123  | Switch                   | Item            | None    | None  | None         | 2   | VERIFIED
  1127  | Tera Orb                 | Item            | None    | None  | None         | 1   | VERIFIED
  1080  | Unfair Stamp             | Item            | None    | None  | ACE SPEC     | 1   | VERIFIED
  1227  | Lillie's Determination   | Supporter       | None    | None  | None         | 4   | VERIFIED
  1182  | Boss’s Orders            | Supporter       | None    | None  | None         | 2   | VERIFIED
  1192  | Carmine                  | Supporter       | None    | None  | None         | 2   | VERIFIED
  1213  | Judge                    | Supporter       | None    | None  | None         | 1   | VERIFIED
  1201  | Briar                    | Supporter       | None    | None  | None         | 1   | VERIFIED
  1264  | Battle Cage              | Stadium         | None    | None  | None         | 2   | VERIFIED
  1     | Basic {G} Energy         | Basic Energy    | {G}     | None  | None         | 10  | VERIFIED
  7     | Basic {D} Energy         | Basic Energy    | {D}     | None  | None         | 2   | VERIFIED
  18    | Grow Grass Energy        | Special Energy  | {G}     | None  | None         | 1   | VERIFIED

  -> TEST 4 PASSED: 100% metadata parity confirmed across all 24 unique card entries.

  ==========================================================================================
  TEST 5: Deck Construction Invariant — Max 4 Copies (Except Basic Energy)
  ==========================================================================================
  ID   96 | Teal Mask Ogerpon ex      | Copies: 4  (Limit <= 4)            -> OK
  ID  920 | Tapu Bulu                 | Copies: 2  (Limit <= 4)            -> OK
  ID  112 | Munkidori                 | Copies: 2  (Limit <= 4)            -> OK
  ID  140 | Fezandipiti ex            | Copies: 1  (Limit <= 4)            -> OK
  ID  184 | Latias ex                 | Copies: 1  (Limit <= 4)            -> OK
  ID  235 | Budew                     | Copies: 1  (Limit <= 4)            -> OK
  ID 1094 | Bug Catching Set          | Copies: 4  (Limit <= 4)            -> OK
  ID 1152 | Poké Pad                  | Copies: 4  (Limit <= 4)            -> OK
  ID 1121 | Ultra Ball                | Copies: 4  (Limit <= 4)            -> OK
  ID 1086 | Buddy-Buddy Poffin        | Copies: 3  (Limit <= 4)            -> OK
  ID 1097 | Night Stretcher           | Copies: 3  (Limit <= 4)            -> OK
  ID 1118 | Energy Retrieval          | Copies: 2  (Limit <= 4)            -> OK
  ID 1123 | Switch                    | Copies: 2  (Limit <= 4)            -> OK
  ID 1127 | Tera Orb                  | Copies: 1  (Limit <= 4)            -> OK
  ID 1080 | Unfair Stamp              | Copies: 1  (Limit <= 4)            -> OK
  ID 1227 | Lillie's Determination    | Copies: 4  (Limit <= 4)            -> OK
  ID 1182 | Boss’s Orders             | Copies: 2  (Limit <= 4)            -> OK
  ID 1192 | Carmine                   | Copies: 2  (Limit <= 4)            -> OK
  ID 1213 | Judge                     | Copies: 1  (Limit <= 4)            -> OK
  ID 1201 | Briar                     | Copies: 1  (Limit <= 4)            -> OK
  ID 1264 | Battle Cage               | Copies: 2  (Limit <= 4)            -> OK
  ID    1 | Basic {G} Energy          | Copies: 10 (Basic Energy: UNLIMITED) -> OK
  ID    7 | Basic {D} Energy          | Copies: 2  (Basic Energy: UNLIMITED) -> OK
  ID   18 | Grow Grass Energy         | Copies: 1  (Limit <= 4)            -> OK
  -> TEST 5 PASSED: Standard format copy rules strictly respected.

  ==========================================================================================
  TEST 6: Deck Construction Invariant — Exactly 1 ACE SPEC Card
  ==========================================================================================
  ACE SPEC cards detected in deck: [(1080, 'Unfair Stamp', 1)]
  -> TEST 6 PASSED: Exactly 1 ACE SPEC card present (Unfair Stamp, ID 1080, quantity 1).

  ==========================================================================================
  TEST 7: Macro Composition & Energy Curve Cross-Validation
  ==========================================================================================
  Pokémon count:  11 (11 Basics: 4 Ogerpon ex, 2 Bulu, 2 Munkidori, 1 Fezandipiti ex, 1 Latias ex, 1 Budew)
  Item count:     24 (Search, Recovery, Switch, 1 ACE SPEC)
  Supporter count:10 (Draw & Gust)
  Stadium count:   2 (Battle Cage)
  Energy count:   13 (10 Grass, 2 Darkness, 1 Grow Grass)
  Total Cards:    60
  -> TEST 7 PASSED: Macro distribution matches tournament specifications.

  ==========================================================================================
  FINAL EMPIRICAL VERDICT: CONFIRMED
  100% of structural, relational, and rule-based invariants are verified.
  ==========================================================================================
  ```

## 2. Logic Chain

1. **Card ID Existence & Schema Grounding**:
   - Direct execution of `SELECT id, name, category, stage, hp, energy_type, weakness, rule, metadata_complete FROM cards WHERE id IN (...)` confirmed all 24 unique card IDs exist in `model/results.db`.
   - Every single entry has `metadata_complete = 1`.

2. **Cross-Artifact Parity**:
   - The card collection in `agent/deck.json` contains exactly 60 integer IDs.
   - The collection in `experiments/decks/deck_supreme_60.json` has `card_count = 60` and sums to exactly 60 cards across 24 unique entries.
   - The multiset histogram `Counter(agent_deck)` is identical to `{c['id']: c['quantity'] for c in supreme_cards}` across all 24 keys.

3. **Metadata Parity (100%)**:
   - `name`: 100% exact match across all 24 cards (including diacritics in `Poké Pad`, `Lillie's Determination`, `Boss’s Orders`).
   - `hp`: 100% exact match (e.g. 210 HP for Ogerpon ex, Fezandipiti ex, Latias ex; 140 for Tapu Bulu; 110 for Munkidori; 30 for Budew; null for trainers/energy).
   - `type` / `energy_type`: 100% match (`{G}`, `{P}`, `{D}`, and null).
   - `stage`: 100% match (`Basic Pokémon`, `Item`, `Supporter`, `Stadium`, `Basic Energy`, `Special Energy`).
   - `rule`: 100% match (`Pokémon ex` for IDs 96, 140, 184; `ACE SPEC` for ID 1080; null for remainder).

4. **Rule Constraints Validation**:
   - **4-Copy Limit**: Non-Basic Energy cards have quantities: 4x (Ogerpon ex, Bug Catching Set, Poké Pad, Ultra Ball, Lillie's Determination), 3x (Buddy-Buddy Poffin, Night Stretcher), 2x (Tapu Bulu, Munkidori, Energy Retrieval, Switch, Boss's Orders, Carmine, Battle Cage), 1x (Fezandipiti ex, Latias ex, Budew, Tera Orb, Unfair Stamp, Judge, Briar, Grow Grass Energy). None exceeds 4. Basic {G} Energy has 10 copies and Basic {D} Energy has 2 copies.
   - **ACE SPEC Limit**: Exactly 1 ACE SPEC exists in the entire deck list: Card ID 1080 (`Unfair Stamp`), quantity 1.

## 3. Caveats

- **SQLite `category` column semantics**: In SQLite `cards` table, the column `category` is used for special sub-archetypes/mechanics (such as `Tera(Grass)` for Card ID 96), whereas macro classifications (`Basic Pokémon`, `Item`, `Supporter`, `Stadium`, `Basic Energy`, `Special Energy`) are stored in `stage`. Both the macro stage and sub-mechanics are fully aligned.
- No other caveats.

## 4. Conclusion

**FINAL VERDICT: CONFIRMED**

The deck specifications in `agent/deck.json` and `experiments/decks/deck_supreme_60.json` satisfy 100% of the relational, physical, mathematical, and tournament legality invariants against `model/results.db`.

## 5. Verification Method

To independently verify all findings on CPU:

```bash
uv run python scratch/validate_m1_deck.py
```

Expected exit code: `0`.
Invariants guaranteed:
- 60 valid integer Card IDs.
- Zero missing records in `cards` table.
- 100% metadata parity (name, stage, type, HP, rule).
- Standard format 4-copy rule compliant.
- Exactly 1 ACE SPEC card (`Unfair Stamp` ID 1080).
- Summed quantities match 1-to-1.
