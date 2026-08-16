# Forensic Audit Report — Milestone 1 (Pokémon TCG AI Project)

**Auditor:** `auditor_m1` (Forensic Auditor)  
**Profile:** General Project / Integrity Forensics  
**Integrity Mode:** Development  
**Date:** 2026-08-16T19:07:30Z  
**Verdict:** **CLEAN**

---

## 1. Observation

Direct empirical observations and execution logs gathered during forensic analysis:

### 1.1 Tool Execution Traces

#### Trace A: Automated Pytest Suite Execution
```text
$ uv run pytest tests/test_deck_m1_validation.py -v
============================= test session starts ==============================
platform darwin -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0 -- /Users/alefita/workdir/pokemon-tcg/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /Users/alefita/workdir/pokemon-tcg
configfile: pyproject.toml
plugins: anyio-4.14.2
collecting ... collected 1 item

tests/test_deck_m1_validation.py::test_deck_validation PASSED            [100%]

============================== 1 passed in 0.01s ===============================
```

#### Trace B: Independent Python Forensic Audit Probe
```text
$ uv run python scratch/forensic_auditor_m1_probe.py
================================================================================
               MILESTONE 1 FORENSIC INTEGRITY AUDIT SUITE                       
================================================================================

>>> AUDIT CHECK 1: Zero GPU / MPS / Metal Contention
  [✓] SQLite connection: strictly read-only mode verified (file:model/results.db?mode=ro).
  [✓] Process execution: 100% CPU/RAM on Apple Silicon (zero Metal/MPS shader allocations).
  [PASS] AUDIT CHECK 1 PASSED.

>>> AUDIT CHECK 2: Authentic SQLite Database Parity
  [✓] Exact 1-to-1 parity between agent/deck.json and deck_supreme_60.json verified.

  --- Verifying All 24 Unique Card IDs in model/results.db ---
  ID     | Database Name              | DB Stage         | DB Type  | DB HP  | DB Rule      | Qty 
  ------------------------------------------------------------------------------------------
  96     | Teal Mask Ogerpon ex       | Basic Pokémon    | {G}      | 210    | Pokémon ex   | 4   
  920    | Tapu Bulu                  | Basic Pokémon    | {G}      | 140    | None         | 2   
  112    | Munkidori                  | Basic Pokémon    | {P}      | 110    | None         | 2   
  140    | Fezandipiti ex             | Basic Pokémon    | {D}      | 210    | Pokémon ex   | 1   
  184    | Latias ex                  | Basic Pokémon    | {P}      | 210    | Pokémon ex   | 1   
  235    | Budew                      | Basic Pokémon    | {G}      | 30     | None         | 1   
  1094   | Bug Catching Set           | Item             | None     | None   | None         | 4   
  1152   | Poké Pad                   | Item             | None     | None   | None         | 4   
  1121   | Ultra Ball                 | Item             | None     | None   | None         | 4   
  1086   | Buddy-Buddy Poffin         | Item             | None     | None   | None         | 3   
  1097   | Night Stretcher            | Item             | None     | None   | None         | 3   
  1118   | Energy Retrieval           | Item             | None     | None   | None         | 2   
  1123   | Switch                     | Item             | None     | None   | None         | 2   
  1127   | Tera Orb                   | Item             | None     | None   | None         | 1   
  1080   | Unfair Stamp               | Item             | None     | None   | ACE SPEC     | 1   
  1227   | Lillie's Determination     | Supporter        | None     | None   | None         | 4   
  1182   | Boss’s Orders              | Supporter        | None     | None   | None         | 2   
  1192   | Carmine                    | Supporter        | None     | None   | None         | 2   
  1213   | Judge                      | Supporter        | None     | None   | None         | 1   
  1201   | Briar                      | Supporter        | None     | None   | None         | 1   
  1264   | Battle Cage                | Stadium          | None     | None   | None         | 2   
  1      | Basic {G} Energy           | Basic Energy     | {G}      | None   | None         | 10  
  7      | Basic {D} Energy           | Basic Energy     | {D}      | None   | None         | 2   
  18     | Grow Grass Energy          | Special Energy   | {G}      | None   | None         | 1   
  ------------------------------------------------------------------------------------------
  [✓] All 60/60 card instances exist authentically in model/results.db (zero synthetic IDs).
  [PASS] AUDIT CHECK 2 PASSED.

>>> AUDIT CHECK 3: No Synthetic / Facade Data (Hypergeometric Verification)
  - Setup (n=7): Calculated 1137524/1462905 (77.7579%) vs JSON 1137524/1462905
  - Mulligan (n=7): Calculated 325381/1462905 (22.2421%) vs JSON 325381/1462905
  - Setup within 1 mulligan: Calculated 2034218243864/2140091039025 (95.0529%) vs JSON 2034218243864/2140091039025
  - Mulligan within 1 mulligan: Calculated 105872795161/2140091039025 (4.9471%) vs JSON 105872795161/2140091039025
  - T1 Energy (n=7): Calculated 9797437/11703240 (83.7156%) vs JSON 9797437/11703240
  - T1 Search Access (n=7): Calculated 74479/76995 (96.7323%) vs JSON 74479/76995
  [✓] All hypergeometric probability values are mathematically exact and non-synthetic.
  [PASS] AUDIT CHECK 3 PASSED.

>>> AUDIT CHECK 4: Deck Rules Integrity
  [✓] Rule 1: Deck contains exactly 60 cards.
  [✓] Rule 2: Max 4 copies per card name respected for all non-Basic Energy cards.
  [✓] Rule 3: Exactly 1 ACE SPEC card present: Unfair Stamp.
  [✓] Rule 4: Exactly 11 Basic Pokémon present (>= 10 requirement met).
  [PASS] AUDIT CHECK 4 PASSED.
```

---

## 2. Logic Chain

1. **Check 1 — Zero GPU / MPS / Metal Contention:**
   - *Observation:* `manage_task(Action='list')` reported 0 background tasks. Process memory and execution occurred purely on CPU without CUDA/MPS context initialization. SQLite query was performed using strict URI read-only connection (`file:model/results.db?mode=ro`).
   - *Conclusion:* 100% of M3 Pro compute resources remain uncompromised for Codex autoresearch and self-play.

2. **Check 2 — Authentic SQLite Database Parity:**
   - *Observation:* Every Card ID in `agent/deck.json` and `experiments/decks/deck_supreme_60.json` (24 unique IDs, 60 total slots) was queried against `model/results.db` in table `cards`.
   - *Result:* All 24 IDs exist with identical metadata (Name, Stage, Type, HP, Rule). Zero synthetic or placeholder IDs exist.

3. **Check 3 — No Synthetic / Facade Data (Hypergeometric Proofs):**
   - *Observation:* For $N=60, n=7, K_b=11, K_e=13, K_s=22$:
     - $P(\text{Setup } n=7) = 1 - \frac{\binom{49}{7}}{\binom{60}{7}} = \frac{1137524}{1462905} \approx 77.7579\%$
     - $P(\text{Mulligan } n=7) = \frac{\binom{49}{7}}{\binom{60}{7}} = \frac{325381}{1462905} \approx 22.2421\%$
     - $P(\text{Setup within 1 mulligan}) = 1 - \left(\frac{325381}{1462905}\right)^2 = \frac{2034218243864}{2140091039025} \approx 95.0529\% \ge 92.0\%$
     - $P(\text{Mulligan within 1 mulligan}) = \left(\frac{325381}{1462905}\right)^2 = \frac{105872795161}{2140091039025} \approx 4.9471\% \le 8.0\%$
   - *Result:* All values stored in `deck_supreme_60.json` match the analytical calculations to 8+ decimal places and exact rational fractions.

4. **Check 4 — Official Deck Rules Compliance:**
   - *Observation:*
     - Total card count: Exactly 60 cards.
     - Copy limit: All non-Basic Energy cards have $\le 4$ copies (max is 4 copies for Teal Mask Ogerpon ex, Bug Catching Set, Poké Pad, Ultra Ball, Lillie's Determination).
     - ACE SPEC limit: Exactly 1 ACE SPEC card (`Unfair Stamp`, ID 1080).
     - Basic Pokémon minimum: Exactly 11 Basic Pokémon ($\ge 10$ required).
   - *Result:* Full compliance with official Pokémon TCG tournament rules.

---

## 3. Caveats

- **No Caveats:** All checks were empirically re-verified through independent Python code and automated test execution against the read-only SQLite database.

---

## 4. Conclusion

- **Audit Verdict:** **CLEAN**
- The Milestone 1 deliverable satisfies all integrity criteria, database parity constraints, mathematical consistency standards, and official tournament rules.

---

## 5. Verification Method

To independently reproduce this verification:

```bash
# 1. Run the test suite
uv run pytest tests/test_deck_m1_validation.py -v

# 2. Run the independent forensic audit probe
uv run python scratch/forensic_auditor_m1_probe.py
```
