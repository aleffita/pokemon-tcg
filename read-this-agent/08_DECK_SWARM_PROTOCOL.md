# 08 — Deck Strategy Swarm & Inter-Agent Integration Protocol

This document establishes the communication and artifact contract between the **Antigravity Cognitive Swarm (Gemini 3.7 Flash High)** and the **Codex Autoresearch / Optimization Engine (GPT-5.6-Luna-Max)**.

---

## 1. Executive Division of Labor

```
+───────────────────────────────────────────────────────────────────────────────+
|                           THE TWO COGNITIVE ENGINES                           |
|                                                                               |
|  [Codex Engine: GPT-5.6-Luna-Max]  ──────► Neural Architecture & Optimization |
|                                            - Policy Updates & GRPO Training   |
|                                            - Trajectory Collection On-Policy  |
|                                            - Full Tournament Harness Exec     |
|                                            - Dedicated Apple Silicon Compute  |
|                                                                               |
|  [Antigravity Swarm: Gemini 3.7]   ──────► Tactical 60-Card Deck Engineering  |
|                                            - Card-by-Card SQLite Data Mining  |
|                                            - Hypergeometric Opening Prob      |
|                                            - Adversarial Meta-Counter Matrix  |
|                                            - Zero Hardware Contention (Read)  |
+───────────────────────────────────────────────────────────────────────────────+
```

---

## 2. Resource Protection Guarantee (Zero Compute Contention)

The Antigravity Deck Swarm commits to:
1. **No background training processes**: Zero GPU/MPS/Metal or PyTorch training loops launched by Antigravity.
2. **Read-only database access**: Read-only SQLite queries to `model/results.db` without locking tables or interfering with live tournament logging.
3. **Dedicated Compute**: 100% of the M3 Pro hardware compute remains dedicated to Codex's autoresearch sweeps and tournament runs.

---

## 3. Deck Artifacts & Integration Path

Candidate decks produced by the Antigravity Swarm will be emitted to:

```text
experiments/decks/
  ├── DECK_SUPREME_60.md        <-- Complete tactical monograph and card justification
  ├── deck_supreme_60.json      <-- Array of exactly 60 Card IDs [int, ...]
  └── candidates/
        ├── deck_agro_yan_633.json
        ├── deck_control_meta_counter.json
        └── ...
```

### Consumption by Tournament Runner:
The Codex coordinator or tournament scripts can immediately ingest any candidate deck by referencing the JSON path or copying to `agent/deck.json`.

---

## 4. Competitive Target

- **Frozen Ladder Window**: August 16 to August 31, 2026.
- **Fitness Criteria**:
  - $P(\text{Valid T1 Setup}) \ge 92\%$.
  - Prize-trade efficiency against 2-prize Pokémon ex.
  - Invariant win-rate robustness across the full external panel (`lb1009`, `lb945`, `lb826_alakazam_seok`, `lb814`, Lucario, Dragapult, and baselines).
