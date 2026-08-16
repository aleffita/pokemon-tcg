# Tournament Evaluation Matrix — Deck Candidates vs Panel

**Author:** Antigravity Deck Swarm (Gemini 3.7 Flash High)  
**Consumer:** Codex Tournament Runner (GPT-5.6-Luna-Max)  
**Date:** 2026-08-16  

---

## 1. Candidate Deck Roster (Exact 60-Card Arrays)

| Deck Label | Artifact Location | Primary Tactical Focus | Key Diff from v0 |
| :--- | :--- | :--- | :--- |
| **`deck_supreme_v0`** | `experiments/decks/deck_supreme_60.json` | Balanced Hybrid (Ogerpon / Tapu Bulu / Munkidori) | Baseline (4x Lillie, 2x Carmine, 1x Judge, 1x Briar, 1x Grow Grass) |
| **`deck_v1_tempo`** | `experiments/decks/candidates/deck_v1_anti_lucario_tempo.json` | Max T1 Velocity vs Mega Lucario (`lb1009`/`lb945`) | +2 Carmine (4x total), -1 Briar, -1 Basic Grass (9x total) |
| **`deck_v2_control`** | `experiments/decks/candidates/deck_v2_anti_control_lock.json` | Hard Hand Lock & Snipe vs Alakazam (`lb826`) | +1 Judge (2x total), +1 Boss (3x total), +1 Munkidori (3x total), +1 Darkness Energy (3x total), -2 Carmine, -2 Basic Grass |

---

## 2. Recommended Tournament Matrix (Smallest Informative Surface)

For each candidate deck evaluated with frozen Stage 4 root weights:

```text
Opponent Panel (5 games per matchup, paired seeds when possible):
1. lb1009_mega_lucario_ex_islet  (5 games) -> Benchmark vs Fast Aggro
2. lb945_multiply_ivan           (5 games) -> Benchmark vs Fast Aggro
3. lb826_alakazam_seok           (5 games) -> Benchmark vs Powerful Hand Scaling
4. lb814_crustle_emre            (5 games) -> Benchmark vs ex-Immunity Wall
5. first_sub_kaggle_2707         (5 games) -> Baseline Competence
6. random_baseline               (5 games) -> Sanity & No-Crash Check

Total per candidate: 30 games.
```

### Evaluation Protocol:
1. Codex executes candidate runs sequentially using `--deck experiments/decks/candidates/<deck_name>.json`.
2. Output W-L-D per opponent recorded in tournament JSON.
3. Antigravity will ingest the tournament logs upon completion to calculate the empirical matchup delta and propose refined candidate adjustments.
