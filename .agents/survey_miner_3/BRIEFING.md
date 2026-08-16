# BRIEFING — 2026-08-16T18:59:48Z

## Mission
Provide exact multivariate hypergeometric and combinatorial modeling for Pokémon TCG (opening hand, Mulligan, setup, search access, energy density, prize trade math, and 60-card macro composition).

## 🔒 My Identity
- Archetype: Hypergeometric & Combinatorial Modeler (Teamwork explorer)
- Roles: Combinatorial Modeler, Probability Theorist, Statistical Validator
- Working directory: /Users/alefita/workdir/pokemon-tcg/.agents/survey_miner_3
- Original parent: b3e60c0a-96a8-4187-8566-46966d2c4075
- Milestone: M1 / M2 — Deck Synthesis & Combinatorial Proofs

## 🔒 Key Constraints
- Read-only investigation — do NOT implement in production repo directly except within .agents/survey_miner_3/
- ZERO GPU/MPS/Metal usage. All analysis is combinatorial and mathematical.
- Package management: ALWAYS use `uv run python` if running combinatorial verification scripts.
- KaTeX compliance: All math formulas must be isolated in standalone display blocks `$$ ... $$` on their own lines (never in headings or bold text).

## Current Parent
- Conversation ID: b3e60c0a-96a8-4187-8566-46966d2c4075
- Updated: 2026-08-16T18:59:48Z

## Investigation State
- **Explored paths**:
  - `/Users/alefita/workdir/pokemon-tcg/.agents/ORIGINAL_REQUEST.md`
  - `/Users/alefita/workdir/pokemon-tcg/scratch/probe_hypergeometric.py`
- **Key findings**:
  - Exact multivariate hypergeometric distribution formalized for population N=60, sample n=7 and n=8.
  - P(Setup) >= 92% in single draw requires k >= 18 basics (k=18 -> 93.015%).
  - With tournament mulligan rules (at most 1 mulligan reroll), k >= 10 basics achieves 93.31% cumulative setup probability.
  - Energy density analysis demonstrates that k_e = 12 basic energies achieves 80.94% P(Energy >= 1 in opening 7) and 85.25% by Turn 1 draw (n=8), enabling reliable Teal Dance / manual acceleration.
  - Single-prize attacker inclusion mathematically increases opponent KO requirements from ceil(6/2) = 3 attacks to 4 attacks, enforcing the "7-Prize Game" advantage.
  - Macro-composition sweet spot established: 12-14 Pokemon (10-12 Basics), 34-36 Trainers (10 Search, 8 Supporter, 4 Stadium, 12 Utility/Tech), 12 Energy.
- **Unexplored areas**: None for combinatorial scope.

## Key Decisions Made
- Derived exact fraction combinatorics for all 1-way, 2-way, and 3-way multivariate hypergeometric partitions.
- Formulated the 7th Prize Asymmetry Theorem.

## Artifact Index
- `.agents/survey_miner_3/DISPATCH.md` — Incoming task prompt
- `.agents/survey_miner_3/BRIEFING.md` — Agent memory and state
- `.agents/survey_miner_3/progress.md` — Liveness heartbeat
- `.agents/survey_miner_3/handoff.md` — Final structured 5-component handoff report
