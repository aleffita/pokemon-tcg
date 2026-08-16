# Execution Plan: 60-Card Supreme Deck Tactical Engineering & Codex Protocol Synchronization

## Architecture & Swarm Topology
- **Orchestrator**: Dispatch-only coordinator in `.agents/orchestrator/`.
- **Compute Isolation**: 100% CPU/GPU preservation for Codex (GPT-5.6-Luna-Max) on Apple Silicon M3 Pro. Zero GPU/MPS usage. All subagent work is combinatorial analysis, hypergeometric optimization, read-only SQLite mining, and artifact authoring.
- **Package Management**: All script executions must use `uv run python`.
- **Database Access**: Read-only on `model/results.db`.

## Feature Inventory & Requirement Mapping
| # | Feature / Deliverable | Source Requirement | Milestone | Target Path |
|---|---|---|---|---|
| 1 | SQLite Meta & Card Mining (Deck #633, Deck #251, high-Elo card synergies) | R1 | Survey Phase | `.agents/survey_miner_1/` |
| 2 | Empirical Opponent Panel Analysis (6 archetypes from AR-019..025) | R3 | Survey Phase | `.agents/survey_miner_2/` |
| 3 | Hypergeometric & Resource Curve Modeling (P(Setup)>=92%, P(Mulligan)<=8%) | R2 | Survey Phase | `.agents/survey_miner_3/` |
| 4 | Combinatorial 60-Card Deck Optimization & Integer ID Validation | R1, R2, Acceptance Criteria | Milestone 1 | `agent/deck.json` |
| 5 | Deck Capsule JSON with Metadata & Resource Probabilities | R4, Acceptance Criteria | Milestone 1 | `experiments/decks/deck_supreme_60.json` |
| 6 | 60-Slot Rationale, Hypergeometric Proof & Red Team Matchup Monograph | R2, R3, Acceptance Criteria | Milestone 2 | `experiments/decks/DECK_SUPREME_60.md` |
| 7 | Codex Swarm Protocol Synchronization with Hashes & Locations | R4, Acceptance Criteria | Milestone 3 | `read-this-agent/08_DECK_SWARM_PROTOCOL.md` |
| 8 | Forensic Integrity & Zero-Contention Audit | Hard Constraints, Acceptance Criteria | Milestone 3 | `.agents/auditor_m3/` |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| 0 | Survey Phase | 3 parallel explorers/spec miners querying SQLite, meta archetypes, and hypergeometric formulations | None | IN_PROGRESS |
| 1 | M1: 60-Card Supreme Deck & Capsule | Synthesize 60 SQLite card IDs, create `agent/deck.json` and `experiments/decks/deck_supreme_60.json` | Milestone 0 | PLANNED |
| 2 | M2: Monograph & Hypergeometric Proof | Author comprehensive monograph `experiments/decks/DECK_SUPREME_60.md` with 60-slot rationale and 6-matchup red team analysis | Milestone 1 | PLANNED |
| 3 | M3: Codex Protocol Sync & Integrity Forensics | Author `read-this-agent/08_DECK_SWARM_PROTOCOL.md`, run forensic audit (zero GPU, hash validation, ID parity) | Milestone 2 | PLANNED |

## Interface Contracts & Validation Criteria
- `agent/deck.json`: JSON array of exactly 60 positive integers. Every integer must exist in `cards.id` of `model/results.db`.
- `experiments/decks/deck_supreme_60.json`: Valid JSON dictionary containing `archetype`, `deck_name`, `card_count` (60), `card_list` (array of objects with `id`, `name`, `quantity`, `category`), `energy_curve`, and `hypergeometric_probabilities`.
- `experiments/decks/DECK_SUPREME_60.md`: Full markdown monograph with KaTeX isolated equations (`$$...$$`), all 60 card slots explained, exact hypergeometric formulas and values, and strategic matchup plans against the 6 panel archetypes.
- `read-this-agent/08_DECK_SWARM_PROTOCOL.md`: Protocol sync document with SHA256 checksums of all generated deck artifacts.
