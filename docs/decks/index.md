---
type: hub
title: "Deck Architecture & Meta Engineering Hub"
description: "Master index of 60-card deck architectures, adversarial matchup analyses, hypergeometric proofs, and tournament empirical evaluations for Pokémon TCG AI."
tags: [pokemon-tcg, deck, meta, architecture, tournaments, hypergeometric, red-team]
timestamp: "2026-08-16T19:30:00-03:00"
---

# Deck Architecture & Meta Engineering Hub

## Purpose

This section documents the systematic engineering, combinatorial modeling, and empirical tournament validation of 60-card deck architectures for the Kaggle Pokémon TCG AI Challenge.

## Index of Deck Strategy & Empirical Records

1. **[[archetype_meta_analysis]]** — Tactical dissection of the 6 dominant leaderboard archetypes (`lb1009`, `lb945`, `lb826`, `lb814`, `lb510`, and Baseline #251).
2. **[[tournament_matrix_and_empirical_learnings]]** — Empirical analysis of the AR-028 round-robin tournament series comparing `deck_supreme_v0`, `deck_v1_tempo`, `deck_v2_control`, and `deck_v3_apex_sovereign`.
3. **[`DECK_SUPREME_60.md`](file:///Users/alefita/workdir/pokemon-tcg/experiments/decks/DECK_SUPREME_60.md)** — Master engineering monograph of the 60-card Supreme baseline.
4. **[`ROUND_ROBIN_EVAL_SPEC.md`](file:///Users/alefita/workdir/pokemon-tcg/experiments/decks/ROUND_ROBIN_EVAL_SPEC.md)** — Specification of the multi-opponent tournament testing protocol.

## Deck Candidate Ledger

| Version | File | Primary Function | Status |
| :--- | :--- | :--- | :--- |
| **v0** | `experiments/decks/deck_supreme_60.json` | Balanced Hybrid (Ogerpon / Tapu Bulu / Munkidori) | Validated (21.67% panel WR) |
| **v1** | `experiments/decks/candidates/deck_v1_anti_lucario_tempo.json` | Max T1 Carmine Ramp vs Mega Lucario | Tested (1 win vs lb945) |
| **v2** | `experiments/decks/candidates/deck_v2_anti_control_lock.json` | Hard Hand Lock & Snipe vs Alakazam/Crustle | Tested (60% WR vs Crustle, 20% vs Alakazam) |
| **v3** | `experiments/decks/candidates/deck_v3_apex_sovereign.json` | Apex Sovereign Fusion (Carmine + Judge + Boss + Munkidori) | Synthesized & Queued |
