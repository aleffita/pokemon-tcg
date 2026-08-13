# Magnum Opus Pipeline: Kaggle PTCG AI Battle

This document serves as the architectural contract for the new segregated pipeline, designed specifically to exploit the rules and physics of the Kaggle environment during the "Locked Meta" phase (August 16-31).

## 1. The Core Philosophy: Pilot vs. Vehicle
The model is not evaluated in a vacuum. A model with high validation accuracy but low win-rate is not flawed; it is a master pilot tied to a weak vehicle (Deck). The new architecture decouples these concepts and grants the model **Situational Awareness**.

## 2. Dimensional Expansion: RoPEND
We implement **N-Dimensional Rotary Positional Embeddings (RoPEND)** to map the space-time continuum of the Kaggle ecosystem without axis interference:
- **Axis 1 (Sequential):** Match Turn/Step.
- **Axis 2 (Chronological):** Meta-Epoch (Calendar Date).
- **Axis 3 (Urgency):** Adaptive Time Compute (Countdown from 600s).
- **Axis 4 (Hierarchy):** Elo / Team Identity.

## 3. The "Apex Predator" Mechanics (Airgap Hacking)
The agent will break out of the "Vending Machine" paradigm via two systemic hacks:
- **Temporal Awakening:** The inference script (`act()`) reads the OS clock via `datetime.now(UTC)`. If `Date >= 2026-08-16`, the agent activates the **Apex Mode Token**, recognizing that the meta is frozen and every opponent is a static target.
- **The Elo Anchor (Stochastic Inference):** The Kaggle sandbox is ephemeral (memoryless between matches). Instead of saving state, the agent *derives* its expected current standing statistically in real-time. On August 16th, a protocol submission injects the exact Anchor Elo. During inference, the agent combines: 1) The Anchor Elo, 2) Days passed since Aug 16 (via OS clock), 3) Expected match volume distribution, and 4) The *current* opponent's estimated Elo (via Aux Heads). With this, it mathematically infers its expected leaderboard standing dynamically, altering its mid/late-game MoE routing.

## 4. Architectural Paradigms
- **Autoregressive Draft:** The first output of the network is the prediction of its own 60-card deck (Data Augmentation). It learns the synergy of its vehicle before making the first move.
- **Base Model Initialization:** The "Hero" checkpoint is a legacy architecture and CANNOT be used for Logit Distillation or surgical freezing. The correct path is to train a new Base Model from scratch (or upcycle from the highly compatible Curriculum V1 Stage 4) using the new architecture. Once the Base Model is validated, it is expanded into the MoE router.

## 5. ETL & The Elite Pool
The dataset will NOT be a raw dump. It will be an adversarial extraction of the **Elite Pool**:
- Filtering only matches where at least one agent is Elo >= 1100 (approx. ~100k matches).
- Data will be strictly orthogonal to prevent gradient confusion across Elo bands.
