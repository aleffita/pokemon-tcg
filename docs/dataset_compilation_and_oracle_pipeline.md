# Dataset Compilation & Oracle Pipeline Specification

**Document**: Level 2 Data Engineering & Simulation Specification  
**Author**: Fitalabs AI Research  
**Target Ingestion**: GPT-5.6 Sol, DeepSeek-V4-Pro, Codex, Claude 3.7  
**Date**: August 14, 2026  

---

## 1. Kaggle Environments Replay Parsing & Temporal Realignment

The raw replay ingestion pipeline (`scripts/bc/build_bc_dataset.py` and `scripts/bc/build_bc_from_zips.py`) processes raw daily JSON episodes downloaded from the Kaggle competition backend.

```
+---------------------------------------------------------------------------------------------------+
|                               RAW REPLAY TO PARQUET FUNNEL                                        |
|                                                                                                   |
|  [Raw Episode JSON]                                                                              |
|         │                                                                                         |
|         ▼                                                                                         |
|  1. Semantic Validation Filter : Drop draws, missing rewards, or non-terminal states               |
|         │                                                                                         |
|         ▼                                                                                         |
|  2. Off-By-One Pointer Realignment : Align observation(t) action_mask with label(t+1)            |
|         │                                                                                         |
|         ▼                                                                                         |
|  3. Engine Oracle Simulation : Compute would_ko via C++ engine lookup                             |
|         │                                                                                         |
|         ▼                                                                                         |
|  4. Telescoping Backward Reward Calculation : Derive prize delta, return, and terminal targets    |
|         │                                                                                         |
|         ▼                                                                                         |
|  5. Columnar Parquet Serialization : Write chunked row-groups with FP16/FP32 static checksums    |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Temporal Realignment (Off-By-One Correction)

In Kaggle Environments, when agent $i$ acts at timestep $t$, the execution result and environmental state change are recorded in observation $t+1$. 

Naively pairing `obs[t]` with `action[t]` introduces a destructive 1-step lag where the model attempts to predict actions that have already resolved.

### Realignment Algorithm
1. Extract step observation sequence:

$$
\mathcal{O} = [o_0, o_1, \dots, o_T]
$$

2. Extract action event sequence:

$$
\mathcal{A} = [a_0, a_1, \dots, a_{T-1}]
$$

3. Pair decision point with strictly synchronous state:

$$
(s_t, a_t) = (o_t, a_t) \quad \text{such that} \quad a_t \in \text{valid\_actions}(o_t)
$$

4. If `action[t]` is not in `valid_actions(o_t)`, reject the transition tuple as an asynchronous frame drop.

---

## 3. Telescoping Backward Rewards & Auxiliary Targets

To avoid double-counting intermediate step rewards in long combo chains (e.g., searching deck, playing energy, evolving, then attacking), the pipeline computes backward telescoping targets.

### 3.1. Turn-Local Targets (`aux_prize_delta` & `aux_ko`)
For any decision step $t$ within turn $k$:

$$
\text{aux\_prize\_delta}_t = \Delta \text{Prizes}_{\text{self}}(k) - \Delta \text{Prizes}_{\text{opp}}(k)
$$

$$
\text{aux\_ko}_t = \mathbb{I}(\Delta \text{Prizes}_{\text{self}}(k) > 0)
$$

These targets are identical for all decision steps within the same turn cascade, training the attention trunk to evaluate the net prize yield of the entire planned combo.

### 3.2. Sequential Return Target (`aux_return`)
To train value estimates, the return target accumulates discounted future transitions without turn-local repetition:

$$
R_t = \sum_{l=t}^{T-1} \gamma^{l-t} r_l + \gamma^{T-t} r_{\text{terminal}}
$$

Where:
- Transition reward:

$$
r_l = \frac{(\text{Prizes\_Me}_l - \text{Prizes\_Me}_{l+1}) - (\text{Prizes\_Opp}_l - \text{Prizes\_Opp}_{l+1})}{6.0}
$$

- Terminal reward:

$$
r_{\text{terminal}} = \begin{cases} +1.0 & \text{if Victory} \\ -1.0 & \text{if Defeat} \end{cases}
$$

---

## 4. Engine Oracle Simulation (`bc_would_ko`)

Instead of forcing the neural network to learn non-linear damage arithmetic, type weaknesses, resistance calculations, and tool modifiers from scratch, the data builder delegates damage simulation to the native C++ engine.

```
Decision Step t: Valid Action = Attack(idx)
       │
       ▼
[Search Agent Engine Binding: annotate_would_ko_with_audit]
       │
       ├── 1. Query Active Attacker Base Damage
       ├── 2. Apply Weakness Multiplier (2x) / Resistance (-20 / -30)
       ├── 3. Evaluate Attached Tool Effects (e.g., Choice Belt +30)
       └── 4. Check Defender HP Remaining:
              Estimated_Damage >= Defender_HP ? would_ko = 1 : would_ko = 0
       │
       ▼
Inject into option feature vector: a_opt[would_ko_index] = 1.0
```

---

## 5. Hierarchical Parquet KV Cache Architecture

During TBPTT training on Apple Silicon M3 Pro, sequential multi-step episodes must be delivered in $O(1)$ without causing SSD paging thrash.

```
+---------------------------------------------------------------------------------------------------+
|                                 _ParquetRowGroupCache TOPOLOGY                                    |
|                                                                                                   |
|  [Dataloader Lane Request: Row i .. i+32]                                                         |
|         │                                                                                         |
|         ├── Hit in Hot Zone?       ───> Return NumPy Slice in 0.0001s (RAM)                        |
|         │                                                                                         |
|         ├── Hit in Transient Zone? ───> Promote to Hot Zone, Return Slice (RAM)                   |
|         │                                                                                         |
|         └── Cache Miss?                                                                           |
|                 ├── Read Parquet Row-Group from Disk (SSD)                                        |
|                 ├── Decode to Contiguous NumPy Tensors                                            |
|                 ├── Insert into Hot Zone                                                          |
|                 └── If Total Cache > 10GB: Evict LRU Blocks to .cache_spill/                      |
+---------------------------------------------------------------------------------------------------+
```

---

## 6. Target Dataset Schema for RoPEND & MoE (Next Sprint)

To enable the transition to 4D RoPEND and Mixture of Experts, the next dataset generation run (`build_bc_dataset.py`) must serialize the following schema extensions into the Parquet files:

| Target Column | Type | Mathematical Mapping / Description |
| :--- | :--- | :--- |
| `match_step` | `int16` | Discrete step index $c_1 \in [0, 200]$ |
| `meta_epoch_day` | `int16` | Days elapsed since competition launch $c_2 = \text{Date} - T_0$ |
| `time_remaining_s` | `float32` | Normalized time budget remaining $c_3 = \frac{T_{\text{remain}}}{600.0} \in [0.0, 1.0]$ |
| `inferred_opponent_elo` | `float32` | Normalized opponent Elo rating $c_4 = \frac{R_{\text{opp}} - 600.0}{600.0}$ |
| `team_identity_hash` | `int32` | Deterministic Murmur3 hash of opponent team identifier |
| `vehicle_deck_card_ids` | `list[int16]` | Exactly 60 integers representing the player's own deck list for pre-game draft |
