# Empirical Technical Report: Auxiliary Target Heads & C++ Damage Oracle

**Author**: Oracle and Aux Head Explorer (Milestone 2)  
**Date**: 2026-08-14  
**Workspace**: `/Users/alefita/workdir/pokemon-tcg`  
**Target Modules**: `rl/search_agent.py`, `scripts/bc/build_bc_dataset.py`, `scripts/bc/bc_train_mlx.py`, `rl/policy_mlx.py`, `rl/policy_infer_torch.py`, `rl/encoder/`

---

## 1. Executive Summary

This investigation delivers an empirical audit of the 4 auxiliary target heads (`aux_ko`, `aux_prize_delta`, `aux_terminal`, `aux_return`), `aux_valid` masking, and the native C++ damage oracle (`bc_would_ko`) in the Pokémon TCG AI framework.

### Core Verified Findings

1. **Auxiliary Target Decoupling**:
   - `aux_ko` and `aux_prize_delta` operate as **turn-local lookahead targets** over the remaining window of the current turn.
   - `aux_return` operates as a **telescoping transition discounted return** ($\gamma = 1.0$), ensuring that multi-step actions inside one turn never double-count prize acquisitions in the return sum.
   - `aux_terminal` explicitly flags the final step of the episode ($d = T - 1$).
   - `aux_valid` masks out unparsable states and guarantees 0 loss contribution during backpropagation.

2. **Native C++ Engine Oracle (`bc_would_ko`)**:
   - Bound directly to `cg.api` (`search_begin`, `search_step`, `search_release`, `search_end`).
   - Executes 1-ply determinized rollouts with seeded sampling (`n_var = 10` for variable attacks, 1 rollout for fixed attacks).
   - Early stopping triggers when 3 consecutive rollouts yield identical outcomes (`len(seen) == 1`).
   - Resolves post-attack sub-selects via `_advance_resolve` using minimum count bounds (`minCount`) and seeded sampling up to 64 steps.

3. **Option Feature Offsets & Encodings**:
   - `_opt_struct(o)` produces a 15-dimensional vector per option (`OPT_STRUCT = 15`).
   - `would_ko` features occupy indices `[11:14]` (`OPT_WK = 11`):
     - `opt_attr[:, 11]` = `would_ko` ($\text{KO rate} \in [0, 1]$)
     - `opt_attr[:, 12]` = `would_ko_prizes` ($\min(\text{prizes} / 6.0, 1.0) \in [0, 1]$)
     - `opt_attr[:, 13]` = `would_ko_win` ($P(\text{ends game}) \in [0, 1]$)
     - `opt_attr[:, 14]` = `already_picked` flag (`OPT_PICKED = 14`).

4. **Automated Test Suite Verification**:
   - `scripts/validate/test_would_ko_dataset.py`: 9/9 unit tests passed (1.22s).
   - `scripts/validate/test_aux_targets.py`: 6/6 unit tests passed (<0.01s).

---

## 2. Mathematical Formulation of Auxiliary Target Heads

### 2.1. Target Specifications

| Target | Target Tensor | Formulation | Loss Function | Default Loss Weight | Empirical Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `aux_ko` | `[B]` float32 | $\mathbb{I}(\Delta \text{Prizes}_{\text{self}}(\text{turn}) > 0 \lor \Delta \text{Prizes}_{\text{opp}}(\text{turn}) > 0)$ | Binary Cross-Entropy | `cfg.aux_ko_weight = 0.5` | 9.39% |
| `aux_prize_delta` | `[B]` float32 | $\Delta \text{Prizes}_{\text{self}}(\text{turn}) - \Delta \text{Prizes}_{\text{opp}}(\text{turn})$ | Mean Squared Error | `cfg.aux_prize_weight = 0.25` | 8.91% non-zero |
| `aux_terminal` | `[B]` float32 | $\mathbb{I}(t = T - 1)$ | Binary Cross-Entropy | `cfg.aux_terminal_weight = 0.25` | 1.52% |
| `aux_return` | `[B]` float32 | $R_t = \sum_{l=t}^{T-1} r_l + r_{\text{term}}$ | Mean Squared Error | `cfg.aux_return_weight = 0.25` | 99.99% non-zero |
| `aux_valid` | `[B]` float32 | $\mathbb{I}(\text{State is parsable and valid})$ | Row Mask (0/1) | N/A (Masking multiplier) | 100.00% parsed |

### 2.2. Computation in `scripts/bc/build_bc_dataset.py`

In `_compute_aux_targets(states, outcome)`:
- State extraction reads `obs["current"]["players"][*]["prize"]` count.
- Turn-local lookahead computes:
  ```python
  prizes_i_took = s["my_prize"] - end_state["my_prize"]
  prizes_opp_took = s["opp_prize"] - end_state["opp_prize"]
  aux[s["i"]] = {
      "aux_ko": 1 if (prizes_i_took != 0 or prizes_opp_took != 0) else 0,
      "aux_prize_delta": float(prizes_i_took - prizes_opp_took),
      "aux_terminal": int(is_terminal),
      "aux_valid": 1,
  }
  ```
- Step-to-step transition rewards compute:
  ```python
  prizes_i_took_step = s["my_prize"] - nxt["my_prize"]
  prizes_opp_took_step = s["opp_prize"] - nxt["opp_prize"]
  reward = float(prizes_i_took_step - prizes_opp_took_step) / 6.0 + terminal_bonus
  ```
- Cumulative backward suffix sum computes `aux_return`:
  ```python
  running = 0.0
  for d in reversed(valid_idx):
      running += reward_by_i[s["i"]]
      entry["aux_return"] = running
  ```

### 2.3. MLX Loss Function & Masking in `scripts/bc/bc_train_mlx.py`

In `_aux_loss(aux_dict, aux_targets, weights)`:
```python
ko_bce = mx.logaddexp(mx.array(0.0, dtype=mx.float32), ko_logit) - ko_tgt * ko_logit
terminal_bce = (
    mx.logaddexp(mx.array(0.0, dtype=mx.float32), terminal_logit)
    - terminal_tgt * terminal_logit
)
prize_mse = (prize_pred - prize_tgt) ** 2
return_mse = (return_pred - return_tgt) ** 2

loss = (
    weights["ko"] * mx.sum(valid * ko_bce)
    + weights["prize"] * mx.sum(valid * prize_mse)
    + weights["terminal"] * mx.sum(valid * terminal_bce)
    + weights["return"] * mx.sum(valid * return_mse)
)
```

Unweighted validation metrics in `_aux_metrics`:
```python
valid_sum = max(float(valid.sum()), 1.0)
return {
    "aux_ko_bce": float((valid * ko_bce).sum() / valid_sum),
    "aux_prize_mse": float((valid * prize_mse).sum() / valid_sum),
    "aux_terminal_bce": float((valid * terminal_bce).sum() / valid_sum),
    "aux_return_mse": float((valid * return_mse).sum() / valid_sum),
}
```

---

## 3. C++ Damage Oracle Architecture (`bc_would_ko`)

### 3.1. Engine API Flow in `rl/search_agent.py`

```
  ┌────────────────────────────────────────────────────────┐
  │         Observation (obs) + Agent Deck (60 IDs)        │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Filter: sel["type"] == 0 and has attack options (opts) │
  └──────────────────────────┬─────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   Fixed Attack (ndet=1)             Variable Attack (ndet=10)
            │                                 │
            └────────────────┬────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ 1. Determinization: det = _determinize(obs, deck, rng) │
  │ 2. C++ Engine Init: ss = api.search_begin(obs, **det)  │
  │ 3. Attack Step: st = api.search_step(ss.searchId, [a]) │
  │ 4. Sub-select Resolution: _advance_resolve(...)        │
  │ 5. Outcome Recording: (ko_bit, took_prizes, won)       │
  │ 6. C++ Engine Release: api.search_release(ss.searchId) │
  │ 7. Early Stopping: break if trials >= 3 and len(seen)==1│
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Compute Option Triplet:                                │
  │ - would_ko = kos / trials                              │
  │ - would_ko_prizes = prize_sum / trials                 │
  │ - would_ko_win = wins / trials                         │
  │ Write onto options in-place via write_would_ko()       │
  └────────────────────────────────────────────────────────┘
```

### 3.2. Sub-Select Resolution (`_advance_resolve`)

The function `_advance_resolve(api, sid, obs, me, rng, audit, max_steps=64)` steps through non-branching post-attack resolution states:
- Extracts `min_count = int(sel.get("minCount", 1) or 0)`.
- Randomly samples `min_count` options via `rng.sample(range(len(opts)), min_count)`.
- Increments audit counters (`resolved_subselects`, `resolved_subselect_choices`, `ambiguous_subselects`).
- Steps the engine until `obs["current"]["yourIndex"] != me` or game ends.

### 3.3. Failure Isolation & Provenance Audit

`annotate_would_ko_with_audit` produces detailed audit structures:
- `search_begin_failures`, `simulation_failures`, `search_release_failures`, `search_end_failures`.
- Explicit tracking per option: `requested_trials`, `valid_trials`, `failed_trials`, `computed`, `zero_result`.
- Prevents conflation of failed simulations with true zero-damage attacks.

---

## 4. Option Feature Layout & Offsets

### 4.1. Structural Attribute Layout (`_opt_struct`)

The option attribute vector `opt_attr[i]` has dimension 15:

| Index | Field Name | Type | Description |
| :--- | :--- | :--- | :--- |
| `0` | `count` | float32 | `min(count / 5.0, 1.0)` |
| `1` | `number` | float32 | `min(number / 15.0, 1.0)` |
| `2` | `attack_dmg` | float32 | `min(base_dmg, 350) / 350.0` |
| `3` | `attack_var` | float32 | `is_variable` flag (0.0 / 1.0) |
| `4` | `attack_cost` | float32 | `min(energy_cost, 5) / 5.0` |
| `5` | `attack_eff` | float32 | `has_effect` flag (0.0 / 1.0) |
| `6..10` | `special_cond` | float32[5] | One-hot special condition (poison, burn, asleep, paralyzed, confused) |
| `11` | `would_ko` | float32 | KO / prize-take rate $\in [0, 1]$ (`OPT_WK = 11`) |
| `12` | `would_ko_prizes` | float32 | `min(expected_prizes / 6.0, 1.0)` (`OPT_WK + 1 = 12`) |
| `13` | `would_ko_win` | float32 | $P(\text{ends game in our favor}) \in [0, 1]$ (`OPT_WK + 2 = 13`) |
| `14` | `already_picked` | float32 | Multi-select buffered indicator flag (`OPT_PICKED = 14`) |

Total structural length: `OPT_STRUCT = 2 + 4 + 5 + 3 + 1 = 15`.

---

## 5. Verification & Test Suite Execution

### 5.1. Test Command Execution Results

1. **Would-KO Dataset Tests**:
   - Command: `uv run python -m unittest scripts/validate/test_would_ko_dataset.py`
   - Output: `Ran 9 tests in 1.221s. OK.`
   - Validated: Zero vs failure distinction, subselect minCount resolution, audit metadata sidecar, deduplication, worker config propagation, shape metadata flattening.

2. **Auxiliary Targets Tests**:
   - Command: `uv run python -m unittest scripts/validate/test_aux_targets.py`
   - Output: `Ran 6 tests in 0.000s. OK.`
   - Validated: Turn and prize count extraction, missing prize handling, return non-double-counting, terminal bonus sign, invalid state zeroing.

---

## 6. Recommendations for Downstream Milestones

1. **Elite Pool Ingestion (M2)**: Ensure `WOULD_KO = True` and `WK_NVAR = 10` are configured when re-compiling the Elite match dataset (`Elo >= 1100`).
2. **MoE Forward Integration (M1/M5)**: Ensure the 4 auxiliary prediction heads (`ko_head_aux`, `prize_head_aux`, `terminal_head_aux`, `return_head_aux`) continue to read from the global representation (e.g. `cls_out` or `extra[0]`) without adding routing overhead.
3. **Inference Airgap**: Confirm `bc_would_ko` is enabled in `agent/main.py` when evaluating checkpoints that utilize `would_ko` option features.
