# Pokémon TCG MLX Migration — Complete Implementation Plan

> **Goal:** Migrate Mikaelzinho from PyTorch to MLX with full semantic correctness, Phases A–F + end-to-end pipeline.

---

## Progress Tracker

| Phase | Status | Commits | Tests |
|-------|--------|---------|-------|
| Pre-Flight | ✅ Done | 1 | — |
| A — Canonical Contract | ✅ Done | 5 | 10/10 |
| B — Semantic P0 Fixes | ✅ Done | 2 | 7/7 |
| C — FP16-Native Trainer | ✅ Done | 1 | 7/7 |
| D — Data & Shapes | ✅ Done | 1 | 11/11 |
| E — Inference Semantics | ⬜ Pending | — | — |
| F — Minimal Recurrence | ⬜ Pending | — | — |
| Pipeline — Build + Train | ⬜ Pending | — | — |
| Submission | ⬜ Pending | — | — |

## Execution Rules

1. **Modify existing files.** `rl/policy_mlx.py`, `scripts/bc/bc_train_mlx.py`, `agent/main.py` are edited in-place. No parallel trainers, no "v2" copies.
2. **Tests go in `scripts/validate/`**.
3. **Sequential agents on `develop` branch.** Each agent commits, next starts from that commit.
4. **`uv run` for everything.** No raw `python`/`python3`.
5. **Use real data only.** No synthetic mocks. Smoke tests limit volume via `configs/smoke.json` (`max_episodes`, `max_rows`).
6. **Config hierarchy:** CLI args > `--config` file > `configs/train_config.json` > hardcoded defaults.
7. **Entrypoints:** `tcg-data`, `tcg-build-bc`, `tcg-build-daily`, `tcg-train`, `tcg-evaluate`, `tcg-tournament`, `tcg-submission`.
8. **No PyTorch fallback** in agent or trainer. MLX-only.
9. **No sys.path hacks** (except Kaggle sandbox compat in `agent/main.py`).
10. **Consult wikifita** for project context and uv best practices.

---

## Pre-Flight: Branch + Environment ✅

**Agent 0 — Branch Init**
- Create `develop` from `main`
- `uv sync`
- Verify Metal: `uv run python -c "import mlx.core as mx; print(mx.default_device())"`
- Verify existing encoding test: `uv run python scripts/validate/test_encoding.py`
- Commit: `chore: initialize develop branch for MLX migration`

---

## Phase A ✅ — Canonical Contract

### A.1 — Token Schema Module

**Files:** create `rl/token_schema.py`, modify `rl/policy_mlx.py`

1. Create `rl/token_schema.py` with canonical token-type IDs (matching PyTorch reference exactly):
   ```python
   T_CLS = 0
   T_SELF_DECK = 1; T_OPP_DECK = 2
   T_SELF_PRIZE = 3; T_OPP_PRIZE = 4
   T_SELF_HAND = 5; T_OPP_HAND = 6
   T_SELF_DISC = 7; T_OPP_DISC = 8
   T_STADIUM = 9
   T_SELF_ACTIVE = 10; T_SELF_BENCH = 11
   T_OPP_ACTIVE = 12; T_OPP_BENCH = 13
   T_OPT = 14; T_EFFECT = 15
   T_SEL_TYPE = 16; T_SEL_CTX = 17
   T_CARD_SYNTH = 18
   N_TTYPES = 19
   ```
2. Create `scripts/validate/test_token_schema.py` — asserts all IDs distinct, range 0..18, matches reference
3. Commit: `feat(A.1): centralize token-type schema`

### A.2 — Fix Token-Type Collision

**Files:** modify `rl/policy_mlx.py`

1. Import from `rl/token_schema.py`
2. Fix line ~330: `("opp", (13, 13))` → `("opp", (T_OPP_ACTIVE, T_OPP_BENCH))` = `(12, 13)`
3. Replace ALL hardcoded type IDs in `_encode()`, `_opt_stream()`, `_resolve()`, `_card_stream()`, scratch tokens with named constants
4. Add assertion in `scripts/validate/test_token_schema.py`: build model, inspect type embeddings for self_active vs opp_active — must differ
5. Commit: `fix(A.2): correct opp unit token-type collision`

### A.3 — Architecture Config Versioning

**Files:** modify `rl/policy_mlx.py`, modify `scripts/bc/bc_train_mlx.py`

1. Add `ARCH_VERSION = "1.0.0"` and `get_config()` method to `TokenTransformerMLX`
2. Modify checkpoint save to include: `arch_config`, `gstep`, `seed`, `dataset_path`
3. Modify checkpoint load to validate `arch_config` — raise on mismatch
4. Add `scripts/validate/test_checkpoint.py` — round-trip test
5. Commit: `feat(A.3): versioned architecture config in checkpoints`

### A.4 — Synthetic Data Generator

**Files:** create `scripts/validate/make_synthetic_data.py`

1. Standalone script that generates a directory of `.npy` files with correct keys/shapes
2. Uses `rl.encoder.TokenEncoder.int_keys` to get the key list
3. Generates random but valid card IDs, action masks (≥1 legal per row), labels
4. `uv run python scripts/validate/make_synthetic_data.py --rows 1000 --out data/bc_data/synthetic_1k`
5. Commit: `feat(A.4): synthetic dataset generator for smoke tests`

### A.5 — Phase A Validation

**Files:** create `scripts/validate/test_phase_a.py`

1. End-to-end: load synthetic data → build model → forward pass → verify shapes → save/load checkpoint
2. Run all validations: `uv run python -m pytest scripts/validate/ -v`
3. Commit: `test(A.5): Phase A integration — canonical contract verified`

---

## Phase B ✅ — Semantic P0 Fixes

### B.1 — Additive Attention Mask

**Files:** modify `rl/policy_mlx.py` (`_encode` method)

1. Change mask from boolean to additive float:
   ```python
   # FROM: attn_mask = (~pad)[:, None, None, :]
   # TO:
   attn_mask = mx.where(pad[:, None, None, :], -1e9, 0.0)
   ```
2. Add `scripts/validate/test_attention_mask.py`:
   - Forward pass with right-padded input
   - Verify real token outputs are identical regardless of padding
3. Commit: `fix(B.1): additive attention mask`

### B.2 — MHA Bias

**Files:** modify `rl/policy_mlx.py` (`TransformerEncoderLayerMLX.__init__`)

1. Verify MLX `nn.MultiHeadAttention` bias default
2. Explicitly set `bias=True` if not default
3. Add check in `test_attention_mask.py`: verify projections have bias params
4. Commit: `fix(B.2): explicit MHA bias`

### B.3 — Padding ID Zero

**Files:** modify `rl/policy_mlx.py`

1. Add `_card_emb()` method that masks index 0 to zero:
   ```python
   def _card_emb(self, ids):
       emb = self.card_emb(ids)
       return emb * (ids != 0).astype(mx.float16)[..., None]
   ```
2. Replace all `self.card_emb(...)` calls with `self._card_emb(...)` in `_card_stream`, `_unit_stream`, `_resolve`
3. Add `scripts/validate/test_padding_zero.py` — verify id=0 → zero vector
4. Commit: `fix(B.3): padding_idx=0 semantics for card embeddings`

### B.4 — Static Table Immutability

**Files:** modify `rl/policy_mlx.py`

1. Verify `card_feat` and `atom_support` are plain mx.arrays (not in `parameters()`)
2. Add guard: after optimizer step, assert values unchanged
3. Add check in `test_padding_zero.py`
4. Commit: `fix(B.4): static table immutability guard`

### B.5 — Categorical Value Head

**Files:** modify `rl/policy_mlx.py` (`logits_value` method)

1. When `value_categorical=True`, return scalar expectation:
   ```python
   atom_logits = self.value_head(v_in)
   probs = mx.softmax(atom_logits, axis=-1)
   value = (probs * self.atom_support).sum(axis=-1)
   ```
2. Add check in `scripts/validate/test_phase_b.py`
3. Commit: `fix(B.5): categorical value returns scalar expectation`

### B.6 — Fix Validation Loss

**Files:** modify `scripts/bc/bc_train_mlx.py` (validation section ~line 350)

1. Replace `log(softmax)` with proper CE: `-(logit[label] - logsumexp(logits))`
2. Add `scripts/validate/test_validation_loss.py` — parity with `nn.losses.cross_entropy`
3. Commit: `fix(B.6): proper cross-entropy validation loss`

### B.7 — Phase B Validation

**Files:** create `scripts/validate/test_phase_b.py`

1. Full semantic check: additive mask, padding zero, static immutability, categorical value, finite gradients
2. Run all validations
3. Commit: `test(B.7): Phase B — all P0 semantic fixes verified`

---

## Phase C ✅ — FP16-Native Trainer

### C.1 — Remove FP16→FP32 Round-Trip

**Files:** modify `scripts/bc/bc_train_mlx.py` (`batches()` function)

1. Keep numeric features as fp16, labels/masks as int32/float32:
   ```python
   ob = {k: mx.array(np.asarray(arrs[k][b]).astype(
           np.int32 if k in int_keys
           else (np.float16 if k not in ("action_mask",) else np.float32)))
         for k in keys}
   ```
2. Add fp16 check in test_phase_b or new test
3. Commit: `feat(C.1): native fp16 data pipeline`

### C.2 — Gradient Accumulation

**Files:** modify `scripts/bc/bc_train_mlx.py` (training loop)

1. Add `--accum-steps` argument (default=1, recommended=4)
2. Refactor `train_step` into accumulation loop:
   - Forward + backward K microbatches → accumulate grads in FP32
   - Normalize by total examples, clip once, update once
3. Scheduler counts optimizer steps: `total_opt_steps = epochs * ceil(batches / accum_steps)`
4. `accum_steps=1` preserves existing behavior (backward compatible)
5. Commit: `feat(C.2): FP32 gradient accumulation`

### C.3 — Graph-Safe Gradient Clipping

**Files:** modify `scripts/bc/bc_train_mlx.py`

1. Replace `float()` norm check with MLX-graph-safe version:
   ```python
   gn = mx.sqrt(sum(mx.sum(g ** 2) for g in flat_grads))
   scale = mx.where(gn > max_norm, max_norm / mx.maximum(gn, 1e-6), 1.0)
   ```
2. `mx.eval` at optimizer update boundary
3. Commit: `feat(C.3): graph-safe gradient clipping`

### C.4 — Correct Scheduler + Resume

**Files:** modify `scripts/bc/bc_train_mlx.py`

1. `total_steps` spans ALL epochs × accumulation
2. Save `gstep` in checkpoint, restore on resume
3. Commit: `fix(C.4): correct total step computation and scheduler resume`

### C.5 — Complete Checkpoint

**Files:** modify `scripts/bc/bc_train_mlx.py`

1. Save: model params, optimizer state (if serializable), arch_config, epoch, gstep, val_acc, seed, dataset_path
2. Resume validates arch_config
3. Commit: `feat(C.5): complete checkpoint with full trainer state`

### C.6 — Conservative Slab Default

**Files:** modify `scripts/bc/bc_train_mlx.py`

1. Default `--slab-rows` 262144 → 32768
2. Commit: `fix(C.6): default slab-rows 32k for M3 Pro`

### C.7 — Phase C Validation

**Files:** create `scripts/validate/test_phase_c.py`

1. Generate 1000-row synthetic dataset
2. Train 1 epoch: fp16, accum_steps=4, slab-rows=500
3. Verify loss decreases, checkpoint saves/loads, all fields present
4. Run all validations
5. Commit: `test(C.7): Phase C — FP16 trainer smoke test`

---

## Phase D ✅ — Data and Shapes

### D.1 — Option Bucket Compaction

**Files:** modify `rl/policy_mlx.py` (`_encode` method)

1. Implement finite buckets `(32, 64, 128, 192)` — same as PyTorch reference
2. Compute `max_legal` per batch, round up to bucket
3. Truncate option tokens, preserve SUBMIT + masks
4. Add `scripts/validate/test_compaction.py`
5. Commit: `feat(D.1): option bucket compaction`

### D.2 — State Column Compaction

**Files:** modify `rl/policy_mlx.py` (`_encode` method)

1. Detect all-padding state columns, remove them
2. Remap `opt_src_pos`/`opt_tgt_pos` after compaction
3. Always retain header (CLS/value/submit/select), scratch, referenced positions
4. Add test in `test_compaction.py`: verify real-token outputs preserved
5. Commit: `feat(D.2): exact state column compaction`

### D.3 — Episode Metadata

**Files:** modify `scripts/bc/build_bc_dataset.py`, modify `scripts/bc/build_bc_from_zips.py`

1. Add `episode_meta.npy` sidecar output with: episode_id, side, step_id, decision_id, substep, new_episode, terminal, reward
2. Dataset builder emits metadata alongside existing arrays
3. Add check in `test_phase_d.py`: metadata consistency
4. Commit: `feat(D.3): episode metadata sidecar in dataset builder`

### D.4 — Episode-Level Validation Split

**Files:** modify `scripts/bc/bc_train_mlx.py`

1. If `episode_meta.npy` exists, split at episode boundaries instead of raw row tail
2. Fallback to current behavior if no metadata (backward compatible)
3. Commit: `feat(D.4): episode-level validation split`

### D.5 — Phase D Validation

**Files:** create `scripts/validate/test_phase_d.py`

1. Synthetic data with metadata → compaction → train 1 epoch → verify
2. Commit: `test(D.5): Phase D — compaction and data hygiene`

---

## Phase E ⬜ — Inference Semantics

### E.1 — Complete Logs in Agent

**Files:** modify `agent/main.py` (`choose()` function)

1. Change `obs_for_encode = {"select": select, "current": current, "logs": []}` to pass actual logs:
   ```python
   obs_for_encode = {"select": select, "current": current, "logs": obs.get("logs", [])}
   ```
2. Commit: `fix(E.1): pass complete observation logs to trackers`

### E.2 — Autoregressive Multi-Select

**Files:** modify `agent/main.py` (`choose()` function)

1. Replace `topk(count)` with autoregressive loop:
   - Pick one option → update picked set → rebuild mask → next forward → repeat until SUBMIT or max_count
2. Add `scripts/validate/test_autoregressive.py`:
   - Verify sequential conditional picks, no duplicates, min/max enforcement
3. Commit: `feat(E.2): autoregressive multi-select inference`

### E.3 — FP16 Inference Path

**Files:** modify `agent/main.py`

1. MLX model uses fp16 numeric tensors (not fp32 round-trip)
2. Commit: `feat(E.3): fp16 inference path`

### E.4 — Submission Validation

**Files:** modify `agent/main.py` if needed

1. `uv run python scripts/build_submission.py`
2. Verify self-contained bundle (no external deps)
3. Run `scripts/evaluate.py` smoke test
4. Commit: `feat(E.4): submission bundle validated`

### E.5 — Phase E Validation

**Files:** create `scripts/validate/test_phase_e.py`

1. End-to-end: synthetic model → autoregressive multi-select → verify
2. Run all validations
3. Commit: `test(E.5): Phase E — inference semantics verified`

---

## Phase F ⬜ — Minimal Recurrence

### F.1 — Memory API in Model

**Files:** modify `rl/policy_mlx.py`

1. Add `memory_in` parameter to `_encode()` and `logits_value()`
2. When `memory_in` provided, use it as scratch token input instead of learned init
3. Return `memory_out` (scratch encoder outputs)
4. Add `learned_init` parameter (learned initial register state)
5. Add `scripts/validate/test_memory_api.py`:
   - `memory_in=None` → same output as before
   - `memory_out` shape matches `memory_in`
   - Round-trip: `memory_out` from step t feeds step t+1
6. Commit: `feat(F.1): memory API with persistent scratch registers`

### F.2 — Memory in Agent

**Files:** modify `agent/main.py`

1. Add `"memory": None` to per-side tracker dict
2. On deck submission (new match), reset memory to None
3. On each decision, carry memory forward: `logits, value, mem_out = model.logits_value(ob, memory_in=st["memory"])`
4. Sides don't share memory
5. Commit: `feat(F.2): per-match per-side memory in agent`

### F.3 — TBPTT Support in Trainer

**Files:** modify `scripts/bc/bc_train_mlx.py`

1. Add `--tbptt-chunk` argument (default=0 = disabled)
2. When enabled + `episode_meta.npy` exists:
   - Iterate chunks in episode order (don't shuffle within episode)
   - Carry memory between chunks, `stop_gradient` at boundaries
   - Mask padded timesteps, normalize loss by real decisions
3. When disabled, existing shuffled behavior preserved
4. Commit: `feat(F.3): TBPTT training support in trainer`

### F.4 — Phase F Validation

**Files:** create `scripts/validate/test_phase_f.py`

1. Memory API: persists, resets, isolates between sides/matches
2. TBPTT: memory flows between chunks, stop_gradient at boundaries, no cross-episode leakage
3. Counterfactual: same observation + different history → different memory state
4. Run all validations
5. Commit: `test(F.6): Phase F — minimal recurrence verified`

---

## Final ⬜ — Smoke Training

**Agent FINAL**

1. Generate 1000-row synthetic dataset
2. Train 1 quick epoch with all features: fp16, accum_steps=4, additive mask, compaction, memory API
3. Verify: loss finite + decreasing, checkpoint round-trip, all tests green
4. `uv run python scripts/build_submission.py`
5. Commit: `feat: MLX migration complete — Phase A-F validated`

---

## Agent Dispatch Table

| # | Agent | Phase | Depends | Files Modified | Commit |
|---|-------|-------|---------|----------------|--------|
| 0 | Init | Pre | — | git | `chore:` |
| A.1 | Token Schema | A | 0 | `rl/token_schema.py` ★new, `scripts/validate/test_token_schema.py` ★new | `feat(A.1):` |
| A.2 | Fix Collision | A | A.1 | `rl/policy_mlx.py` | `fix(A.2):` |
| A.3 | Config Version | A | A.2 | `rl/policy_mlx.py`, `scripts/bc/bc_train_mlx.py`, `scripts/validate/test_checkpoint.py` ★new | `feat(A.3):` |
| A.4 | Synthetic Data | A | A.3 | `scripts/validate/make_synthetic_data.py` ★new | `feat(A.4):` |
| A.5 | Phase A Test | A | A.4 | `scripts/validate/test_phase_a.py` ★new | `test(A.5):` |
| B.1 | Additive Mask | B | A | `rl/policy_mlx.py`, `scripts/validate/test_attention_mask.py` ★new | `fix(B.1):` |
| B.2 | MHA Bias | B | B.1 | `rl/policy_mlx.py` | `fix(B.2):` |
| B.3 | Padding Zero | B | B.2 | `rl/policy_mlx.py`, `scripts/validate/test_padding_zero.py` ★new | `fix(B.3):` |
| B.4 | Static Guard | B | B.3 | `rl/policy_mlx.py` | `fix(B.4):` |
| B.5 | Categ Value | B | B.4 | `rl/policy_mlx.py` | `fix(B.5):` |
| B.6 | Val Loss | B | B.5 | `scripts/bc/bc_train_mlx.py` | `fix(B.6):` |
| B.7 | Phase B Test | B | B.6 | `scripts/validate/test_phase_b.py` ★new | `test(B.7):` |
| C.1 | FP16 Pipeline | C | B | `scripts/bc/bc_train_mlx.py` | `feat(C.1):` |
| C.2 | Grad Accum | C | C.1 | `scripts/bc/bc_train_mlx.py` | `feat(C.2):` |
| C.3 | Safe Clip | C | C.2 | `scripts/bc/bc_train_mlx.py` | `feat(C.3):` |
| C.4 | Scheduler | C | C.3 | `scripts/bc/bc_train_mlx.py` | `fix(C.4):` |
| C.5 | Checkpoint | C | C.4 | `scripts/bc/bc_train_mlx.py` | `feat(C.5):` |
| C.6 | Slab Default | C | C.5 | `scripts/bc/bc_train_mlx.py` | `fix(C.6):` |
| C.7 | Phase C Test | C | C.6 | `scripts/validate/test_phase_c.py` ★new | `test(C.7):` |
| D.1 | Opt Compact | D | C | `rl/policy_mlx.py`, `scripts/validate/test_compaction.py` ★new | `feat(D.1):` |
| D.2 | State Compact | D | D.1 | `rl/policy_mlx.py` | `feat(D.2):` |
| D.3 | Metadata | D | D.2 | `scripts/bc/build_bc_dataset.py`, `scripts/bc/build_bc_from_zips.py` | `feat(D.3):` |
| D.4 | Val Split | D | D.3 | `scripts/bc/bc_train_mlx.py` | `feat(D.4):` |
| D.5 | Phase D Test | D | D.4 | `scripts/validate/test_phase_d.py` ★new | `test(D.5):` |
| E.1 | Complete Logs | E | D | `agent/main.py` | `fix(E.1):` |
| E.2 | Autoregress | E | E.1 | `agent/main.py`, `scripts/validate/test_autoregressive.py` ★new | `feat(E.2):` |
| E.3 | FP16 Infer | E | E.2 | `agent/main.py` | `feat(E.3):` |
| E.4 | Submission | E | E.3 | `agent/main.py` | `feat(E.4):` |
| E.5 | Phase E Test | E | E.4 | `scripts/validate/test_phase_e.py` ★new | `test(E.5):` |
| F.1 | Memory API | F | E | `rl/policy_mlx.py`, `scripts/validate/test_memory_api.py` ★new | `feat(F.1):` |
| F.2 | Agent Memory | F | F.1 | `agent/main.py` | `feat(F.2):` |
| F.3 | TBPTT | F | F.2 | `scripts/bc/bc_train_mlx.py` | `feat(F.3):` |
| F.4 | Phase F Test | F | F.3 | `scripts/validate/test_phase_f.py` ★new | `test(F.4):` |
| FINAL | Smoke | All | F | — | `feat:` |

**Total: 31 agents, 31 commits, sequential on `develop`.**

★new = new files. All others modify existing files in-place.

---

## Key Dependencies

```
mlx>=0.32.0     — MultiHeadAttention, nn.Embedding, mx.compile, optimizers
numpy>=2.5.1    — memmap, structured arrays
torch>=2.13.0   — reference only (not used in MLX path)
kaggle-environments>=1.32.0 — submission validation
```

## Smoke Test Data

No real dataset on disk. `scripts/validate/make_synthetic_data.py` generates correctly-shaped `.npy` arrays for pipeline validation. Real data (19 GB replay zips) is used only for actual training runs after migration is complete.
