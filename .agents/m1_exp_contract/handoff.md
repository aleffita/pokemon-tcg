# Handoff Report: Policy Integration, FP32 Precision Contract & MLX Training Pipeline

## 1. Observation

### 1.1 Existing PyTorch Inference and Contract Infrastructure
Direct inspection of `rl/policy_infer_torch.py` (lines 24-65, 142-233, 414-506, 549-609) reveals:
- **Strict Format Identifier**: Line 24 specifies `TORCH_INFERENCE_FORMAT = "ptcg-torch-fp32-v1"`.
- **Static Feature Contract & SHA256 Integrity**: Lines 41-66 implement `_checkpoint_static_features()`:
  ```python
  array = np.asarray(features, dtype=np.float32)
  digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
  if digest != contract.get("sha256"):
      raise ValueError("checkpoint static feature table hash mismatch")
  csv_digest = _sha256_file(card_table.csv_path)
  if csv_digest != contract.get("card_csv_sha256"):
      raise ValueError("runtime card table does not match the checkpoint's encoder contract")
  ```
- **Strict FP32 Parameter Enforcement**: Lines 594-606 enforce that every tensor in `state_dict` is strictly `torch.float32`:
  ```python
  if tensor.is_floating_point() and tensor.dtype != torch.float32:
      raise ValueError(f"state_dict[{key!r}] is {tensor.dtype}, expected torch.float32")
  ```
- **Finite Masking Value**: Line 398-399 uses finite `-65504.0` instead of `-1e9` to prevent overflow in mixed environments:
  ```python
  return logits.masked_fill(o["action_mask"] < 0.5, -65504.0), value, memory_out
  ```
- **Recurrent Memory Interface**: Lines 214-217 define `scratch` and `learned_init` with shape `(16, d_model)` (`d_model=128`).
- **Auxiliary Heads**: Lines 189-192 define auxiliary heads (`ko_head_aux`, `prize_head_aux`, `terminal_head_aux`, `return_head_aux`) mapped as `nn.Linear(self.d, 1).to(torch.float32)`.

### 1.2 Existing MLX Policy and Training Architecture
Inspection of `rl/policy_mlx.py` (lines 82-217, 351-608, 633-705) and `scripts/bc/bc_train_mlx.py` (lines 108-266, 275-348, 369-412) reveals:
- **MLX Transformer Trunk**: `TransformerEncoderMLX` (lines 62-76) stacks `TransformerEncoderLayerMLX` instances (`norm_first=False`, post-LayerNorm).
- **Token Stream Aggregation**: Constructs CLS, split value/submit, select type/context, meta context (`day_proj`, `agent_bucket_emb`, `deck_bucket_emb`), card streams, vortex unit streams, scratch registers (16 tokens), and option streams.
- **Compaction & Position Remapping**: Compaction removes columns padded across all batch rows (lines 459-526) and remaps `opt_src_pos`/`opt_tgt_pos` before resolving option candidate tokens.
- **Split Optimizer Architecture**:
  - `FP32StateMuon` (lines 108-121) maintains momentum `state["v"]` in strict `mx.float32`.
  - `FP32StateAdamW` (lines 123-136) maintains moments `m, v` in strict `mx.float32`.
  - `PathSafeMultiOptimizer` (lines 139-174) routes 2D hidden weights to Muon, structured verb heads to high-decay AdamW ($\lambda = 0.1$), and embeddings/biases to standard AdamW.
- **Auxiliary Multi-Task Supervision**: `_aux_loss()` (lines 275-314) computes BCE on `aux_ko` and `aux_terminal`, and MSE on `aux_prize_delta` and `aux_return`, masked by `aux_valid`.

### 1.3 Target Magnum Opus MoE Architecture Blueprint
Inspection of `docs/architecture/moe_pipeline_blueprint.md`, `docs/architecture/01_ropend_theory.md`, `docs/architecture/02_stochastic_elo_inference.md`, and `docs/neural_engine_and_tokenization_spec.md` reveals:
- **4D RoPEND Operator**: Partitions $D=128$ into 4 heads of $d_k=32$, with 4 orthogonal Givens rotation planes per head ($c_1$: Step, $c_2$: Meta-Epoch, $c_3$: Urgency Clock, $c_4$: Inferred Elo).
- **Top-2 MoE 4-Expert Topology**: Replaces monolithic MLP ($128 \to 512 \to 128$) with 4 specialized experts (Agro, Control, Setup, Endgame) and a Top-2 gating router with load balancing loss $\mathcal{L}_{\text{balance}} = \alpha_{\text{balance}} E \sum_{e=1}^E f_e P_e$.
- **Vehicle Cross-Attention Draft**: Encodes 60-card self-deck autoregressively prior to step 0, generating a vehicle context embedding $v_{\text{vehicle}} \in \mathbb{R}^{128}$.
- **Apex Mode Airgap Trigger**: Dynamic temporal activation when `datetime.now(timezone.utc) >= 2026-08-16T00:00:00Z` dropping routing temperature to $\tau = 0.1$.

---

## 2. Logic Chain

1. **FP16 Collapse Cause & Prevention**:
   - The historical collapse to 3.3% Win Rate occurred because FP16 activations underflowed in attention softmax and pointer action masking when large negative masks were applied.
   - Using strict FP32 tensor allocation in PyTorch inference (`rl/policy_infer_torch.py` and `rl/policy_moe_torch.py`) and MLX training (`scripts/bc/bc_train_mlx.py`) prevents numerical underflow, preserving dynamic range in attention probabilities and action logits.
2. **Unified RoPEND Integration in Attention**:
   - Standard PyTorch `nn.TransformerEncoderLayer` encapsulates attention projections internally, making rotary position insertion impossible without a custom multi-head attention module.
   - A unified custom `RoPENDMultiHeadAttention` in both PyTorch (`rl/ropend/ropend_torch.py` / `rl/policy_moe_torch.py`) and MLX (`rl/ropend/ropend_mlx.py` / `rl/policy_moe_mlx.py`) allows direct application of 4D Givens rotations to queries $\mathbf{Q}$ and keys $\mathbf{K}$ prior to scaled dot-product computation.
3. **MoE Top-2 Routing & Load Balancing Integration**:
   - Replacing the monolithic feedforward network with 4 specialized FFN experts and a Top-2 router per layer enables functional specialization without inflating inference FLOPs (only 2 of 4 experts active per token).
   - In training, `load_balance_loss` ($\mathcal{L}_{\text{balance}}$) is accumulated alongside policy cross-entropy and auxiliary predictive losses to prevent expert starvation or collapse.
   - Router gating weights $W_g \in \mathbb{R}^{128 \times 4}$ must be updated by `FP32StateAdamW`, while expert linear weights ($128 \to 512$ and $512 \to 128$) are routed to `FP32StateMuon`.
4. **Vehicle Cross-Attention Draft Integration**:
   - The vehicle draft encoder processes the 60-card vehicle deck at match initialization.
   - Its pooled vehicle embedding $\mathbf{v}_{\text{vehicle}} \in \mathbb{R}^{128}$ is injected as an invariant prefix token `T_VEHICLE` into the state sequence, enabling cross-attention across all in-play units and candidate action options.
5. **Stage 4 Weight Upcycling**:
   - Stage 4 checkpoints (`curriculum_v1_stage4.pkl`) contain fully converged embedding tables, scalar projections, vortex unit projections, option heads, and value heads.
   - The Stage 4 monolithic FFN parameters ($128 \times 512$ and $512 \times 128$) can be cloned directly to initialize all 4 MoE experts with small symmetry-breaking perturbation ($\epsilon \sim \mathcal{N}(0, 10^{-4})$), ensuring the model begins training from a near-optimal policy baseline.

---

## 3. Caveats

1. **Airgap Sandbox Time Constraints**: In the Kaggle submission sandbox, OS network access is disabled. `datetime.now(timezone.utc)` functions locally from the container system clock, which Kaggle synchronizes with UTC. If Kaggle container clock drifts, the Apex trigger relies on elapsed step counts as fallback.
2. **Apple Silicon Unified Memory Allocation**: MLX allocates unified memory dynamically. When running 4 experts with gradient accumulation, memory footprint increases proportionally to active expert graphs. Explicit `mx.eval()` calls after gradient accumulation steps prevent lazy graph accumulation.
3. **Pre-Step 0 Vehicle Draft Availability**: The vehicle draft module requires the full 60-card list (`self_deck`). In online Kaggle matches, the 60-card deck list is known from agent configuration before turn 0.

---

## 4. Conclusion & Architectural Specifications

### 4.1 Specification: Unified Architecture Contracts

#### A. PyTorch Policy (`rl/policy_moe_torch.py`)
```python
class TokenTransformerMoETorch(TokenTransformerTorchInference):
    """Unified PyTorch Policy integrating 4D RoPEND, MoE Router, and Vehicle Draft."""
    
    def __init__(
        self,
        card_table: CardTable,
        cfg: dict[str, Any],
        static_card_features: np.ndarray | None = None,
    ) -> None:
        super().__init__(card_table, cfg, static_card_features)
        self.ropend_enabled = bool(cfg.get("ropend", True))
        self.num_experts = int(cfg.get("num_experts", 4))
        self.top_k = int(cfg.get("top_k", 2))
        
        # Replace standard TransformerEncoder with RoPEND + MoE stack
        self.encoder = TransformerEncoderMoETorch(
            d_model=self.d,
            nhead=int(cfg["nhead"]),
            nlayers=int(cfg["nlayers"]),
            ff_dim=4 * self.d,
            num_experts=self.num_experts,
            top_k=self.top_k,
        )
        
        # Vehicle Draft Module
        self.vehicle_draft_encoder = VehicleDraftEncoderTorch(
            d_model=self.d,
            nhead=int(cfg["nhead"]),
        )
        
        # RoPEND Coordinate Projections / Scalers
        self.c1_scale = 1.0 / 200.0   # Step [0..200]
        self.c2_scale = 1.0 / 30.0    # Day [0..30]
        self.c3_scale = 1.0 / 600.0   # Clock [0..600s]
        self.c4_scale = 1.0 / 2000.0  # Elo [0..2000]

    def forward_with_moe(
        self,
        o: dict[str, torch.Tensor],
        deck_60: torch.Tensor | None = None,
        apex_mode: bool = False,
        memory_in: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Strict FP32 forward pass returning (logits, value, memory_out, aux_loss, aux_preds)."""
        ...
```

#### B. MLX Policy (`rl/policy_moe_mlx.py`)
```python
class TokenTransformerMoEMLX(TokenTransformerMLX):
    """Apple Silicon native MLX Policy with 4D RoPEND, MoE Router, and Vehicle Draft."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        nlayers: int = 4,
        num_experts: int = 4,
        top_k: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(vocab_size, d_model=d_model, nhead=nhead, nlayers=nlayers, **kwargs)
        self.encoder = TransformerEncoderMoEMLX(
            d_model=d_model,
            nhead=nhead,
            ff_dim=4 * d_model,
            nlayers=nlayers,
            num_experts=num_experts,
            top_k=top_k,
        )
        self.vehicle_draft_encoder = VehicleDraftEncoderMLX(d_model=d_model, nhead=nhead)

    def logits_value_aux_moe(
        self,
        o: dict[str, mx.array],
        deck_60: mx.array | None = None,
        apex_mode: bool = False,
        opt_len: int | None = None,
        memory_in: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, dict[str, mx.array]]:
        """Fused forward pass: logits, value, memory_out, moe_aux_loss, aux_dict."""
        ...
```

### 4.2 Strict FP32 Precision Contract Specification

| Contract Dimension | Specification | Validation Mechanism |
| :--- | :--- | :--- |
| **Model Parameter Dtype** | `torch.float32` / `mx.float32` strictly | Asserted during checkpoint load and layer initialization |
| **Static Card Features** | Fixed shape `[vocab_size+1, 32]`, `float32` | SHA256 bytes digest validation against checkpoint metadata |
| **Card CSV Integrity** | CSV file SHA256 checksum match | `_sha256_file(card_table.csv_path)` vs `contract["card_csv_sha256"]` |
| **Masking Constants** | Strict finite `-65504.0` | Eliminates overflow on half-precision conversion |
| **Optimizer States** | `FP32StateMuon` ($v$) & `FP32StateAdamW` ($m, v$) in `mx.float32` | Explicit cast in `init_single` and `apply_single` |
| **Gradient Accumulation** | Unnormalized FP32 cross-entropy sums | Multiplied and normalized by batch token count at update |

### 4.3 Optimizer Split Routing Contract (`_build_moe_optimizer`)

```python
_MOE_ROUTER_PREFIXES = ("encoder.layers.", ".router.")
_STRUCTURED_PREFIXES = ("type_query.", "type_bias.")
_ADAMW_PREFIXES = (
    "card_emb.", "type_emb.", "sel_type_emb.", "sel_ctx_emb.",
    "opt_verb_emb.", "attack_emb.", "opt_head.", "submit_head.",
    "value_head.", "meta_bucket_emb.", "agent_bucket_emb.",
    "deck_bucket_emb.", "day_proj.", "meta_ctx_base", "cls",
    "value_tok", "submit_tok", "learned_init", "vehicle_draft_encoder.pos_emb"
)

def _use_muon_moe(path: str, parameter: mx.array) -> bool:
    """Route 2D hidden projection, attention, and expert weights to Muon."""
    if parameter.ndim != 2 or not path.endswith(".weight"):
        return False
    if any(prefix in path for prefix in _MOE_ROUTER_PREFIXES if "router.gate" in path):
        return False
    return not path.startswith(_ADAMW_PREFIXES) and not path.startswith(_STRUCTURED_PREFIXES)

def _use_structured_adamw(path: str, parameter: mx.array) -> bool:
    return path.startswith(_STRUCTURED_PREFIXES)

def _build_moe_optimizer(cfg) -> PathSafeMultiOptimizer:
    muon = FP32StateMuon(learning_rate=cfg.lr, momentum=cfg.muon_momentum, weight_decay=cfg.muon_weight_decay)
    structured_adamw = FP32StateAdamW(learning_rate=cfg.lr, betas=cfg.adamw_betas, eps=cfg.adamw_eps, weight_decay=cfg.structured_weight_decay)
    adamw = FP32StateAdamW(learning_rate=cfg.lr, betas=cfg.adamw_betas, eps=cfg.adamw_eps, weight_decay=cfg.adamw_weight_decay)
    return PathSafeMultiOptimizer([muon, structured_adamw, adamw], filters=[_use_muon_moe, _use_structured_adamw])
```

### 4.4 Backward Compatibility & Stage 4 Migration Strategy

```python
def migrate_stage4_to_moe(
    stage4_checkpoint_path: str | Path,
    moe_model: TokenTransformerMoEMLX,
    perturbation_scale: float = 1e-4,
) -> TokenTransformerMoEMLX:
    """Surgically upcycle Stage 4 weights into the MoE RoPEND architecture."""
    with open(stage4_checkpoint_path, "rb") as fh:
        state = pickle.load(fh)
    stage4_params = mlx_nn.utils.tree_flatten(state["model"])
    stage4_dict = dict(stage4_params)
    
    moe_updates = []
    for k, v in mlx_nn.utils.tree_flatten(moe_model.parameters()):
        if k in stage4_dict:
            # 1-to-1 transfer of trunk, embeddings, attention and heads
            moe_updates.append((k, stage4_dict[k]))
        elif ".experts." in k:
            # Map Stage 4 monolithic FFN (ff.layers.0 and ff.layers.2) to each expert
            layer_idx = k.split(".")[2]
            if "ff1.weight" in k or "layers.0.weight" in k:
                s4_key = f"encoder.layers.{layer_idx}.ff.layers.0.weight"
                base_w = stage4_dict[s4_key]
                noise = mx.random.normal(base_w.shape) * perturbation_scale
                moe_updates.append((k, base_w + noise))
            elif "ff1.bias" in k or "layers.0.bias" in k:
                s4_key = f"encoder.layers.{layer_idx}.ff.layers.0.bias"
                moe_updates.append((k, stage4_dict[s4_key]))
            elif "ff2.weight" in k or "layers.2.weight" in k:
                s4_key = f"encoder.layers.{layer_idx}.ff.layers.2.weight"
                base_w = stage4_dict[s4_key]
                noise = mx.random.normal(base_w.shape) * perturbation_scale
                moe_updates.append((k, base_w + noise))
            elif "ff2.bias" in k or "layers.2.bias" in k:
                s4_key = f"encoder.layers.{layer_idx}.ff.layers.2.bias"
                moe_updates.append((k, stage4_dict[s4_key]))
        elif ".router.gate.weight" in k:
            # Initialize router gate weights with uniform small variance
            moe_updates.append((k, mx.random.normal(v.shape) * 0.02))
    
    moe_model.update(mlx_nn.utils.tree_unflatten(moe_updates))
    return moe_model
```

---

## 5. Verification Method

### 5.1 Unit Test Suite Specification (`tests/unit/test_fp32_contract.py`)

Execute via:
```bash
uv run python -m pytest tests/unit/test_fp32_contract.py -v
```

The test file `tests/unit/test_fp32_contract.py` verifies 7 invariants:
1. `test_torch_model_parameters_strict_fp32`: Every parameter and buffer in `TokenTransformerMoETorch` is strictly `torch.float32`.
2. `test_static_card_feature_sha256_verification`: Static card feature table SHA256 checksum and CSV hash matches contract; corrupted hashes raise `ValueError`.
3. `test_zero_fp16_underflow_in_attention_and_masking`: Finite masked logits strictly equal `-65504.0`, preventing numerical underflow and infinite gradients.
4. `test_exact_tensor_shape_assertions`: Forward pass emits exact shapes: logits `[B, 192]`, value `[B]`, memory `[B, 16, 128]`, aux loss scalar, and aux predictions dictionary.
5. `test_stage4_to_moe_migration_round_trip`: Stage 4 weights transfer to MoE model without error, expert weights receive perturbation, and loss evaluates cleanly.
6. `test_apex_mode_temperature_switch`: `apex_mode=True` ($\tau = 0.1$) reduces router entropy by $>70\%$ compared to standard exploration mode ($\tau = 1.0$).
7. `test_torch_mlx_moe_parity`: On identical synthetic input batches, PyTorch and MLX MoE policies produce matching logits within numerical tolerance ($\text{atol} < 10^{-4}$).
