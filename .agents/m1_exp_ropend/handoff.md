# 4D RoPEND Operator Architecture & Technical Blueprint

## 1. Observation
- **Scope Contract**: The Milestone 1 architecture (`.agents/sub_orch_m1/SCOPE.md`, `docs/architecture/01_ropend_theory.md`) defines a 4D Rotary Positional Embedding (RoPEND) operator across 4 independent coordinate axes:
  - $c_1$: Match Turn / Step ($0 \dots 200$)
  - $c_2$: Meta-Epoch / Day Offset ($0 \dots 31$)
  - $c_3$: Urgency Clock / Time Remaining ($0 \dots 600$ seconds)
  - $c_4$: Inferred Elo / Opponent Standing ($300 \dots 2000$)
- **Subspace Geometry**:
  - Embedding dimension $D_{\text{model}} = 128$, Number of attention heads $H = 4$, Head dimension $D_{\text{head}} = 32$.
  - Each 32-dimensional attention head is partitioned into 4 orthogonal coordinate subspaces of dimension $d = 8$:
    - Axis 1 ($c_1$): Head channels $0 \dots 7$
    - Axis 2 ($c_2$): Head channels $8 \dots 15$
    - Axis 3 ($c_3$): Head channels $16 \dots 23$
    - Axis 4 ($c_4$): Head channels $24 \dots 31$
  - Each 8-dimensional subspace contains $P = 4$ Givens 2D rotation planes with geometric inverse frequency base:
    $$\theta_j = 10000^{-2j / 8} = 10000^{-j / 4} \in \{1.0, 0.1, 0.01, 0.001\}, \quad j \in \{0, 1, 2, 3\}$$
- **Codebase Baseline**:
  - PyTorch implementation (`rl/policy.py`, `rl/policy_infer_torch.py`) and MLX implementation (`rl/policy_mlx.py`) currently lack explicit rotary positional encoding in their self-attention layers.
  - The PyTorch inference contract (`rl/policy_infer_torch.py`) strictly enforces FP32 precision (`torch.float32`), while MLX operates natively on Apple Silicon Unified Memory.
  - Test suites (`scripts/validate/`) execute via `uv run` and use deterministic assertion runners.

---

## 2. Logic Chain

### 2.1 Mathematical Formulation & Givens Rotations
Let $\mathbf{x} \in \mathbb{R}^{B \times L \times H \times 32}$ be the query or key tensor, where $B$ is batch size, $L$ is sequence length, $H=4$ is the number of attention heads, and $D=32$ is head dimension.

The embedding vector $\mathbf{x}_{b,l,h} \in \mathbb{R}^{32}$ is decomposed into 4 disjoint 8-dimensional sub-vectors:
$$\mathbf{x}_{b,l,h} = \begin{pmatrix} \mathbf{x}^{(1)} \\ \mathbf{x}^{(2)} \\ \mathbf{x}^{(3)} \\ \mathbf{x}^{(4)} \end{pmatrix}, \quad \mathbf{x}^{(k)} \in \mathbb{R}^8$$

For each axis $k \in \{1, 2, 3, 4\}$, the 8 dimensions are paired into 4 Givens 2D rotation planes $(x_{2j}^{(k)}, x_{2j+1}^{(k)})$ for $j \in \{0, 1, 2, 3\}$.
The rotation angle for plane $j$ under coordinate value $c_k \in \mathbb{R}$ is:
$$\phi_{k, j} = c_k \cdot \theta_j = c_k \cdot 10000^{-j / 4}$$

The Givens 2D rotation transforms each pair as:
$$\begin{pmatrix} {x'}_{2j}^{(k)} \\ {x'}_{2j+1}^{(k)} \end{pmatrix} = \begin{pmatrix} \cos(\phi_{k,j}) & -\sin(\phi_{k,j}) \\ \sin(\phi_{k,j}) & \cos(\phi_{k,j}) \end{pmatrix} \begin{pmatrix} x_{2j}^{(k)} \\ x_{2j+1}^{(k)} \end{pmatrix}$$

Expanding in closed vector form:
$${x'}_{2j}^{(k)} = x_{2j}^{(k)} \cos(\phi_{k,j}) - x_{2j+1}^{(k)} \sin(\phi_{k,j})$$
$${x'}_{2j+1}^{(k)} = x_{2j}^{(k)} \sin(\phi_{k,j}) + x_{2j+1}^{(k)} \cos(\phi_{k,j})$$

### 2.2 Algebraic Properties & Proof of Invariance
1. **Orthogonality & Norm Preservation**:
   Because each Givens block is an element of the Special Orthogonal group $\mathrm{SO}(2)$, the full transformation matrix $R(\mathbf{c}) = \operatorname{diag}(R_{\Theta_1, c_1}, R_{\Theta_2, c_2}, R_{\Theta_3, c_3}, R_{\Theta_4, c_4})$ is in $\mathrm{SO}(32)$.
   $$R(\mathbf{c})^\top R(\mathbf{c}) = I_{32}$$
   $$\| R(\mathbf{c}) \mathbf{x} \|_2 = \| \mathbf{x} \|_2$$

2. **Relative Positional Dot-Product Invariance**:
   For query $\mathbf{q}$ at coordinate $\mathbf{c}^q = (c_1^q, c_2^q, c_3^q, c_4^q)$ and key $\mathbf{k}$ at coordinate $\mathbf{c}^k = (c_1^k, c_2^k, c_3^k, c_4^k)$:
   $$\langle R(\mathbf{c}^q) \mathbf{q}, R(\mathbf{c}^k) \mathbf{k} \rangle = \mathbf{q}^\top R(\mathbf{c}^q)^\top R(\mathbf{c}^k) \mathbf{k} = \mathbf{q}^\top R(\mathbf{c}^k - \mathbf{c}^q) \mathbf{k}$$
   $$\langle \mathbf{q}', \mathbf{k}' \rangle = \sum_{k=1}^4 \sum_{j=0}^3 \left( (q_{2j}^{(k)} k_{2j}^{(k)} + q_{2j+1}^{(k)} k_{2j+1}^{(k)}) \cos((c_k^k - c_k^q)\theta_j) + (q_{2j}^{(k)} k_{2j+1}^{(k)} - q_{2j+1}^{(k)} k_{2j}^{(k)}) \sin((c_k^k - c_k^q)\theta_j) \right)$$
   The inner product depends strictly on coordinate deltas $\Delta c_k = c_k^k - c_k^q$ along each axis independently.

3. **Subspace Isolation (Zero Cross-Axis Interference)**:
   Because the rotation blocks are strictly diagonal by axis subspaces $[0..7], [8..15], [16..23], [24..31]$, a translation along $c_1$ produces zero variation on the representation in subspaces $k \in \{2, 3, 4\}$.

### 2.3 Vectorized Tensor Transformations (PyTorch vs MLX)
To achieve maximum computational efficiency without explicit loops:
1. **Angle Expansion**:
   Given coordinates $c_1, c_2, c_3, c_4 \in \mathbb{R}^{B \times L}$:
   - Compute $\boldsymbol{\phi}_k = c_k \otimes \boldsymbol{\theta} \in \mathbb{R}^{B \times L \times 4}$.
   - Interleave/repeat each angle twice to align with Givens pairs: $\boldsymbol{\Phi}_k \in \mathbb{R}^{B \times L \times 8}$.
   - Concatenate all 4 axes: $\boldsymbol{\Phi} = [\boldsymbol{\Phi}_1, \boldsymbol{\Phi}_2, \boldsymbol{\Phi}_3, \boldsymbol{\Phi}_4] \in \mathbb{R}^{B \times L \times 32}$.
   - Broadcast to attention heads: $\boldsymbol{\Phi} \in \mathbb{R}^{B \times L \times 1 \times 32}$.
2. **Pairwise Negation**:
   Let $\mathbf{x}_{\text{even}} = \mathbf{x}[..., 0::2] \in \mathbb{R}^{B \times L \times H \times 16}$ and $\mathbf{x}_{\text{odd}} = \mathbf{x}[..., 1::2] \in \mathbb{R}^{B \times L \times H \times 16}$.
   The orthogonal conjugate vector is constructed by:
   $$\mathbf{x}_{\text{rot}} = \operatorname{interleave}(-\mathbf{x}_{\text{odd}}, \mathbf{x}_{\text{even}})$$
3. **Rotated State**:
   $$\mathbf{x}' = \mathbf{x} \odot \cos(\boldsymbol{\Phi}) + \mathbf{x}_{\text{rot}} \odot \sin(\boldsymbol{\Phi})$$

**PyTorch Vector Transformation**:
```python
def apply_ropend_4d(
    x: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    c3: torch.Tensor,
    c4: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    # x: (B, L, H, 32) float32
    # c_i: (B, L) float32, theta: (4,) float32
    coords = [c1, c2, c3, c4]
    axis_angles = [c.unsqueeze(-1) * theta for c in coords]  # [(B, L, 4)]
    axis_angles_rep = [a.repeat_interleave(2, dim=-1) for a in axis_angles]  # [(B, L, 8)]
    phi = torch.cat(axis_angles_rep, dim=-1).unsqueeze(2)  # (B, L, 1, 32)
    
    cos = torch.cos(phi)
    sin = torch.sin(phi)
    
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack([-x_odd, x_even], dim=-1).flatten(-2)
    
    return (x * cos + x_rot * sin).to(torch.float32)
```

**MLX Vector Transformation**:
```python
def apply_ropend_4d_mlx(
    x: mx.array,
    c1: mx.array,
    c2: mx.array,
    c3: mx.array,
    c4: mx.array,
    theta: mx.array,
) -> mx.array:
    # x: (B, L, H, 32) float32
    # c_i: (B, L) float32, theta: (4,) float32
    B, L, H, D = x.shape
    coords = [c1, c2, c3, c4]
    axis_angles_rep = []
    for c in coords:
        a = mx.expand_dims(c, -1) * theta  # (B, L, 4)
        a_rep = mx.reshape(mx.repeat(a, 2, axis=-1), (B, L, 8))
        axis_angles_rep.append(a_rep)
    
    phi = mx.expand_dims(mx.concatenate(axis_angles_rep, axis=-1), 2)  # (B, L, 1, 32)
    cos = mx.cos(phi)
    sin = mx.sin(phi)
    
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    stacked = mx.stack([-x_odd, x_even], axis=-1)
    x_rot = mx.reshape(stacked, (B, L, H, D))
    
    return (x * cos + x_rot * sin).astype(mx.float32)
```

### 2.4 Caching Strategies & Frequency Precomputation
- **Precomputed Base Inverse Frequencies**:
  $\boldsymbol{\theta} = [1.0, 0.1, 0.01, 0.001]$ is constant across all episodes and steps.
  - In PyTorch: Registered as a persistent buffer or module attribute:
    `self.register_buffer("theta", precompute_freqs_4d_torch(dim=32, num_axes=4), persistent=False)`
  - In MLX: Precomputed at `__init__` and cached as `self.theta = precompute_freqs_4d_mlx(dim=32, num_axes=4)`.
- **Dynamic Coordinate Angle Evaluation**:
  Because $c_2$ (Meta-Epoch), $c_3$ (Urgency Clock), and $c_4$ (Inferred Elo) are continuous physical scalars that change dynamically across turns and match matchmaking contexts, dynamic outer-product evaluation is required.
  Since the operation is purely $O(B \cdot L \cdot D)$ memory-bandwidth bound and requires only 16 multiplications and 32 trigonometric evaluations per token, compute overhead on Apple Silicon / CUDA is $< 0.02\%$ of the attention FLOP budget.

---

## 3. Caveats
1. **Coordinate Scale Alignment**:
   - Discrete steps $c_1 \in [0, 200]$: Natural rotation frequencies range from $\phi \in [0, 200]$ to $[0, 0.2]$.
   - Continuous Elo $c_4 \in [300, 2000]$: Large unscaled rating values could wrap through multiple cycles at $\theta_0 = 1.0$. If subtle Elo differences within $[1100, 1800]$ need smooth continuous representation without high-frequency aliasing, scaling $c_4$ by a normalization constant (e.g. $c_4 = (R - 1000) / 100$ or $c_4 = R / 100$) is recommended and supported.
2. **Head Dimension Assumption**:
   - The specification fixes $D_{\text{model}} = 128$, $H = 4$, $D_{\text{head}} = 32$. If head dimension or axes count changes in future iterations, `precompute_freqs_4d` dynamically scales via `axis_dim = head_dim // num_axes` and `num_planes = axis_dim // 2`.
3. **Query/Key Only Application**:
   - As dictated by rotary embedding theory, RoPEND must be applied to Query ($Q$) and Key ($K$) projections only. Value ($V$) projections must NOT be rotated, preserving value semantic content.

---

## 4. Conclusion & Architectural Recommendations

1. **Exact Module Layout**:
   - Create `rl/ropend/__init__.py`: Export `RoPENDTorch`, `apply_ropend_4d`, `precompute_freqs_4d_torch`, `RoPENDMLX`, `apply_ropend_4d_mlx`, `precompute_freqs_4d_mlx`.
   - Create `rl/ropend/ropend_torch.py`: PyTorch `nn.Module` and functional kernel.
   - Create `rl/ropend/ropend_mlx.py`: MLX `nn.Module` and functional kernel.
2. **Integration Touchpoints**:
   - In `rl/policy_moe_torch.py` and `rl/policy_moe_mlx.py`: Apply `apply_ropend_4d` immediately following $W_q(X)$ and $W_k(X)$ projections before multihead dot-product attention.
3. **Coordinate Extractors**:
   - Match step $c_1$ extracted from token positions / episode turn counter.
   - Meta-Epoch $c_2$ extracted from `(timestamp - anchor_timestamp) / 86400.0`.
   - Urgency Clock $c_3$ extracted from normalized timer `cls_scalars[..., clock_idx]`.
   - Inferred Elo $c_4$ extracted from anchor Elo + opponent estimation head $\hat{R}_{\text{opp}}$.

---

## 5. Verification Method

To verify the RoPEND implementation and mathematical properties, execute:
```bash
uv run python tests/unit/test_ropend_math.py
```

### Test Suite Structure (`tests/unit/test_ropend_math.py`)
The unit test file must implement 8 mathematical verification targets:
1. `test_subspace_partitioning_and_shapes`: Asserts output shape `(B, L, 4, 32)` and broadcasting across `(B, L)`, `(B, 1)`, `(L,)`, scalar.
2. `test_algebraic_orthogonality_and_norm_preservation`: Asserts $| \| \text{RoPEND}(x) \|_2 - \| x \|_2 | < 10^{-5}$.
3. `test_identity_at_zero`: Asserts $\text{RoPEND}(x, \mathbf{0}) = x$.
4. `test_inverse_rotation`: Asserts $\text{RoPEND}(\text{RoPEND}(x, \mathbf{c}), -\mathbf{c}) = x$.
5. `test_relative_positional_shift_invariance`: Asserts $\langle \text{RoPEND}(q, \mathbf{c}^q + \mathbf{\delta}), \text{RoPEND}(k, \mathbf{c}^k + \mathbf{\delta}) \rangle = \langle \text{RoPEND}(q, \mathbf{c}^q), \text{RoPEND}(k, \mathbf{c}^k) \rangle$.
6. `test_4d_coordinate_subspace_isolation`: Asserts shifting $c_i$ alters strictly channels $[8(i-1) .. 8i-1]$ with $0.0$ leakage into other 24 channels.
7. `test_pytorch_mlx_numerical_parity`: Asserts $\max | \text{out}_{\text{torch}} - \text{out}_{\text{mlx}} | < 10^{-6}$ across random batches and coordinate configurations.
8. `test_fp32_contract_enforcement`: Asserts all outputs and intermediate buffers are strictly `float32`.
