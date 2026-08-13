# RoPEND: N-Dimensional Rotary Positional Embeddings

## 1. Abstract
Standard Rotary Positional Embedding (RoPE) encodes absolute positional information with a rotation matrix and naturally incorporates relative position dependency in self-attention. RoPEND (N-Dimensional RoPE) extends this to $N$ independent dimensions, allowing the Transformer to understand spatial, temporal, and hierarchical metadata simultaneously without vector interference.

## 2. Mathematical Formulation
Given an embedding dimension $D$ and $N$ independent coordinate axes (e.g., Turn, Meta-Epoch, Clock, Elo), we partition the embedding vector $\mathbf{x} \in \mathbb{R}^D$ into $N$ sub-vectors:
$$ D = d_1 + d_2 + \dots + d_N $$

For a query vector $\mathbf{q}$ and key vector $\mathbf{k}$, each sub-vector $\mathbf{q}_i$ is rotated by its specific coordinate $c_i$ in the $i$-th dimension:
$$ \mathbf{q}'_i = R_{\Theta_i, c_i} \mathbf{q}_i $$
$$ \mathbf{k}'_i = R_{\Theta_i, c_i} \mathbf{k}_i $$

The rotation matrix $R_{\Theta_i, c_i}$ applies block-diagonal 2D rotations. For a 2D slice $(x_{2j}, x_{2j+1})$, the transformation is:
$$ \begin{pmatrix} q'_{2j} \\ q'_{2j+1} \end{pmatrix} = \begin{pmatrix} \cos(c_i \theta_j) & -\sin(c_i \theta_j) \\ \sin(c_i \theta_j) & \cos(c_i \theta_j) \end{pmatrix} \begin{pmatrix} q_{2j} \\ q_{2j+1} \end{pmatrix} $$
Where $\theta_j = 10000^{-2j/d_i}$.

## 3. Dot Product and Relative Attention
The inner product of the transformed query and key inherently computes the relative distance across all $N$ dimensions independently:
$$ \langle \mathbf{q}', \mathbf{k}' \rangle = \sum_{i=1}^N \mathbf{q}_i^\top R_{\Theta_i, c_i^{q}}^\top R_{\Theta_i, c_i^{k}} \mathbf{k}_i = \sum_{i=1}^N \mathbf{q}_i^\top R_{\Theta_i, c_i^{k} - c_i^{q}} \mathbf{k}_i $$

## 4. Implementation in Kaggle Meta
We partition $D=128$ into 4 axes of $d=32$:
1. **$c_1$ (Turn Step):** The discrete step of the match ($0, 1, 2 \dots$).
2. **$c_2$ (Meta-Epoch):** The calendar day offset from a fixed origin (e.g., `Date - StartDate`).
3. **$c_3$ (Time Remaining):** Normalized countdown from 600s.
4. **$c_4$ (Elo/Identity):** Estimated continuous Elo standing.

By applying RoPEND, the attention matrix mathematically scales based on the exact temporal and hierarchical distance between any two states.
