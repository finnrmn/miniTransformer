# LLaMA - Model Notes

- **Title** LLaMA: Open and Efficient Foundation Language Models
- **Date** 27. Feb 2023
- **Link** [https://arxiv.org/abs/2302.13971](https://arxiv.org/abs/2302.13971)
  
## Architecture

```
                               ┌─────────────────┐
                               │     Softmax     │
                               └─────────────────┘
                                        ▲
                                        │
                               ┌────────┴────────┐
                               │     Linear      │
                               └─────────────────┘
                                        ▲
                                        │
                               ┌────────┴────────┐
                               │    RMS Norm     │
                               └─────────────────┘
                                        ▲
                                        │
   + - - - - - - - - - - - - - - - - - -│- - - - - - - - - - - - - - - - - - +
   :                                    │                                    :
   :                    ┌─────────────►(+)                                   :
   :                    │               ▲                                    :
   :                    │               │                                    :
   :                    │      ┌────────┴────────┐                           :
   :                    │      │  Feed Forward   │                           :
   :                    │      │     SwiGLU      │                           :
   :                    │      └─────────────────┘                           :
   :                    │               ▲                                    :
   :                    │               │                                    :
   :                    │      ┌────────┴────────┐                           :
   :                    │      │    RMS Norm     │                           :
   :                    │      └─────────────────┘                           :
   :                    │               ▲                                    :
   :                    └───────────────┤                                    :   Nx
   :                                    │                                    :
   :    ┌─────────────────────────────►(+)                                   :
   :    │                               ▲                                    :
   :    │                               │                                    :
   :    │      ┌────────────────────────┴────────────────────────┐           :
   :    │      │ Self-Attention (Grouped Multi-Query Attention)  │           :
   :    │      │                  with KV Cache                  │           :
   :    │      └─────────────────────────────────────────────────┘           :
   :    │           Q (~)               K (~)               V                :   (~) = Rotary
   :    │           ▲                   ▲                   ▲                :         Positional Encodings
   :    │           └───────────────────┼───────────────────┘                :
   :    │                               │                                    :
   :    │                      ┌────────┴────────┐                           :
   :    │                      │    RMS Norm     │                           :
   :    │                      └─────────────────┘                           :
   :    │                               ▲                                    :
   :    └───────────────────────────────┤                                    :
   :                                    │                                    :
   + - - - - - - - - - - - - - - - - - -│- - - - - - - - - - - - - - - - - - +
                                        │
                               ┌────────┴────────┐
                               │   Embeddings    │
                               └─────────────────┘
                                        ▲
                                        │
                                      Input
```

Difference to the miniTransformer, which is based on the original Transformer (decoder-only): 
- **RMSNorm** instead of **LayerNorm**
- **Pre-Normalization** instaed of **Post-Normalization**
- **RoPE** instead of **PE**
- **GQA** instead of **MHA**
- **SwiGLU** instead of **ReLU** (activation function in FFN)
---
### RMSNorm
- **Title** Root Mean Square Layer Normalization
- **Date** 16 Oct. 2019
- **Link** [https://arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467)
  
$$
\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\mathrm{RMS}(x)}, 
\qquad 
\mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}
$$

where $\gamma \in \mathbb{R}^d$ is the learned gain parameter (initialized to 1) and $\epsilon$ is a small constant for numerical stability (not part of the original paper formula, but used in implementations to avoid division by zero; typically $10^{-5}$ to $10^{-8}$).

### Pre-Norm

The position of the normalization in the network was original after each residual sum ("post-norm"), which interrupts the stream once
per sublayer. In later variants its placed before the sublayer ("pre-norm") leaving the stream unbroken and trains more stably at depth. 

### RoPE 
- **Title** RoFormer: Enhanced Transformer with Rotary Position Embedding
- **Date** 20 Apr. 2021
- **Link** [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)

The original Transformer adds a position vector to the token embedding once, at the beginning. Rotary Position Encoding (RoPE) apply position inside the attention operator, to the projected queries and keys of every block. 

The per-head dimension `dim_head` is split into $d_{h}/2$ pairs of cords, and pair $t$ of a query position $i$ is rotated by the angle $i\theta{_t}$ with $\theta_t = 10000^{-2(t-1)/d_h}$. 
The key at position $j$ is rotated by $j\theta{_t}$.

Because both are rotated, the attention score only depends on the distance $j-i$ (paper Eq. 16), with $R_m$ being the rotation for position $m$:

$$
\tilde{q}_i^{\top}\tilde{k}_j = (R_i\,q_i)^{\top}(R_j\,k_j) = q_i^{\top}\,R_{j-i}\,k_j
$$

For the intuition and a worked example see [RoPE.md](RoPE.md).

#### RoPE in code

RoPE has **no learned parameters**. It consists of a table that is precomputed once and a rotation that is applied to $Q$ and $K$ in every attention layer.

| Symbol | Meaning | Code |
|---|---|---|
| $d_h$ | per-head dimension, must be even | `head_size = embedding_dim // num_heads` |
| $T_{max}$ | maximum sequence length | `block_size` |
| $m$ | token position, $0 \dots T-1$ | index on the time axis |
| $t$ | pair index, $0 \dots d_h/2-1$ | index on the last axis |

**Step 1 - precompute the angles** (once, in `__init__`)

- Input: `head_size`, `block_size`
- Output: two buffers `cos`, `sin` of shape `[block_size, head_size/2]`

$$
\theta_t = 10000^{-2t/d_h}, \qquad \Phi_{m,t} = m\cdot\theta_t
$$

Here $t$ is 0-indexed, which gives the same values as the 1-indexed formula above.

**Step 2 - rotate** (in every attention `forward`)

- Input: $x$ = projected queries or keys, shape `[B, T, head_size]`
- Output: rotated $x'$, same shape

Pair $t$ consists of the neighbouring dimensions $(x_{2t},\, x_{2t+1})$ of the token at position $m$:

$$
\begin{aligned}
x'_{2t}   &= x_{2t}\cos\Phi_{m,t} - x_{2t+1}\sin\Phi_{m,t} \\
x'_{2t+1} &= x_{2t}\sin\Phi_{m,t} + x_{2t+1}\cos\Phi_{m,t}
\end{aligned}
$$

This is the 2D rotation matrix per pair (paper Eq. 15), computed elementwise instead of with a sparse $d_h \times d_h$ matrix (paper Eq. 34).

```python
# Step 1: __init__
theta  = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))  # [head_size/2]
m      = torch.arange(block_size).float()                                       # [block_size]
angles = torch.outer(m, theta)                                                  # [block_size, head_size/2]
self.register_buffer('cos', angles.cos())
self.register_buffer('sin', angles.sin())

# Step 2: forward(x)                                  x: [B, T, head_size]  (Q or K)
T = x.shape[-2]
cos, sin = self.cos[:T], self.sin[:T]                 # [T, head_size/2]
x_even, x_odd = x[..., 0::2], x[..., 1::2]            # [B, T, head_size/2] each
out_even = x_even * cos - x_odd * sin
out_odd  = x_even * sin + x_odd * cos
out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)   # interleave -> [B, T, head_size]
```

**Step 3 - where it is called**

```
LlamaTransformer.forward
  x = token_emb(idx)                       # no positional encoding is added here
  └─ TransformerBlock.forward              # N times
       x = x + attn(norm1(x))
       └─ Attention.forward
            Q = query(x); K = key(x); V = value(x)
            Q = rope(Q);  K = rope(K)      # V stays unrotated
            scores = Q @ K^T / sqrt(head_size)
```

- **V is not rotated:** position only decides who attends to whom, not what is passed on.
- **LLaMA reference code:** it computes the same rotation with complex numbers, $(x_{2t} + i\,x_{2t+1}) \cdot e^{i\Phi_{m,t}}$, using `torch.polar` and `torch.view_as_complex`.
- **KV cache:** when generating one token at a time, the slice must start at the absolute position: `cos[start_pos : start_pos + T]`.


### GQA
- **Title** GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- **Date** 22 May 2023
- **Link** [https://arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)

Multi-Head Attention (MHA) gives every head its own query, key and value projection. Grouped-Query Attention (GQA) keeps all $H$ query heads, but divides them into $G$ groups, and each group shares a single key and value head.

| Variant | Query heads | Key/Value heads |
|---|---|---|
| MHA (= GQA-$H$) | $H$ | $H$ |
| GQA-$G$ | $H$ | $G$ |
| MQA (= GQA-1) | $H$ | $1$ |

Fewer key/value heads mean fewer parameters and, more importantly, a smaller KV cache during generation. The quality stays close to MHA. GQA is not part of LLaMA-1, it appears in the [Llama 2](https://arxiv.org/abs/2307.09288) reference code as `n_kv_heads`.

| Symbol | Meaning | Code |
|---|---|---|
| $d$ | model dimension | `embedding_dim` |
| $H$ | number of query heads | `num_heads` |
| $G$ | number of key/value heads, must divide $H$ | `num_kv_heads` |
| $n_{rep}$ | query heads per group, $H/G$ | `n_rep` |
| $d_h$ | per-head dimension, $d/H$ | `head_size` |

**Step 1 - project** (no bias)

$$
Q = xW_q,\quad K = xW_k,\quad V = xW_v
\qquad
W_q \in \mathbb{R}^{d \times H d_h},\quad W_k, W_v \in \mathbb{R}^{d \times G d_h}
$$

**Step 2 - assign and attend**

Query head $h$ (0-indexed) uses the key/value head $g(h) = \lfloor h / n_{rep} \rfloor$. With $\tilde{Q}, \tilde{K}$ being the RoPE-rotated versions and $M$ the causal mask:

$$
A^{(h)} = \mathrm{softmax}\left(\frac{\tilde{Q}^{(h)}\,\tilde{K}^{(g(h))\top}}{\sqrt{d_h}} + M\right),
\qquad
O^{(h)} = A^{(h)}\,V^{(g(h))}
$$

**Step 3 - merge**

$$
\mathrm{GQA}(x) = \mathrm{concat}\left(O^{(0)}, \dots, O^{(H-1)}\right) W_o,
\qquad
W_o \in \mathbb{R}^{H d_h \times d}
$$

```python
# __init__
self.n_rep = num_heads // num_kv_heads
self.query = torch.nn.Linear(embedding_dim, num_heads    * head_size, bias=False)
self.key   = torch.nn.Linear(embedding_dim, num_kv_heads * head_size, bias=False)
self.value = torch.nn.Linear(embedding_dim, num_kv_heads * head_size, bias=False)
self.proj  = torch.nn.Linear(num_heads * head_size, embedding_dim, bias=False)

# forward(x)                                                        x: [B, T, embedding_dim]
Q = self.query(x).view(B, T, num_heads,    head_size).transpose(1, 2)   # [B, H, T, head_size]
K = self.key(x).view(B, T, num_kv_heads, head_size).transpose(1, 2)     # [B, G, T, head_size]
V = self.value(x).view(B, T, num_kv_heads, head_size).transpose(1, 2)   # [B, G, T, head_size]

Q, K = rope(Q), rope(K)                                 # V stays unrotated

K = K.repeat_interleave(self.n_rep, dim=1)              # [B, H, T, head_size]
V = V.repeat_interleave(self.n_rep, dim=1)              # [B, H, T, head_size]

scores = Q @ K.transpose(-2, -1) / math.sqrt(head_size)             # [B, H, T, T]
scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
out = torch.softmax(scores, dim=-1) @ V                             # [B, H, T, head_size]
out = out.transpose(1, 2).reshape(B, T, num_heads * head_size)      # [B, T, embedding_dim]
return self.proj(out)
```

- **One module instead of a list of heads:** the heads live in an extra tensor dimension, because the groups share K and V. The RoPE code above works unchanged, since it only uses the last two dimensions.
- **Called in:** `TransformerBlock.forward` as `x = x + attn(norm1(x))`.

#### KV Cache

The KV cache has no paper of its own, it is an inference technique. The GQA paper uses its size as the main motivation, and the [Llama 2 reference code](https://github.com/meta-llama/llama/blob/main/llama/model.py) contains an implementation (`cache_k`, `cache_v`).

During generation only one token is new per step. Because of the causal mask, the keys and values of all earlier tokens never change. So they are stored once per layer and only the new token is computed.

| | Input per step | Attention cost per step |
|---|---|---|
| without cache | whole sequence `[B, T, d]` | $O(T^2)$ |
| with cache | new token `[B, 1, d]` | $O(T)$ |

For a new token at the absolute position $p$ (`start_pos`):

$$
\begin{aligned}
q_p &= \mathrm{RoPE}(x_p W_q,\ p), \qquad k_p = \mathrm{RoPE}(x_p W_k,\ p), \qquad v_p = x_p W_v \\
K_{0:p} &= [\,K_{cache},\ k_p\,], \qquad V_{0:p} = [\,V_{cache},\ v_p\,] \\
o_p &= \mathrm{softmax}\left(\frac{q_p\,K_{0:p}^{\top}}{\sqrt{d_h}}\right) V_{0:p}
\end{aligned}
$$

The cache is stored with $G$ heads, before `repeat_interleave`. Its size per layer is

$$
2 \cdot B \cdot G \cdot T_{max} \cdot d_h
$$

values, which is $H/G$ times smaller than with MHA.

```python
# __init__   (buffers, not saved with the model)
self.register_buffer('cache_k', torch.zeros(max_batch, num_kv_heads, block_size, head_size), persistent=False)
self.register_buffer('cache_v', torch.zeros(max_batch, num_kv_heads, block_size, head_size), persistent=False)

# forward(x, start_pos)   replaces the RoPE and mask lines of the GQA code above
Q, K = rope(Q, start_pos), rope(K, start_pos)           # rope slices cos[start_pos : start_pos + T]

self.cache_k[:B, :, start_pos : start_pos + T] = K      # write the new tokens
self.cache_v[:B, :, start_pos : start_pos + T] = V
K = self.cache_k[:B, :, : start_pos + T]                # [B, G, start_pos + T, head_size]
V = self.cache_v[:B, :, : start_pos + T]

# ... repeat_interleave and scores as above ...          scores: [B, H, T, start_pos + T]
mask = self.tril[start_pos : start_pos + T, : start_pos + T]
scores = scores.masked_fill(mask == 0, float('-inf'))
```

**Where it is called**

```
generate (inference only, torch.no_grad)
  logits = model(prompt, start_pos=0)             # T tokens, fills the cache
  pos = T
  loop:
    next   = sample(logits[:, -1])
    logits = model(next, start_pos=pos)           # 1 token, reads the cache
    pos   += 1
```

- **Training:** runs without the cache, on the whole sequence with the full mask.
- **Keys are cached after RoPE:** they are rotated by their absolute position once and never again.
- **Limit:** `start_pos + T` must stay within `block_size`. A new prompt starts again at `start_pos=0` and overwrites the cache.


### SwiGLU
- **Title** GLU Variants Improve Transformer
- **Date** 12 Feb. 2020
- **Link** [https://arxiv.org/abs/2002.05202](https://arxiv.org/abs/2002.05202)

The original FFN expands, applies ReLU and compresses again. SwiGLU replaces the activation by a gate: the input is projected twice, one projection goes through Swish and is multiplied elementwise with the other one (paper Eq. 6, without bias):

$$
\mathrm{FFN}_{SwiGLU}(x) = \left(\mathrm{Swish}_1(xW) \otimes xV\right) W_2,
\qquad
\mathrm{Swish}_\beta(x) = x \cdot \sigma(\beta x)
$$

$\mathrm{Swish}_1$ (with $\beta = 1$) is the same as SiLU, in PyTorch `F.silu`. For the activation functions in comparison see [GLU.md](GLU.md).

| Symbol | Shape | Code |
|---|---|---|
| $W$ | $d \times d_{ff}$ | `w1` (goes through Swish) |
| $V$ | $d \times d_{ff}$ | `w3` (linear path) |
| $W_2$ | $d_{ff} \times d$ | `w2` (back to `embedding_dim`) |

**Hidden dimension**

SwiGLU has three matrices instead of two. To keep the number of parameters constant, the paper reduces the hidden dimension by a factor of $\frac{2}{3}$:

$$
d_{ff} = \frac{2}{3} \cdot 4d
\qquad\Rightarrow\qquad
3 \cdot d \cdot \tfrac{8}{3}d = 8d^2 = 2 \cdot d \cdot 4d
$$

The LLaMA code additionally rounds $d_{ff}$ up to the next multiple of `multiple_of`. Example with `embedding_dim = 64` and `multiple_of = 8`: $\frac{2}{3} \cdot 256 = 170 \rightarrow 176$.

```python
# __init__
hidden = int(2 * (4 * embedding_dim) / 3)
hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)   # round up
self.w1 = torch.nn.Linear(embedding_dim, hidden, bias=False)         # W
self.w3 = torch.nn.Linear(embedding_dim, hidden, bias=False)         # V
self.w2 = torch.nn.Linear(hidden, embedding_dim, bias=False)         # W_2

# forward(x)                                       x: [B, T, embedding_dim]
return self.w2(F.silu(self.w1(x)) * self.w3(x))    # [B, T, embedding_dim]
```

- **Called in:** `TransformerBlock.forward` as `x = x + ffn(norm2(x))`.
- **Position-wise:** like the original FFN, it is applied to every token independently.