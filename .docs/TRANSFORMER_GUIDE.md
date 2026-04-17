# Transformer Guide — Concepts & Intuition

A compact walkthrough of every component in [model.py](../model.py). The goal is understanding, not transcription — for the full code, open the source. For the paper, see [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

---

## 1. Architecture Overview

```
Input text ─▶ Tokenize ─▶ Token Embedding ─▶ + Positional Encoding
                                                      ↓
                               ┌─ N × Transformer Block ─┐
                               │   LN ▶ Multi-Head Attn  │  (+ residual)
                               │   LN ▶ Feed-Forward     │  (+ residual)
                               └─────────────────────────┘
                                                      ↓
                                       Final LayerNorm ─▶ Linear ─▶ logits
```

**Tensor shape trace** (defaults: `batch=32, block_size=8, embedding_dim=64, vocab_size=116`):

| Stage | Shape |
|---|---|
| Token IDs | `[32, 8]` |
| After embedding & pos enc | `[32, 8, 64]` |
| After each block | `[32, 8, 64]` |
| Logits | `[32, 8, 116]` |

The embedding dimension stays constant through every block — each block is a *same-shape transformation*. Only the final projection changes the last dimension to `vocab_size`.

---

## 2. Tokenization & Batch Sampling

**Character-level tokenization** assigns each unique character an integer ID. The German-jokes corpus has 116 unique chars (letters, digits, punctuation, umlauts) — that is our `vocab_size`.

```python
chars = sorted(set(text))
stoi  = {ch: i for i, ch in enumerate(chars)}
itos  = {i: ch for i, ch in enumerate(chars)}
```

**Why characters?** Simplest possible tokenizer — no external library. Trade-off: char-level models need more steps to learn word structure. Production models (GPT-4, Claude) use byte-pair encoding (BPE) at the subword level (~50k–100k vocabulary).

**Three splits** (80/10/10): train updates weights, val detects overfitting during training, test is touched once at the very end. Tuning hyperparameters on test loss contaminates the number.

**The sliding-window trick:** each sampled sequence of length `block_size` contains `block_size` training examples simultaneously — at position `t`, the model predicts token `t+1` given tokens `0..t`.

```
Input:  [D, e, r,  , A, r, z, t]
Target: [e, r,  , A, r, z, t,  ]
```

This is **teacher forcing**: we always feed ground-truth previous tokens during training, even if the model would have predicted differently. It keeps training stable and parallel.

---

## 3. Token Embeddings

`nn.Embedding(vocab_size, embedding_dim)` is a learnable lookup table — each integer ID maps to a dense 64-dim vector. Initially random, the embeddings are updated by backprop so that related characters end up near each other in the 64-dim space.

Shape: `[32, 8]` → `[32, 8, 64]`.

GPT-3 uses `embedding_dim=12288`. The principle is identical — just bigger.

---

## 4. Positional Encoding

**The problem:** self-attention treats its input as a *set*, not a *sequence*. Without positional information, "Der Hund beißt den Mann" and "Den Mann beißt der Hund" would look identical to the model.

**The solution** (from the original paper): add a sinusoidal signal to each token embedding based on its position.

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

Each dimension pair oscillates at a different frequency — low dimensions fast, high dimensions slow. Together they form a unique fingerprint for every position, and the model can learn to recover relative positions via linear combinations.

**`register_buffer` vs `nn.Parameter`:** sinusoidal PE is stored as a buffer — saved in `state_dict` and moved to GPU via `.to(device)`, but *never updated by the optimizer*. The values encode pure positional geometry; there is nothing to learn.

**Sinusoidal vs learned PE:** GPT-2/3 use learned positional embeddings (another `nn.Embedding` table). Both work well. Sinusoidal generalizes to sequences longer than training; learned is more flexible but cannot extrapolate.

---

## 5. Self-Attention: Q, K, V

This is the heart of the Transformer. Everything else is scaffolding.

**Metaphor:** every token independently produces a **Query** ("what am I looking for?"), a **Key** ("what do I advertise?"), and a **Value** ("what do I share if someone attends to me?"). The attention weight between token `i` and token `j` is how well `i`'s query matches `j`'s key.

**The formula:**

```
Attention(Q, K, V) = softmax(Q K^T / √d_k) V
```

Step by step:

1. **Project** the input into Q, K, V via learned linear layers (`bias=False`).
2. **Score:** `scores = Q @ K^T` — a `[T, T]` matrix of pairwise compatibilities.
3. **Scale** by `1/√d_k`. Without this, scores grow with `d_k`, softmax saturates, and gradients vanish. Dividing normalizes variance back to ~1.
4. **Causal mask:** positions in the upper triangle of the score matrix get set to `-inf` — token `i` must never see token `j > i` during training.
5. **Softmax** along the key dimension → rows sum to 1.
6. **Mix values:** `out = weights @ V`.

The causal mask stored as a buffer:

```python
self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
# After softmax, exp(-inf) = 0 — future positions get exactly zero weight.
```

---

## 6. Multi-Head Attention

A single head can only model one type of relationship. Multi-head attention runs `num_heads` heads in parallel, each on a `head_size = embedding_dim / num_heads` subspace. Different heads specialize — some track grammatical agreement, some long-range dependencies, some local patterns. None of this is hand-designed; specialization emerges from training.

```
embedding_dim=64, num_heads=4 → each head operates on head_size=16

[B, T, 16] × 4 heads ─▶ concat ─▶ [B, T, 64] ─▶ output projection ─▶ [B, T, 64]
```

The final **output projection** is a learned linear layer that recombines head outputs rather than just concatenating them blindly.

**`nn.ModuleList` matters:** plain Python lists hide sub-modules from PyTorch's parameter tracking, so their weights never get optimized or saved. Always use `ModuleList` for dynamic sub-module collections.

---

## 7. Feed-Forward Network

Attention *routes* information between tokens — but the value aggregation itself is essentially linear. The FFN adds the non-linear processing power that transforms representations.

```
FFN(x) = Linear(ReLU(Linear(x)))
         d_model → 4·d_model → d_model
```

**Why 4× expansion?** Empirically found in the original paper. Intuition: give each token a "working memory" 4× its embedding size for computation, then compress. The FFN contains about 2/3 of all parameters per layer — it stores factual knowledge while attention handles routing.

Processing is **position-wise**: the same MLP runs independently on each of the `block_size` token representations.

Modern architectures (LLaMA, PaLM) replace ReLU with SwiGLU for better performance, but the 4× principle stays.

---

## 8. LayerNorm & Residuals

These two mechanisms make training deep Transformers *possible*. Without them, gradients either vanish or explode before reaching early layers.

### LayerNorm

Normalizes each token *across its own feature dimension* — statistics are computed per-token, independent of batch size or sequence length:

```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
```

`γ` and `β` are learned scale/shift; `μ`, `σ²` are the mean and variance of the 64 features of one token. After normalization every token has mean ≈ 0, variance ≈ 1.

**Pre-Norm vs Post-Norm:** the original paper placed LayerNorm *after* each sub-layer. Modern practice (and this implementation) uses **pre-norm** — normalize *before* the sub-layer, then add the residual:

```
x = x + SubLayer(LayerNorm(x))
```

Pre-norm is now standard because training is more stable (no warmup needed), gradients flow more directly to early layers, and deep models train without careful initialization tricks.

### Residual Connections

The gradient through `output = f(x) + x` has derivative `∂f/∂x + 1`. That `+1` is a **gradient highway** from the loss all the way back to early layers, regardless of how small `∂f/∂x` becomes. This is what makes 96-layer Transformers trainable at all.

---

## 9. The Transformer Block

The repeating unit. Two sub-layers, each wrapped in pre-norm and a residual:

```python
x = x + self.attn(self.ln1(x))   # multi-head self-attention
x = x + self.ffn(self.ln2(x))    # position-wise feed-forward
```

Every block is **shape-preserving** — in and out are both `[batch, block_size, embedding_dim]`. Stacking blocks is trivially composable.

**Why stack?** Each block does one round of information mixing (attention) followed by one round of computation (FFN). Deeper stacks build progressively more abstract representations — character → word → phrase → discourse. GPT-3 uses 96 blocks; this model uses 2.

---

## 10. Full Model & Loss

The top-level assembly:

```
token IDs ─▶ embed ─▶ + pos enc ─▶ N blocks ─▶ final LayerNorm ─▶ Linear ─▶ logits
```

The final LayerNorm stabilizes the scale before the output projection. The output `Linear(embedding_dim, vocab_size)` produces a logit per vocabulary entry at every position — *not* a softmax; softmax happens inside the loss.

**Parameter count** (defaults):
- Token embedding: 116 × 64 ≈ 7.4k
- Per block (attention + FFN): ~41k → 2 blocks ≈ 82k
- Final LN + output projection: ~7.5k
- **Total: ~97k parameters.**

**Cross-entropy loss.** At every position, the task is: predict the next token. Cross-entropy measures how wrong the predicted distribution is:

```
loss = -log P(correct_token | previous_tokens)
```

Randomly initialized over 116 classes, loss starts at `log(116) ≈ 4.75`. After training, losses below 2.0 indicate the model has learned real structure. Reshape logits and targets before feeding them to `CrossEntropyLoss`:

```python
loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))
```

---

## 11. Training Step

```
sample batch → forward → cross-entropy → zero_grad → backward → clip → step
```

- **`zero_grad()` before `backward()`** — PyTorch accumulates gradients by default.
- **`backward()`** runs autograd from the loss backward through every op.
- **`step()`** updates parameters. With Adam: per-parameter adaptive learning rates via first/second moment estimates — robust to varied gradient magnitudes across layers.
- **Gradient clipping** (`clip_grad_norm_`) caps the gradient norm to prevent NaN loss on unstable runs.

A checkpoint bundles everything needed to resume: model state, optimizer state (Adam's moments), vocab, train config, iteration, and loss. Resuming without optimizer state restarts with "cold" moments, and the first few hundred iterations behave differently.

---

## 12. Text Generation

Autoregressive, one token at a time:

```
context = encode(prompt)
for _ in range(max_new_chars):
    ids     = last block_size tokens of context (left-pad if shorter)
    logits  = model(ids)[:, -1, :] / temperature
    probs   = softmax(logits)
    next_id = multinomial(probs)
    context.append(next_id)
```

Key choices:

- **`@torch.no_grad()`** disables autograd for inference (~50% less memory).
- **`multinomial` vs `argmax`:** sampling produces natural variation; argmax is deterministic and often repetitive.
- **Only the last position's logits matter** — the earlier positions predict already-known tokens.
- **Temperature** scales logits before softmax: `< 1` sharpens (more conservative), `> 1` flattens (more creative), `→ 0` is argmax.
- **Top-k** zeros out all but the k most likely tokens — a safety valve that prevents nonsense at high temperature.

---

## Extensions Worth Knowing

- **Longer training** — 10k iters is a demo; real improvement kicks in around 100k+.
- **Bigger model** — doubling `embedding_dim` roughly quadruples parameters (because of the 4× FFN). Scaling `num_blocks` adds linearly.
- **Dropout** — zeros out a random fraction of activations during training; remember `model.eval()` for inference.
- **Cosine LR schedule** — high LR at start, decays smoothly to near-zero. Standard for long LM runs.
- **Longer context** — attention is O(n²) in sequence length, so doubling `block_size` quadruples attention cost.
- **KV cache** — during generation, cache past K and V tensors instead of recomputing them every step. Turns per-token cost from O(T) to O(1) amortized. Standard in vLLM, Hugging Face, TensorRT-LLM.

---

## Monitoring & Diagnostics

**Perplexity.** `ppl = exp(loss)`. Interpretable as "effective number of equally likely next-token choices." Random init over 116 chars gives ppl ≈ 116; a decent model should reach ppl < 20.

**Loss curves.** Train loss below val loss is normal — the model has seen the train data. Val loss *rising* while train loss keeps falling means overfitting.

**Qualitative sampling.** Numbers hide a lot. Early (~1k iters): random chars. Mid (~50k): recognizable words. Late (~500k): coherent fragments, joke-like structure.

**Warning signs:**

| Pattern | Likely cause | Fix |
|---|---|---|
| Val rises while train falls | Overfitting | More data, add dropout, smaller model |
| Loss oscillates wildly | LR too high | Reduce by 10× |
| Loss barely moves | LR too low, or bug | Increase LR; check data pipeline |
| Loss = NaN | Gradient explosion | Enable `clip_grad_norm_` |

---

## Takeaways

- **Decoder-only Transformer** = token embeddings + positional encoding + stacked self-attention blocks + LM head. Same shape as GPT-2, just smaller.
- **Self-attention** routes information; **FFN** transforms it; **residuals + pre-norm** keep deep training stable.
- **Causal masking** prevents future-leak during training; **teacher forcing** makes training parallel.
- Claude, GPT-4, LLaMA, PaLM — all share this core. Scale, data quality, and alignment training are what separate a toy model from a frontier one.
