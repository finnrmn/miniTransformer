# Inference Guide — Sampling & Generation

A conceptual walkthrough of [inference.py](../inference.py). Loads any checkpoint and generates text with controllable sampling.

---

## Usage

```bash
# Default: "Hallo" prompt, 1000 new chars, standard sampling
python inference.py --name <name>

# Different prompt, creative output
python inference.py --name <name> --prompt "Der Arzt" \
    --temperature 1.4 --num_samples 5 --len_out 300

# Conservative output (more coherent, more repetitive)
python inference.py --name <name> --prompt "Ein Mann" --temperature 0.7

# Top-k sampling: only ever pick from the 10 most likely tokens
python inference.py --name <name> --top_k 10

# A specific checkpoint instead of latest.pt
python inference.py --name <name> \
    --ckpt checkpoints/<name>/ckpt_iter_0050000.pt --prompt "Warum"
```

---

## The Autoregressive Loop

Generation proceeds one character at a time. The model always sees at most `block_size` tokens — the context is a sliding window over the growing output.

```
context = encode(prompt)            # e.g. "Hallo" → [12, 25, 67]

for _ in range(len_out):
    ids     = context[-block_size:]     # trim to window
    ids     = left-pad with 0 if short  # model expects exactly block_size
    logits  = model(ids)[:, -1, :]      # last-position logits only
    logits /= temperature
    logits  = top_k_filter(logits, k)   # optional
    probs   = softmax(logits)
    next_id = multinomial(probs)
    context.append(next_id)

return decode(context)
```

Three things to understand:

- **Only the last position matters.** The model outputs logits for every position in the window, but positions `[0..T-2]` predict already-known tokens. We care about the prediction for position `T`, which lives at `logits[0, -1, :]`.
- **Left-padding with token 0** is safe because the causal mask prevents real tokens from attending to padding positions on their left.
- **`@torch.no_grad()`** decorates `generate()` — no backward graph, ~50% less memory, faster forward pass.

---

## Temperature

The single most useful inference knob. Divide logits by temperature *before* softmax:

```python
probs = torch.softmax(logits / temperature, dim=-1)
```

| Temp | Effect |
|---|---|
| `→ 0` | Equivalent to argmax — fully deterministic, often repetitive |
| `< 1.0` | Sharpens distribution — more conservative, coherent, predictable |
| `1.0` | Unchanged — standard sampling |
| `> 1.0` | Flattens distribution — more creative, more random |
| `≫ 1.0` | Near-uniform — often nonsense |

Intuition: temperature rescales the gap between the most and least likely tokens. Small temperature amplifies the top choice; large temperature evens everything out.

---

## Top-k Sampling

A safety valve for high temperature. After scaling by temperature, zero out everything except the `k` most likely tokens:

```python
kth_value   = torch.topk(logits, top_k).values[-1]
logits      = logits.masked_fill(logits < kth_value, float("-inf"))
```

- `top_k=0` — disabled, full distribution.
- `top_k=1` — argmax (combined with low temperature = deterministic).
- `top_k=10–50` — keeps creativity but prevents sampling very unlikely tokens.

**Rule of thumb:** pair high temperature with moderate `top_k`. That gives you diverse output without letting the model ever pick garbage.

---

## Temperature + Top-k, Visualized

```
raw logits:   [-1.2,  3.4,  0.8,  2.1, -0.5, ...]
              ÷ temperature (0.7)
sharpened:    [-1.71, 4.86, 1.14, 3.0, -0.71, ...]
              top_k=3: keep top 3, rest → -inf
filtered:     [-inf,  4.86, -inf, 3.0, -inf, ...]
              softmax
probs:        [ 0.00, 0.87, 0.00, 0.13, 0.00, ...]
              multinomial sample
              → token index 1 (very likely, but not guaranteed)
```

---

## Loading a Checkpoint

`load_model_for_inference` returns `(model, stoi, itos, block_size)` — everything needed to run `generate()`. The checkpoint dict carries its own `model_config` and `chars`, so the model architecture and vocabulary are reconstructed automatically:

```python
ckpt  = torch.load(ckpt_path, map_location=device)
model = MiniTransformer(**ckpt["model_config"]).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
stoi  = {ch: i for i, ch in enumerate(ckpt["chars"])}
itos  = {i: ch for i, ch in enumerate(ckpt["chars"])}
```

`model.eval()` toggles any dropout/BN layers into inference mode. Harmless without them; essential when they are added later.

---

## Sampling Cheat Sheet

| Goal | Flags |
|---|---|
| Deterministic baseline | `--temperature 0.1 --top_k 1` |
| Standard (balanced) | `--temperature 1.0` (default) |
| More variety | `--temperature 1.2 --top_k 20` |
| Max creativity | `--temperature 1.5` |
| Safe creativity | `--temperature 1.2 --top_k 10` |

---

## Expected Output

```
Loaded: checkpoints/jokes_de_d64h4b2_ctx8_lr1e-3_bs32/latest.pt
  iter 10,000 | loss 2.87 | vocab 116 | block_size 8

Prompt: 'Der'
Settings: temp=1.0 | top_k=0 | len_out=200 | samples=3
────────────────────────────────────────────────────────────
── Sample 1 ──
Der Mann fragt seine Frau: "Warum hast du das Fenster aufgemacht?"
"Weil es so heiß ist!"

── Sample 2 ──
Der Arzt sagt zum Patienten: "Sie müssen aufhören zu rauchen."
...
```

Quality scales with training time and model size. Ten thousand iterations on the small model produces fragments; a million iterations on the medium model produces recognizable joke structure.

---

## What evaluation.py Reuses

```python
from inference import load_model_for_inference, generate
```

`generate()` is pure (no globals, no I/O) — [evaluation.py](../evaluation.py) calls it directly to produce qualitative samples alongside its quantitative metrics.
