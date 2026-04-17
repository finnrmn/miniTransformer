# miniTransformer — A Minimal Transformer from Scratch

A character-level Transformer language model built from scratch in PyTorch — around 650 lines of code split across four files, with a complete training, inference, and evaluation pipeline.

`PyTorch` · `Python 3.10+` · `Character-level` · `~97k–2M parameters`

---

## What This Is

This repository is an educational, production-shaped implementation of a decoder-only Transformer — the same architecture family as GPT. It is inspired by Andrej Karpathy's nanoGPT, but separates concerns into distinct modules (architecture, training, inference, evaluation) rather than collapsing them into a single script.

The model is trained on a German-jokes dataset scraped with the included web scraper (2.8M characters, vocab of 116 unique characters). Three pretrained runs at different scales are shipped in [checkpoints/](checkpoints/) so you can skip training and jump straight to inference.

The goal is understanding, not state-of-the-art performance: every component — attention scaling, causal masking, positional encoding, pre-norm residuals — is written out explicitly and documented in [.docs/](.docs/).

---

## Quick Start

```bash
# 1. Train a small model (~97k params, ~10 min on CPU)
python train.py --path data/jokes_de.txt --num_iterations 100000

# 2. Generate text from the trained checkpoint
python inference.py --prompt "Hallo" --len_out 200 --temperature 1.0

# 3. Evaluate train/val/test loss, perplexity, and qualitative samples
python evaluation.py --samples
```

Run names are auto-generated from dataset and hyperparameters — the command above produces `jokes_de_d64h4b2_ctx8_lr1e-3_bs32`, matching one of the shipped checkpoints.

---

## Architecture

```
token IDs ─▶ Token Embedding ─▶ + Positional Encoding ─▶ N × Transformer Block ─▶ LayerNorm ─▶ Linear ─▶ logits
                                                                │
                                                    ┌───────────┴───────────┐
                                                    │  LN ▶ Multi-Head Attn │  (+ residual)
                                                    │  LN ▶ Feed-Forward    │  (+ residual)
                                                    └───────────────────────┘
```

Six components in [model.py](model.py):

- `MiniTransformer` — top-level container wiring everything together
- `TransformerBlock` — pre-norm layout with residual connections around attention and FFN
- `PositionalEncoding` — sinusoidal, stored as a non-trainable buffer
- `SingleHeadSelfAttention` — Q/K/V projections, causal mask, scaled dot-product
- `MultiHeadSelfAttention` — parallel heads concatenated and projected back
- `FeedForward` — two-layer MLP (expand ×4, ReLU, compress)

**Default config** (matches the smallest shipped checkpoint):
`embedding_dim=64`, `num_heads=4`, `num_blocks=2`, `block_size=8` → ~97k parameters.

Three pretrained scales are available in [checkpoints/](checkpoints/):

| Scale | Config | Params |
|---|---|---|
| Small | `d64h4b2` | ~97k |
| Medium | `d128h8b4` | ~800k |
| Large | `d256h8b6` | ~2M |

→ Full architectural derivation with tensor-shape traces: [.docs/TRANSFORMER_GUIDE.md](.docs/TRANSFORMER_GUIDE.md)

---

## Project Layout

```
nanoGPT/
├── model.py              # Pure architecture — no training/inference logic
├── train.py              # CLI training with checkpointing, resume, LR scheduling
├── inference.py          # Text generation with temperature & top-k sampling
├── evaluation.py         # Loss/perplexity, run comparison, checkpoint sweep
├── data/
│   ├── jokes_de.txt           # 2.8M chars German jokes dataset
│   └── scrape_jokes_de.py     # Web scraper that built the dataset
├── checkpoints/               # Three pretrained runs (small/medium/large)
└── .docs/
    ├── TRANSFORMER_GUIDE.md   # Architecture deep dive
    ├── TRAIN_GUIDE.md         # Training pipeline internals
    ├── INFERENCE_GUIDE.md     # Sampling strategies
    └── EVALUATION_GUIDE.md    # Metrics & diagnostics
```

### Files & Key Exports

| File | Key classes / functions |
|---|---|
| [model.py](model.py) | `MiniTransformer`, `TransformerBlock`, `PositionalEncoding`, `MultiHeadSelfAttention`, `FeedForward` |
| [train.py](train.py) | `TrainConfig`, `train()`, `get_batch()`, `save_checkpoint()`, `load_checkpoint()`, `evaluate_loss()` |
| [inference.py](inference.py) | `InferenceConfig`, `load_model_for_inference()`, `generate()` |
| [evaluation.py](evaluation.py) | `EvalConfig`, `evaluate_run()`, `compare_runs()`, `checkpoint_sweep()`, `compute_split_loss()` |

---

## Training

Each iteration: sample a random batch → forward pass → cross-entropy loss → backward → gradient clip → optimizer step → optional cosine LR schedule. Validation loss and perplexity are computed every `--eval_interval` steps; numbered checkpoints are written every `--ckpt_interval` steps alongside a rolling `latest.pt`.

A checkpoint is a self-contained dictionary holding the model state, optimizer state, full `TrainConfig`, vocabulary (`chars`), iteration, and loss — so resuming a run needs nothing beyond the file itself:

```bash
python train.py --resume --name small
```

→ Full CLI reference and expected output: [.docs/TRAIN_GUIDE.md](.docs/TRAIN_GUIDE.md)

---

## Inference — Sampling Cheat Sheet

| Goal | Flags |
|---|---|
| Deterministic | `--temperature 0.1 --top_k 1` |
| Standard | `--temperature 1.0` |
| Variety | `--temperature 1.2 --top_k 20` |
| Max creativity | `--temperature 1.5` |

Generation is autoregressive: the context is trimmed to `block_size`, logits for the last position are scaled by temperature, optionally filtered to the top-k tokens, passed through softmax, and sampled from via `torch.multinomial`.

→ Sampling internals and theory: [.docs/INFERENCE_GUIDE.md](.docs/INFERENCE_GUIDE.md)

---

## Evaluation

Three modes, one script:

- **Single run** — `python evaluation.py --name <run> --samples`
  Train/val/test loss and perplexity, plus qualitative samples and an overfitting check based on the val-train gap.

- **Compare runs** — `python evaluation.py --compare run_a run_b run_c`
  Side-by-side table of runs, sorted by test perplexity.

- **Checkpoint sweep** — `python evaluation.py --sweep --name <run>`
  Evaluates every `ckpt_iter_*.pt` file in a run directory and prints a learning curve, flagging the iteration where validation loss starts rising.

→ Metrics, diagnostics, and overfitting heuristics: [.docs/EVALUATION_GUIDE.md](.docs/EVALUATION_GUIDE.md)

---

## What I Took Away From Building This

- **Pre-norm beats post-norm** for deep stacks because the residual path stays clean and gradients flow without being squeezed through a LayerNorm at every step.
- **The `1/√d_k` scale in attention** is not cosmetic: without it, dot-product variance grows with `d_k`, softmax saturates, and gradients collapse.
- **Causal masks belong in `register_buffer`**, not `nn.Parameter` — they're state that needs to move with `.to(device)` but must never be updated by the optimizer.
- **Perplexity is just `exp(loss)`**, but framing it as "effective vocabulary size the model is choosing from" makes loss numbers visceral in a way raw cross-entropy doesn't.
- **Overfitting shows up in the val-train gap long before samples go bad** — a checkpoint sweep makes this visible and tells you exactly which iteration to roll back to.

---

## Requirements

```
torch >= 2.0
```

The scraper additionally requires `requests` and `beautifulsoup4`.

---

## Credits & References

- Inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT).
- Architecture: *Attention Is All You Need*, Vaswani et al., 2017.
- Positional encoding follows the original sinusoidal formulation from the same paper.
