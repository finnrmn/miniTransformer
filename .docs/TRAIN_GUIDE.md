# Train Guide — CLI, Config & Training Loop

A conceptual walkthrough of [train.py](../train.py). Every knob, every design choice, none of the boilerplate.

---

## Usage

```bash
# Default: char-level transformer on the jokes dataset, 10k iterations
python train.py --path data/jokes_de.txt

# Bigger model, longer run
python train.py --path data/jokes_de.txt \
    --embedding_dim 128 --num_heads 8 --num_blocks 4 \
    --num_iterations 100000 --use_lr_schedule

# Resume an interrupted run
python train.py --name <name> --resume

# Quick smoke test
python train.py --num_iterations 1000 --log_interval 100 --eval_interval 500
```

Run names are auto-generated from dataset + hyperparameters if `--name` is left at its default. The command above with defaults produces `jokes_de_d64h4b2_ctx8_lr1e-3_bs32`.

---

## The Config Dataclass

Every hyperparameter lives in `TrainConfig` — one source of truth, no magic numbers scattered across the code. Fields map 1:1 to CLI flags.

| Field | Default | What it does |
|---|---|---|
| `path` | `data/input.txt` | Training text file |
| `embedding_dim` | 64 | Size of all internal representations |
| `num_heads` | 4 | Attention heads per block (must divide `embedding_dim`) |
| `num_blocks` | 2 | Stacked transformer blocks |
| `block_size` | 8 | Context window — chars the model sees at once |
| `batch_size` | 32 | Sequences per gradient update |
| `num_iterations` | 10,000 | Total training steps |
| `learning_rate` | 1e-3 | Adam initial LR |
| `use_lr_schedule` | `False` | Cosine annealing LR decay |
| `grad_clip` | 1.0 | Max gradient norm (0 disables) |
| `name` | auto | Run name → checkpoint subdirectory |
| `log_interval` | 500 | Print train loss every N iters |
| `eval_interval` | 1000 | Compute val loss every N iters |
| `ckpt_interval` | 5000 | Save numbered checkpoint every N iters |
| `resume` | `False` | Resume from `latest.pt` |

**Design choices worth knowing:**

- `grad_clip=1.0` is on by default — costs nothing, prevents NaN loss on unstable runs.
- `use_lr_schedule=False` by default — safe for short runs; enable for 100k+ iterations.
- `__post_init__` generates a descriptive run name from config if `name` is still `"run"`.

---

## What One Training Iteration Looks Like

```python
xb, yb = get_batch(train_data, block_size, batch_size, device)

logits = model(xb)                                         # [B, T, vocab]
loss   = loss_fn(logits.view(-1, vocab_size), yb.view(-1))

optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
optimizer.step()
if scheduler: scheduler.step()
```

That's the entire loop. The rest is periodic logging, evaluation, and checkpointing.

**Why reshape logits?** `CrossEntropyLoss` expects `[N, C]` inputs and `[N]` targets. The model outputs `[B, T, C]`, so we flatten the batch and time dimensions together: `[B*T, C]` and `[B*T]`.

**Why `zero_grad()` before `backward()`?** PyTorch accumulates gradients — skipping the zero step mixes stale gradients from the previous iteration into the current one.

---

## Data Pipeline

Five small, reusable utilities — all pure functions, importable by `inference.py` and `evaluation.py`:

- `load_text(path)` — read UTF-8 text from disk.
- `build_vocab(text)` — sort unique chars, build `stoi` and `itos`.
- `encode(text, stoi)` → `LongTensor` of token IDs.
- `split_data(data, 0.8, 0.1)` → 80/10/10 train/val/test tensors.
- `get_batch(data, block_size, batch_size, device)` → random `(X, Y)` pair, already on device.

`get_batch` samples `batch_size` random start positions, then grabs sequences of length `block_size` for `X` and the same sequences shifted by one for `Y`. Each sequence contains `block_size` training examples thanks to teacher forcing.

---

## Checkpoints

Each checkpoint is a self-contained dict with everything needed to resume:

```python
{
    "iteration": int,
    "loss": float,
    "chars": list,                   # rebuild vocab
    "model_config": {...},           # rebuild architecture
    "train_config": dict,            # full TrainConfig (asdict)
    "model_state_dict": ...,         # learned weights
    "optimizer_state_dict": ...,     # Adam moments
}
```

Two files per save:

- `ckpt_iter_0005000.pt` — numbered, kept forever for analysis.
- `latest.pt` — rolling, always the most recent. Resume reads this.

**Why save the optimizer state?** Adam's running estimates of gradient mean (`m`) and variance (`v`) are part of the training state. Resuming with a fresh optimizer means the first few hundred steps behave differently. With the optimizer state restored, resumption is seamless.

**Why save `chars`?** To rebuild `stoi`/`itos` at inference time. Without it, a checkpoint cannot encode or decode text.

**Why save `train_config`?** Self-documenting artifacts. Any checkpoint tells you exactly how it was trained.

---

## Evaluation During Training

`evaluate_loss` runs every `eval_interval` iterations. It samples `eval_batches` random val batches and averages the loss.

```python
@torch.no_grad()
def evaluate_loss(model, val_data, config, device):
    model.eval()
    ...
    model.train()
    return total / config.eval_batches
```

**`@torch.no_grad()`** disables autograd — no backward graph is built, memory drops ~50%. Always use this for eval.

**`model.eval()` / `model.train()`** toggles dropout behavior (even if dropout is not yet used, this is the right habit — swap in dropout later and eval behavior is already correct).

Perplexity is reported alongside loss: `ppl = exp(val_loss)`. Lower is better; a random init gives `ppl ≈ vocab_size`.

---

## Optional: Learning Rate Schedule

Enable with `--use_lr_schedule`. Uses `CosineAnnealingLR` from `T_max = num_iterations - start_iteration`. LR starts at `learning_rate` and decays smoothly to near-zero:

```
LR(t) = 0.5 · LR_max · (1 + cos(π · t / T_max))
```

Standard for long LM runs. For 10k-iter demos, a constant LR is usually fine.

---

## Expected Output

```
Device: cpu | Run: jokes_de_d64h4b2_ctx8_lr1e-3_bs32
Dataset: 2,847,193 characters from 'data/jokes_de.txt'
Vocab size: 116
Split → train: 2,277,754  val: 284,719  test: 284,720
Parameters: 97,396

Training: iter 0 → 10000

iter       0 | loss 4.7531 | lr 1.00e-03 | 0.0s
  [eval] val_loss 4.7489 | ppl 115.7
iter     500 | loss 3.1204 | lr 1.00e-03 | 4.2s
  [eval] val_loss 3.0891 | ppl 21.9
...
  [ckpt] saved → ckpt_iter_0005000.pt
...

Training complete.
```

**Sanity checks:**
- `loss ≈ 4.75` at iter 0 — correct, since `log(116) ≈ 4.75`.
- `ppl ≈ vocab_size` initially — the model is equally unsure about all characters.
- After training: `loss < 2.5`, `ppl < 15` on val means the model learned real structure.

---

## What Other Scripts Reuse

Both [inference.py](../inference.py) and [evaluation.py](../evaluation.py) import from `train.py`:

```python
from train import load_text, build_vocab, encode, split_data, get_batch, load_checkpoint
```

Writing these as top-level functions (not inside `train()`) is what makes them reusable — no duplication, no drift between train and eval preprocessing.
