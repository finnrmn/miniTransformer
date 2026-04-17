# Evaluation Guide — Metrics & Diagnostics

A conceptual walkthrough of [evaluation.py](../evaluation.py). Three modes: evaluate one run, compare runs, or sweep every checkpoint in a run.

---

## Usage

```bash
# Full evaluation of one run: loss on all splits, perplexity, overfitting check
python evaluation.py --name <name>

# Add qualitative text samples
python evaluation.py --name <name> --samples --num_samples 3

# Evaluate a specific checkpoint rather than latest.pt
python evaluation.py --name <name> \
    --ckpt checkpoints/<name>/ckpt_iter_0050000.pt

# Rank multiple runs by test perplexity
python evaluation.py --compare run_a run_b run_c

# Sweep every checkpoint in a run → shows the learning curve
python evaluation.py --name <name> --sweep
```

---

## Three Views of Model Quality

| View | Metric | What it tells you |
|---|---|---|
| Quantitative | train/val/test loss + perplexity | How good overall? Is it overfitting? |
| Qualitative | generated text samples | Does it produce coherent language? |
| Comparative | perplexity across runs | Which model size / config wins? |

The **test split** is the honest number — the model never saw it during training *or* tuning. Val loss guides training decisions; test loss is reported once, at the end. Tuning hyperparameters on test loss contaminates the test set.

---

## Core Metric: `compute_split_loss`

Averages cross-entropy over `eval_batches` random batches from one split and returns `(loss, perplexity)`:

```python
@torch.no_grad()
def compute_split_loss(model, data, block_size, batch_size, vocab_size, eval_batches, device):
    ...
    return avg_loss, math.exp(avg_loss)
```

**Why `eval_batches=50` by default?** During training, `eval_batches=10` keeps evaluations cheap. Standalone evaluation has time to spare — 50 batches gives a noticeably more stable estimate.

**Perplexity.** `ppl = exp(loss)`. Interpretable as "effective number of equally likely next-token choices." A random init over 116 characters gives `ppl ≈ 116`. A well-trained small model on this dataset reaches `ppl` in the low teens.

---

## Mode 1: Single Run Evaluation

`evaluate_run` loads a checkpoint, rebuilds the data splits, and prints loss/ppl for train, val, and test. Optionally generates samples.

**Overfitting check.** After computing losses, it prints the `val - train` gap with a diagnosis:

| Gap | Diagnosis |
|---|---|
| `< 0.1` | Healthy — not memorizing |
| `0.1 – 0.3` | Normal — val is always slightly higher |
| `> 0.3` | Watch out — possible overfitting |

A negative gap (`val < train`) is a red flag — usually means `eval_batches` is too low and the estimate is noisy.

**Sample output:**

```
────────────────────────────────────────────────────────────
Run: jokes_de_d64h4b2_ctx8_lr1e-3_bs32  |  iter 10,000
────────────────────────────────────────────────────────────
  train   loss 2.7841  |  ppl 16.18
  val     loss 2.8903  |  ppl 17.96
  test    loss 2.8812  |  ppl 17.80

  val - train gap: +0.1062  ✓
```

---

## Mode 2: Compare Runs

`compare_runs` evaluates `latest.pt` of every run in `--compare` and prints a ranked table sorted by **test perplexity**:

```
────────────────────────────────────────────────────────────
  Run                  Train PPL    Val PPL   Test PPL
────────────────────────────────────────────────────────────
  jokes_big                 9.41      11.23      11.18  ← best
  jokes_small              16.18      17.96      17.80
  jokes_tiny               28.44      29.11      29.03
────────────────────────────────────────────────────────────
```

**Why sort by test, not val?** Val perplexity may have influenced training decisions (early stopping, LR schedule), so it is mildly optimistic. Test perplexity is the honest ranking.

---

## Mode 3: Checkpoint Sweep

`checkpoint_sweep` globs every `ckpt_iter_*.pt` in a run directory and evaluates them in order. This gives you the **learning curve as a table** and flags the iteration where val loss starts rising — the overfitting onset.

```
══════════════════════════════════════════════════════════════════════
  Sweep: jokes_small  (10 checkpoints)
══════════════════════════════════════════════════════════════════════
      Iter       Train         Val        Test     Val PPL
──────────────────────────────────────────────────────────────────────
     1,000      3.4201      3.5100      3.5041     33.45
    10,000      2.7841      2.8903      2.8812     17.96
    50,000      2.3201      2.5811      2.5799     13.21
   100,000      2.1034      2.4501      2.4489     11.59
   200,000      1.9812      2.4201      2.4189     11.24
   500,000      1.7401      2.4801      2.4799     11.93  ↑
 1,000,000      1.5901      2.5411      2.5399     12.68  ↑
══════════════════════════════════════════════════════════════════════
  ↑ = val loss rose since previous checkpoint
```

Here the model overfits after ~200k iterations — train loss keeps dropping while val loss ticks back up. The `↑` markers tell you exactly which checkpoint to use for downstream inference (the one just *before* the first `↑`).

**Without a sweep you would not know this.** Relying on `latest.pt` after a long run often means using an overfit model.

---

## What to Do When Overfitting Shows Up

1. Use the checkpoint right before val loss turned — that's your best model.
2. For future runs on the same data: add dropout, reduce `num_blocks` or `embedding_dim`, or simply stop training earlier.
3. If the dataset is small and you keep hitting this wall: get more data. Regularization only buys you so much.

---

## Why Three Splits Matter

```
80% train    → weights updated on this
10% val      → read during training to detect overfitting
10% test     → read exactly once, at the very end
```

A model that looks great on val but bad on test means you overfit to val (via tuning choices that kept working until they didn't). A model that matches on all three is actually generalizing.

The test split is only meaningful if it stays sealed. Every peek before the final evaluation erodes its honesty.

---

## Import Graph

`evaluation.py` is the downstream consumer — it imports from both upstream scripts:

```python
from train     import load_text, build_vocab, encode, split_data, get_batch
from inference import load_model_for_inference, generate
```

This is the payoff of writing `train.py` and `inference.py` as libraries of pure functions instead of monoliths: evaluation adds a new orchestration layer without duplicating any preprocessing or generation code.
