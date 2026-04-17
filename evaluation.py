import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from train import load_text, build_vocab, encode, split_data, get_batch
from inference import load_model_for_inference, generate


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    name:           str   = "run"
    ckpt_dir:       str   = "checkpoints"
    ckpt:           str   = ""

    data_path:      str   = "data/jokes_de.txt"
    eval_batches:   int   = 50

    samples:        bool  = False
    num_samples:    int   = 3
    prompt:         str   = "Hallo"
    len_out:        int   = 500
    temperature:    float = 1.0
    top_k:          int   = 0

    compare:        list  = None
    sweep:          bool  = False


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate a trained MiniTransformer.")

    parser.add_argument("--name",           type=str,   default=EvalConfig.name)
    parser.add_argument("--ckpt_dir",       type=str,   default=EvalConfig.ckpt_dir)
    parser.add_argument("--ckpt",           type=str,   default=EvalConfig.ckpt)
    parser.add_argument("--data_path",      type=str,   default=EvalConfig.data_path)
    parser.add_argument("--eval_batches",   type=int,   default=EvalConfig.eval_batches)
    parser.add_argument("--samples",        action="store_true")
    parser.add_argument("--num_samples",    type=int,   default=EvalConfig.num_samples)
    parser.add_argument("--prompt",         type=str,   default=EvalConfig.prompt)
    parser.add_argument("--len_out",        type=int,   default=EvalConfig.len_out)
    parser.add_argument("--temperature",    type=float, default=EvalConfig.temperature)
    parser.add_argument("--top_k",          type=int,   default=EvalConfig.top_k)
    parser.add_argument("--compare",        type=str,   nargs="+")
    parser.add_argument("--sweep",          action="store_true")

    args = parser.parse_args()
    return EvalConfig(**vars(args))


# ── Core Metric ───────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_split_loss(model, data, block_size, batch_size, vocab_size, eval_batches, device):
    """Returns (avg_loss, perplexity) over eval_batches random batches."""
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total   = 0.0
    for _ in range(eval_batches):
        xb, yb = get_batch(data, block_size, batch_size, device)
        logits  = model(xb)
        total  += loss_fn(logits.view(-1, vocab_size), yb.view(-1)).item()
    avg_loss = total / eval_batches
    return avg_loss, math.exp(avg_loss)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_inference_config(config: EvalConfig, ckpt_path: Path):
    from inference import InferenceConfig
    return InferenceConfig(
        name        = config.name,
        ckpt_dir    = config.ckpt_dir,
        ckpt        = str(ckpt_path),
    )


def _resolve_ckpt_path(config: EvalConfig) -> Path:
    if config.ckpt:
        return Path(config.ckpt)
    return Path(config.ckpt_dir) / config.name / "latest.pt"


# ── Single Run Evaluation ─────────────────────────────────────────────────────

def evaluate_run(config: EvalConfig, ckpt_path: Path = None) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if ckpt_path is None:
        ckpt_path = _resolve_ckpt_path(config)

    model, stoi, itos, block_size = load_model_for_inference(
        _make_inference_config(config, ckpt_path), device
    )
    ckpt       = torch.load(ckpt_path, map_location=device)
    vocab_size = ckpt["model_config"]["vocab_size"]
    iteration  = ckpt["iteration"]

    # Data
    text                            = load_text(config.data_path)
    _, stoi_data, _                 = build_vocab(text)
    data                            = encode(text, stoi_data)
    train_data, val_data, test_data = split_data(data)

    print(f"\n{'─'*60}")
    print(f"Run: {config.name}  |  iter {iteration:,}")
    print(f"{'─'*60}")

    results = {"name": config.name, "iteration": iteration}

    for split_name, split_data_ in [
        ("train", train_data), ("val", val_data), ("test", test_data)
    ]:
        loss, ppl = compute_split_loss(
            model        = model,
            data         = split_data_,
            block_size   = block_size,
            batch_size   = 32,
            vocab_size   = vocab_size,
            eval_batches = config.eval_batches,
            device       = device,
        )
        print(f"  {split_name:<6}  loss {loss:.4f}  |  ppl {ppl:.2f}")
        results[f"{split_name}_loss"] = loss
        results[f"{split_name}_ppl"]  = ppl

    # Overfitting signal
    gap = results["val_loss"] - results["train_loss"]
    print(f"\n  val - train gap: {gap:+.4f}", end="  ")
    if gap > 0.3:
        print("⚠  possible overfitting")
    elif gap < 0:
        print("(val < train — check eval_batches)")
    else:
        print("✓")

    # Qualitative samples
    if config.samples:
        print(f"\n{'─'*60}")
        print(f"Samples  prompt='{config.prompt}' | temp={config.temperature} | top_k={config.top_k}")
        print(f"{'─'*60}")
        for i in range(config.num_samples):
            print(f"\n── Sample {i+1} ──")
            print(generate(
                model         = model,
                prompt        = config.prompt,
                stoi          = stoi,
                itos          = itos,
                block_size    = block_size,
                len_out       = config.len_out,
                temperature   = config.temperature,
                top_k         = config.top_k,
                device        = device,
            ))

    return results


# ── Multi-Run Comparison ──────────────────────────────────────────────────────

def compare_runs(config: EvalConfig):
    print(f"\n{'═'*60}")
    print(f"  Comparison: {config.compare}")
    print(f"{'═'*60}")

    all_results = []
    for name in config.compare:
        cfg = EvalConfig(
            name           = name,
            ckpt_dir       = config.ckpt_dir,
            data_path      = config.data_path,
            eval_batches   = config.eval_batches,
        )
        all_results.append(evaluate_run(cfg))

    all_results.sort(key=lambda r: r["test_ppl"])

    print(f"\n{'─'*60}")
    print(f"  {'Run':<20} {'Train PPL':>10} {'Val PPL':>10} {'Test PPL':>10}")
    print(f"{'─'*60}")
    for i, r in enumerate(all_results):
        marker = "  ← best" if i == 0 else ""
        print(
            f"  {r['name']:<20}"
            f"  {r['train_ppl']:>8.2f}"
            f"  {r['val_ppl']:>8.2f}"
            f"  {r['test_ppl']:>8.2f}"
            f"{marker}"
        )
    print(f"{'─'*60}")


# ── Checkpoint Sweep ──────────────────────────────────────────────────────────

def checkpoint_sweep(config: EvalConfig):
    run_dir    = Path(config.ckpt_dir) / config.name
    ckpt_paths = sorted(
        run_dir.glob("ckpt_iter_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1])
    )

    if not ckpt_paths:
        print(f"No checkpoints found in {run_dir}")
        return

    print(f"\n{'═'*70}")
    print(f"  Sweep: {config.name}  ({len(ckpt_paths)} checkpoints)")
    print(f"{'═'*70}")
    print(f"  {'Iter':>8}  {'Train':>10}  {'Val':>10}  {'Test':>10}  {'Val PPL':>8}")
    print(f"{'─'*70}")

    prev_val = None
    for ckpt_path in ckpt_paths:
        r     = evaluate_run(config, ckpt_path=ckpt_path)
        trend = " ↑" if prev_val is not None and r["val_loss"] > prev_val + 0.01 else ""
        prev_val = r["val_loss"]
        print(
            f"  {r['iteration']:>8,}"
            f"  {r['train_loss']:>10.4f}"
            f"  {r['val_loss']:>10.4f}"
            f"  {r['test_loss']:>10.4f}"
            f"  {r['val_ppl']:>8.2f}"
            f"{trend}"
        )

    print(f"{'═'*70}")
    print("  ↑ = val loss rose since previous checkpoint")


# ── Dispatch ──────────────────────────────────────────────────────────────────

def run(config: EvalConfig):
    if config.compare:
        compare_runs(config)
    elif config.sweep:
        checkpoint_sweep(config)
    else:
        evaluate_run(config)


if __name__ == "__main__":
    config = parse_args()
    run(config)