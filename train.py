import argparse
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

from model import MiniTransformer


def _fmt_lr(lr: float) -> str:
    s = f"{lr:.0e}"
    mantissa, exp = s.split("e")
    return f"{mantissa}e{int(exp)}"


@dataclass
class TrainConfig:
    # ── Data ──────────────────────────────────────────────────────────────
    path: str = "data/input.txt"
    
    # ── Model ──────────────────────────────────────────────────────────────
    embedding_dim: int = 64             # size of all internal representations
    num_heads: int = 4                  # attention heads per block (must divide embedding_dim evenly)
    num_blocks: int = 2                 # number of stacked transformer blocks
    block_size: int = 8                 # context window — how many chars the model sees at once
    
    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int = 32                # sequences per gradient update
    num_iterations: int = 10_000        # total training steps
    learning_rate: float = 1e-3         # Adam initial learning rate
    use_lr_schedule: bool = False       # cosine annealing LR decay (recommended for long runs)
    grad_clip: float = 1.0              # max gradient norm (set 0.0 to disable)
    
    # ── Logging & Checkpoints ──────────────────────────────────────────────────────────────
    name: str = "run"                   # name of this training run (used for checkpoint dir)
    ckpt_dir: str = "checkpoints" # root dir; actual path: checkpoint_dir/name/
    log_interval: int = 500             # print train loss every N iterations
    eval_interval: int = 1000           # compute val loss every N iterations
    eval_batches: int = 10              # how many val batches to average for eval
    ckpt_interval: int = 5000           # save a numbered checkpoint every N iterations
    
    # ── Resume ──────────────────────────────────────────────────────────────
    resume: bool = False                # if True, load latest.pt from checkpoint_dir/name/

    def __post_init__(self):
        if self.name == "run":
            dataset = Path(self.path).stem
            lr_str = _fmt_lr(self.learning_rate)
            self.name = (
                f"{dataset}"
                f"_d{self.embedding_dim}h{self.num_heads}b{self.num_blocks}"
                f"_ctx{self.block_size}"
                f"_lr{lr_str}"
                f"_bs{self.batch_size}"
            )

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the MiniTransformer on a text file.")
    
    # ── Data ──
    parser.add_argument("--path",             type=str, default=TrainConfig.path,     help="Path to the training text file")
    # ── Model ──
    parser.add_argument("--embedding_dim",    type=int, default=TrainConfig.embedding_dim,  help="Transformer hidden size")
    parser.add_argument("--num_heads",        type=int, default=TrainConfig.num_heads,      help="Attention heads per block")
    parser.add_argument("--num_blocks",       type=int, default=TrainConfig.num_blocks,     help="Number of transformer blocks")
    parser.add_argument("--block_size",       type=int, default=TrainConfig.block_size,     help="Context window length")
    # ── Training ──
    parser.add_argument("--batch_size",       type=int, default=TrainConfig.batch_size,     help="Sequences per training step")
    parser.add_argument("--num_iterations",   type=int, default=TrainConfig.num_iterations, help="Total training iterations")
    parser.add_argument("--learning_rate",    type=float, default=TrainConfig.learning_rate,help="Initial Adam learning rate")
    parser.add_argument("--use_lr_schedule",  action="store_true",                          help="Enable cosine LR decay")
    parser.add_argument("--grad_clip",        type=int, default=TrainConfig.grad_clip,      help="Max gradient norm (0 disables)")
    # ── Logging & Checkpoints ──
    parser.add_argument("--name",             type=str, default=TrainConfig.name,           help="Name of the run for ckpts")
    parser.add_argument("--ckpt_dir",         type=str, default=TrainConfig.ckpt_dir,       help="Path to store the ckpts")
    parser.add_argument("--log_interval",     type=int, default=TrainConfig.log_interval,   help="Print logs of loss on N iterations")
    parser.add_argument("--eval_interval",    type=int, default=TrainConfig.eval_interval,  help="Run validation every N iterations")
    parser.add_argument("--eval_batches",     type=int, default=TrainConfig.eval_batches,   help="Validation batches per eval")
    parser.add_argument("--ckpt_interval",    type=int, default=TrainConfig.ckpt_interval,  help="Save checkpoint every N iterations")
    # ── Resume ──
    parser.add_argument("--resume",           action="store_true",                          help="Resume from latest checkpoint")
    
    args = parser.parse_args()
    return TrainConfig(**vars(args))


# ── Data Utilities ────────────────────────────────────────────────────────────

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Dataset: {len(text):,} characters from '{path}'")
    return text

def build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    print(f"Vocab size: {len(chars)}")
    return chars, stoi, itos

def encode(text: str, stoi: dict) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)

def split_data(data: torch.Tensor, train_ratio=0.8, val_ratio=0.1):
    n1    = int(train_ratio * len(data))
    n2    = int(val_ratio   * len(data))
    train = data[:n1]
    val   = data[n1: n1 + n2]
    test  = data[n1 + n2 :]
    print(f"Split → train: {len(train):,} ")
    return train, val, test

def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(low=0, high=len(data) - block_size, size=(batch_size,))
    X  = torch.stack([data[i     : i +     block_size] for i in ix])
    Y  = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return X.to(device), Y.to(device)


# ── Checkpoint Utilities ──────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, config: TrainConfig, chars, iteration, loss):
    run_dir = Path(config.ckpt_dir) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt = {
        "iteration": iteration,
        "loss":      float(loss),
        "chars":     chars,
        "model_config": {
            "vocab_size":    len(chars),
            "embedding_dim": config.embedding_dim,
            "num_heads":     config.num_heads,
            "num_blocks":    config.num_blocks,
            "block_size":    config.block_size
        },
        "train_config":         asdict(config),
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    
    step_path   = run_dir / f"ckpt_iter_{iteration:07d}.pt"
    latest_path = run_dir / "latest.pt"
    
    torch.save(ckpt, step_path)
    torch.save(ckpt, latest_path)
    print(f"  [ckpt] saved → {step_path.name}")
    
def load_checkpoint(run_dir: Path, device: str) -> dict:
    path = run_dir / "latest.pt"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")
    ckpt = torch.load(path, map_location=device)
    print(f"  [ckpt] resumed from iteration {ckpt["iteration"]}, loss {ckpt["loss"]:.4f}")
    return ckpt

# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_loss(model, val_data, config: TrainConfig, device) -> float:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total = 0.0
    for _ in range(config.eval_batches):
        xb, yb, = get_batch(val_data, config.block_size, config.batch_size, device)
        logits = model(xb)
        total += loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1)).item()
    model.train()
    return total / config.eval_batches


# ── Training ──────────────────────────────────────────────────────────────────
def train(config: TrainConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Run: {config.name}")
    
    # Data 
    text = load_text(config.path)
    chars, stoi, itos = build_vocab(text)
    data = encode(text, stoi)
    train_data, val_data, _ = split_data(data)
    
    # Model
    model = MiniTransformer(
        vocab_size    = len(chars),
        embedding_dim = config.embedding_dim,
        num_heads     = config.num_heads,
        num_blocks    = config.num_blocks,
        block_size    = config.block_size
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn   = torch.nn.CrossEntropyLoss()
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Resume
    start_iteration = 0
    if config.resume:
        run_dir = Path(config.ckpt_dir) / config.name
        ckpt    = load_checkpoint(run_dir, device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        chars           = ckpt["chars"]
        stoi            = {ch: i for i, ch in enumerate(chars)}
        itos            = {i: ch for i, ch in enumerate(chars)}
        start_iteration = ckpt["iteration"] + 1
    
    # LR Schedule
    scheduler = None
    if config.use_lr_schedule:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=config.num_iterations - start_iteration)
    
    # Loop
    print(f"\nTraining: iter {start_iteration} → {config.num_iterations}\n")
    t0 = time.time()
    
    for iteration in range(start_iteration, config.num_iterations):
        xb, yb = get_batch(train_data, config.block_size, config.batch_size, device)
        
        logits = model(xb)
        loss   = loss_fn(logits.view(-1, len(chars)), yb.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        if iteration % config.log_interval == 0:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"{iteration:7d}/{config.num_iterations:7d} | loss {loss.item():.4f} | lr {lr_now:.2e} | {elapsed:.1f}s")
            t0 = time.time()
        
        if iteration % config.eval_interval == 0:
            val_loss = evaluate_loss(model, val_data, config, device)
            print(f"  [eval] val_loss {val_loss:.4f} | ppl {math.exp(val_loss):.1f}")
            
        if (iteration + 1) % config.ckpt_interval == 0:
            save_checkpoint(model, optimizer, config, chars, iteration + 1, loss.item())
            
    save_checkpoint(model, optimizer, config, chars, config.num_iterations, loss.item())
    print("\nTraining complete.")

# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = parse_args()
    train(config)

    

     

    
