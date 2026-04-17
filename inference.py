import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from model import MiniTransformer
from train import load_checkpoint


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class InferenceConfig:
    name: str           = "run"
    ckpt_dir: str       = "checkpoints"
    ckpt: str           = ""
    
    prompt: str         = "Hallo"
    len_out: int        = 1000
    num_samples: int    = 1
    
    temperature: float  = 1.0  # > 1.0 = random, 1.0 = unchanged, < 1.0 = decisive, 0 = fully deterministic 
    top_k: int          = 0    # 0 = disabled; N = only sample from top-N tokens
    
def parse_args() -> InferenceConfig:
    parser = argparse.ArgumentParser(description="Generate text from a trained MiniTransformer.")
    
    parser.add_argument("--name",           type=str,   default=InferenceConfig.name,       help="The name of the model.")
    parser.add_argument("--ckpt_dir",       type=str,   default=InferenceConfig.ckpt_dir,   help="Path to checkoint directory")
    parser.add_argument("--ckpt",           type=str,   default=InferenceConfig.ckpt,       help="Which checkpoint to load (default=latest.pt)")
    parser.add_argument("--prompt",         type=str,   default=InferenceConfig.prompt,     help="String to start from (e.g., 'Hallo')")
    parser.add_argument("--len_out",        type=int,   default=InferenceConfig.len_out,    help="Length of charackter to output")
    parser.add_argument("--num_samples",    type=int,   default=InferenceConfig.num_samples,help="Number of outputs")
    parser.add_argument("--temperature",    type=float, default=InferenceConfig.temperature,help="The temp to use (1.0 = random, 1.0 = unchanged, < 1.0 = decisive, 0 = fully deterministic)")
    parser.add_argument("--top_k",          type=int,   default=InferenceConfig.top_k,      help="Number of top token to sample based on the temp (0=disabled, n=only sample fro top-n tokens)")
    
    args = parser.parse_args()
    return InferenceConfig(**vars(args))


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model_for_inference(config: InferenceConfig, device: str):
    """Returns (model, stoi, itos, block_size). Model is already in eval mode."""
    if config.ckpt:
        ckpt_path = Path(config.ckpt)
    else:
        ckpt_path = Path(config.ckpt_dir) / config.name / "latest.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt   = torch.load(ckpt_path, map_location=device)
    chars  = ckpt["chars"]
    stoi   = {ch: i for i, ch in enumerate(chars)}
    itos   = {i: ch for i, ch in enumerate(chars)}
    model  = MiniTransformer(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    block_size = ckpt["model_config"]["block_size"]

    print(f"Loaded: {ckpt_path}")
    print(f"  iter {ckpt['iteration']:,} | loss {ckpt['loss']:.4f} | "
          f"vocab {len(chars)} | block_size {block_size}")

    return model, stoi, itos, block_size

# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model:         torch.nn.Module,
    prompt:        str,
    stoi:          dict,
    itos:          dict,
    block_size:    int,
    len_out: int   = 200,
    temperature:   float = 1.0,
    top_k:         int   = 0,
    device:        str   = "cpu",
) -> str:
    for ch in prompt:
        if ch not in stoi:
            raise ValueError(f"Character '{ch}' not in model vocabulary.")

    context = [stoi[ch] for ch in prompt]

    for _ in range(len_out):
        # Trim to block_size, then left-pad if still short
        context_ids = context[-block_size:]
        while len(context_ids) < block_size:
            context_ids = [0] + context_ids

        x           = torch.tensor([context_ids], device=device)  # [1, block_size]
        logits      = model(x)                                     # [1, block_size, vocab_size]
        logits_last = logits[0, -1, :] / temperature              # [vocab_size]

        if top_k > 0:
            kth_value   = torch.topk(logits_last, top_k).values[-1]
            logits_last = logits_last.masked_fill(logits_last < kth_value, float('-inf'))

        probs    = torch.softmax(logits_last, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        context.append(next_idx)

    return "".join(itos[i] for i in context)


# ── Run ───────────────────────────────────────────────────────────────────────

def run(config: InferenceConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stoi, itos, block_size = load_model_for_inference(config, device)

    print(f"\nPrompt: '{config.prompt}'")
    print(f"Settings: temp={config.temperature} | top_k={config.top_k} | "
          f"max_new={config.len_out} | samples={config.num_samples}")
    print("─" * 60)

    for i in range(config.num_samples):
        if config.num_samples > 1:
            print(f"\n── Sample {i + 1} ──")
        text = generate(
            model         = model,
            prompt        = config.prompt,
            stoi          = stoi,
            itos          = itos,
            block_size    = block_size,
            len_out       = config.len_out,
            temperature   = config.temperature,
            top_k         = config.top_k,
            device        = device,
        )
        print(text)

    print("\n" + "─" * 60)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = parse_args()
    run(config)