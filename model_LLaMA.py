import torch
import math

# ===== LLaMA Transformer Architecture ====

class LlamaTransformer(torch.nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, num_heads, num_blocks, block_size):
        super().__init__()
        
        self.token_emb = torch.nn.Embedding(vocab_size, embedding_dim)
        # Transformer-Block
        # RMSNorm
        # Linear Layer
        
        
# class RotaryPositionEncoding(torch.nn.Module):
    