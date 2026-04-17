import torch
import math


# ===== Transformer Architecture ======

class MiniTransformer(torch.nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, num_heads, num_blocks, block_size):
        super().__init__()
        self.token_emb = torch.nn.Embedding(vocab_size, embedding_dim)
        self.pos_enc   = PositionalEncoding(embedding_dim, block_size)
        self.blocks    = torch.nn.Sequential(
            *[TransformerBlock(num_heads, embedding_dim // num_heads, block_size, embedding_dim) 
              for _ in range(num_blocks)]
        )
        self.ln_final     = torch.nn.LayerNorm(embedding_dim)
        self.output_layer = torch.nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # x: [batch, blocksize] - token IDs as integers
        
        x = self.token_emb(x)    # [batch, block_size, embedding_dim]
        x = self.pos_enc(x)      # [batch, block_size, embedding_dim]  (position added)
        x = self.blocks(x)       # [batch, block_size, embedding_dim]  (N blocks)
        x = self.ln_final(x)     # [batch, block_size, embedding_dim]  (stabilized)
        logits = self.output_layer(x)  # [batch, block_size, vocab_size]
        
        return logits


class TransformerBlock(torch.nn.Module):
    """The Transformer Block is the repeating unit of the architecture - stacking N of these is what gives the model its depth."""
    
    def __init__(self, num_heads, head_size, block_size, embedding_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(num_heads, head_size, block_size, embedding_dim)
        self.ffn = FeedForward(embedding_dim, 4* embedding_dim)
        self.ln1 = torch.nn.LayerNorm(embedding_dim)
        self.ln2 = torch.nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        # Sub-layer 1: Multi-Head Self-Attention with Pre-Norm + Residual
        x = x + self.attn(self.ln1(x))
        
        # Sub-layer 2: Feed-Forward with Pre-Norm + Resiudal
        x = x + self.ffn(self.ln2(x))
        
        return x

# ===== Positional Encoding =====

class PositionalEncoding(torch.nn.Module):
    """Adds a postition-dependet signal to each token's embedding before it enters the Transformer.
    
    Formular
    -------
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    
    with: 
    - pos:   the position in the sequence (0, 1, 2, ...)
    - i:     the dimension index (0, 1, 2, ..., d_model/2 - 1)
    - 10000: large base that spreads the frequencies widely
    - each even dim uses sin, each odd dim uses cosine
    """

    def __init__(self, embedding_dim, block_size, device='cpu'):

        super().__init__()
        self.embedding_dim = embedding_dim

        # PE Matrix [block_size, embedding_dim]
        pe = torch.zeros(block_size, embedding_dim)

        for pos in range(block_size):
            for i in range(embedding_dim // 2):
                freq = 1 / (10000 ** (2 * i / embedding_dim))
                pe[pos, 2 * i] = math.sin(pos * freq)         # even dimensions: sine
                pe[pos, 2 * i + 1] = math.cos(pos * freq)     # odd dimensions: cosine

        # register_buffer: saved with model state but NOT a trainable parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, block_size, embedding_dim]
        return x + self.pe[: x.shape[1], :]                   # broadcast across batch dimension


# ===== Self-Attention =====


class SingleHeadSelfAttention(torch.nn.Module):
    """Self-attention lets every token look at every other token and decide how much to incorporate 
    from each one when building its own representation.
    
    Formular
    -------
    Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
    
    with:
    - Query (Q): "What information am I looking for?"
    - Key (K): "What information do I advertise?"
    - Value (V): "What information do I actually share if someone attends to me?"
    """
    
    def __init__(self, head_size, block_size, embedding_dim):
        super().__init__()
        self.head_size = head_size
        self.key   = torch.nn.Linear(embedding_dim, head_size, bias=False)
        self.query = torch.nn.Linear(embedding_dim, head_size, bias=False)
        self.value = torch.nn.Linear(embedding_dim, head_size, bias=False)
        # Lower-triangular mask: registered as buffer (not a parameter)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self,x):
        B, T, C = x.shape  # batch, time (sequence length), channels (embedding_dim)

        Q = self.query(x)  # [B, T, head_size]
        K = self.key(x)    # [B, T, head_size]
        V = self.value(x)  # [B, T, head_size]

        # Scaled dot-product attention scores
        scores = Q @ K.transpose(-2, -1) * (self.head_size ** -0.5)  # [B, T, T]

        # Causal mask: future positions → -inf
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Attention weights + weighted value aggregation
        attn_weights = torch.softmax(scores, dim=-1)   # [B, T, T]
        out = attn_weights @ V                         # [B, T, head_size]
        return out 

class MultiHeadSelfAttention(torch.nn.Module):
    """
    A single attention head can only model one type of relationship at a time. 
    But natural language is rich with overlapping structure:word order, 
    subject-verb agreement, pronoun reference, semantic similarity, and more.

    With num_heads=4 heads, each head can independently learn to focus on different patterns:

        - Head 1 might track grammatical agreement between nouns and adjectives
        - Head 2 might capture long-range subject-verb dependencies
        - Head 3 might look at the immediately preceding word
        - Head 4 might learn semantic similarity across the sentence
    None of this is hand-designed — the specialization emerges from training.
    """
    def __init__(self, num_heads, head_size, block_size, embedding_dim):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [SingleHeadSelfAttention(head_size, block_size, embedding_dim) for _ in range(num_heads)]
        )
        # Output projection: maps concatenated heads back to embedding_dim
        self.proj = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        # Run all heads in parallel (PyTorch handles this efficiently)
        head_outputs = [head(x) for head in self.heads]  # list of [B, T, head_size]

        # Concatenate along the last dimension
        out = torch.cat(head_outputs, dim=-1)            # [B, T, embedding_dim]

        # Learned recombination of all head outputs
        out = self.proj(out)                             # [B, T, embedding_dim]
        return out

class FeedForward(torch.nn.Module):
    """
    The attention mechanism is good at routing information — deciding which tokens to incorporate from where.
    But it is essentially a linear operation on value vectors (the softmax weighting is non-linear, but the value aggregation itself is linear). 
    The FFN adds the non-linear processing power that transforms representations.

    Every position is processed independently and identically — the same two-layer MLP runs on each of the block_size token representations. This is called "position-wise."
    
    Formular
    -------
    FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
    
    with:
    - W1: [embedding_dim, 4 × embedding_dim] — expands from 64 → 256
    - W2: [4 × embedding_dim, embedding_dim] — projects back from 256 → 64
    """
    
    def __init__(self, embedding_dim, ff_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, ff_dim),  # expand: 64 → 256
            torch.nn.ReLU(),
            torch.nn.Linear(ff_dim, embedding_dim),  # compress: 256 → 64
        )

    def forward(self, x):
        return self.net(x)   # applied identically to each position in [B, T, embedding_dim]



