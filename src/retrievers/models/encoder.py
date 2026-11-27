import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for positional encoding"

        positions = torch.arange(max_len).unsqueeze(1).float()       # (max_len, 1)
        dim_pairs = torch.arange(0, d_model, 2).float()              # (d_model / 2,)
        freqs = 1.0 / torch.pow(10000.0, dim_pairs / d_model)        # (d_model / 2,)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * freqs)                   # even indices
        pe[:, 1::2] = torch.cos(positions * freqs)                   # odd indices

        self.register_buffer("pe", pe.unsqueeze(0))                  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, emb_dim: int, hidden_dim: int, n_heads: int, dropout: float = 0.3) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")

        self.n_heads = n_heads
        self.d_head = hidden_dim // n_heads

        self.q_proj = nn.Linear(emb_dim, hidden_dim)
        self.k_proj = nn.Linear(emb_dim, hidden_dim)
        self.v_proj = nn.Linear(emb_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, emb_dim), mask: (B, T) with 1 = real token, 0 = padding
        B, T, _ = x.shape

        # (B, T, hidden_dim) -> (B, T, H, d_head) -> (B, H, T, d_head)
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Mask out padded key positions using dtype-safe minimum for float16 AMP
        scores = scores.masked_fill(mask[:, None, None, :] == 0, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)  # (B, H, T, T)
        attn = self.dropout(attn)
        out = attn @ v                    # (B, H, T, d_head)

        # Merge heads: (B, H, T, d_head) -> (B, T, hidden_dim) -> (B, T, emb_dim)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out), attn
    
class EncoderLayer(nn.Module):
    
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        n_heads: int,
        feed_forward_dim: int
        ):

        super().__init__()

        self.attn = MultiHeadSelfAttention(emb_dim, hidden_dim, n_heads)

        self.ff = nn.Sequential( 
            nn.Linear(emb_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, emb_dim),
        )

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_out, attn = self.attn(x, mask)
        self.last_attn = attn # attention maps
        x = self.norm1(x + attn_out) 
        x = self.norm2(x + self.ff(x)) 
        return x

class TransformerEncoder(nn.Module):
    
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        emb_dim: int, 
        hidden_dim: int,
        feed_forward_dim: int, 
        pooling_type: str, 
        vocab_size: int, 
        max_seq_length: int, 
        token_embedding: nn.Embedding | None = None,
        ):

        """
        Initialize the TransformerEncoder.
        """
        
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim =    emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.pooling_type = pooling_type
        self.max_seq_length = max_seq_length    
        self.setup_model()
        self.token_embedding = (token_embedding if token_embedding is not None
                                else nn.Embedding(vocab_size, emb_dim))
        
    def setup_model(self):
        """
        Instantiate the model.
        """
        self.pos_enc = PositionalEncoding(d_model=self.emb_dim, max_len=self.max_seq_length)

        self.model = nn.ModuleList([
            EncoderLayer(self.emb_dim, self.hidden_dim, self.n_heads, self.feed_forward_dim) 
            for _ in range(self.n_layers)
        ])

    def pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Given a (batch_size, sequence_length, emb_dim) tensor, pool along sequence_length.
        Returns: A pooled vector of shape (batch_size, emb_dim)
        """
        if self.pooling_type == 'cls':
            return x[:, 0, :]

        elif self.pooling_type == 'max':
            return x.masked_fill(mask.unsqueeze(-1) == 0, torch.finfo(x.dtype).min).max(dim=1).values

        elif self.pooling_type == 'mean':
            m = mask.unsqueeze(-1).to(x.dtype)
            return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)

        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        '''
        Forward pass through the TransformerEncoder.
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length) containing token ids.
            attention_mask: Tensor of shape (batch_size, sequence_length) containing 1 for real tokens and 0 for padding tokens.
        Returns:
            Tensor of shape (batch_size, emb_dim) containing the pooled embeddings.'''
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        x = self.token_embedding(input_ids)      
        x = self.pos_enc(x)
        for layer in self.model:
            x = layer(x, attention_mask)
    
        x = self.pool(x, attention_mask)

        return x
