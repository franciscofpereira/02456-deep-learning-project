import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from transformers import AutoModel

class PretrainedEncoder(nn.Module):
    def __init__(self,
                 model_name: str = "distilbert-base-uncased",
                 pooling_type: str = "mean",
                 emb_dim: int | None = None,
                 max_seq_length: int | None = None,
                ):

        super().__init__()
        self.pooling_type = pooling_type
        self.backbone = AutoModel.from_pretrained(model_name, local_files_only=True)
        hidden_size = self.backbone.config.hidden_size

        self.proj = (
            nn.Linear(hidden_size, emb_dim) if emb_dim is not None and emb_dim != hidden_size
            else nn.Identity()
        )

        self.max_seq_length = (
            max_seq_length
            if max_seq_length is not None
            else self.backbone.config.max_position_embeddings
        )

    def pool(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if self.pooling_type == "cls":
            return x[:, 0, :]
        elif self.pooling_type == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return torch.mean(x, dim=1)
        elif self.pooling_type == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1) == 0
                x = x.masked_fill(mask, float("-inf"))
            return torch.max(x, dim=1).values
        raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(outputs.last_hidden_state, attention_mask)
        return self.proj(pooled)
        
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
        pos_embedding: nn.Embedding | None = None,
    ):
        """
        Initialize the TransformerEncoder.
        """
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.pooling_type = pooling_type
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

        self.token_embedding = (
            token_embedding if token_embedding is not None else nn.Embedding(self.vocab_size, emb_dim)
        )
        self.pos_embedding = (
            pos_embedding if pos_embedding is not None else nn.Embedding(max_seq_length, emb_dim)
        )

        self.model = self.setup_model()

    def setup_model(self):
        """
        Instantiate the model.
        """
        
        model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=self.n_heads,
                dim_feedforward=self.feed_forward_dim,
                dropout=0.2,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            ),
            num_layers=self.n_layers
        )

        return model

    def pool(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:

        if self.pooling_type == 'cls':
            return x[:, 0, :]

        elif self.pooling_type == 'max':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1) == 0  # (B, L, 1)
                x = x.masked_fill(mask, float('-inf'))
            return torch.max(x, dim=1).values

        elif self.pooling_type == 'mean':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
                return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return torch.mean(x, dim=1)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.tensor:
        """
        Combines token embeddings with positional embeddings and runs a forward pass on the model
        """

        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        # x is a tensor of dimensions batch_size x sequence_length x emb_dim
        # Pass the input through the transformer encoder
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.model(x, src_key_padding_mask=key_padding_mask)
        
        # Pool the output to get a single vector representation for each input sequence
        x = self.pool(x, attention_mask)

        return x


# ---------------------------------------------------------------------------
# Hand-made transformer encoder
#
# Building blocks adapted from Celinna's work (branch DPR_experimentation).
# The transformer logic (positional encoding, multi-head self-attention,
# encoder layer, pooling) is written by hand. Only basic PyTorch building
# blocks are used: nn.Module, nn.Embedding, nn.Linear, nn.LayerNorm,
# nn.Dropout, nn.GELU and softmax. No nn.TransformerEncoder,
# nn.MultiheadAttention or F.scaled_dot_product_attention.
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding ("Attention Is All You Need").

    PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

    Not trained: stored as a buffer, so it moves with .to(device) but is not a parameter.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        assert d_model % 2 == 0, "emb_dim must be even for sinusoidal positional encoding"
        t = torch.arange(max_len).unsqueeze(1).float()           # (max_len, 1): positions 0, 1, 2, ...
        two_i = torch.arange(0, d_model, 2).float()              # (d_model/2,): 0, 2, 4, ...
        freqs = 1.0 / torch.pow(10000.0, two_i / d_model)        # one frequency per pair of dimensions

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(t * freqs)                       # even dimensions
        pe[:, 1::2] = torch.cos(t * freqs)                       # odd dimensions
        # persistent=False: not saved in state_dict (it can always be recomputed)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model) -> add the position signal to every token
        return x + self.pe[:, :x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention, written by hand.

    attention(Q, K, V) = softmax(Q K^T / sqrt(d_head)) V, computed per head.
    """
    def __init__(self, emb_dim: int, hidden_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = hidden_dim // n_heads                     # size of each head

        self.q_proj = nn.Linear(emb_dim, hidden_dim)
        self.k_proj = nn.Linear(emb_dim, hidden_dim)
        self.v_proj = nn.Linear(emb_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, emb_dim)          # mixes the heads back together
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: (B, T, emb_dim)   mask: (B, T) with 1 = real token, 0 = padding
        B, T, _ = x.shape

        # (B, T, hidden) -> (B, T, H, d_head) -> (B, H, T, d_head)
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / sqrt(self.d_head)    # (B, H, T, T)

        # Where the KEY is padding, set the score to a very large negative number,
        # so softmax gives it ~0 weight: no attention to padding.
        # (finfo.min instead of -inf avoids NaNs, also under fp16 autocast.)
        scores = scores.masked_fill(mask[:, None, None, :] == 0, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)                        # (B, H, T, T)
        out = self.attn_dropout(attn) @ v                       # (B, H, T, d_head)

        # Merge the heads: (B, H, T, d_head) -> (B, T, hidden)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out), attn


class EncoderLayer(nn.Module):
    """One transformer block: self-attention + feed-forward, each with residual + LayerNorm (post-norm)."""
    def __init__(self, emb_dim: int, hidden_dim: int, n_heads: int, feed_forward_dim: int, dropout: float = 0.2):
        super().__init__()
        self.attn = MultiHeadSelfAttention(emb_dim, hidden_dim, n_heads, dropout)
        self.ff = nn.Sequential(                                # MLP applied to each token independently
            nn.Linear(emb_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, emb_dim),
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_attn = None                                   # attention maps of the last forward pass

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_out, attn = self.attn(x, mask)
        self.last_attn = attn.detach()                          # kept for visualisation only
        x = self.norm1(x + self.dropout(attn_out))              # residual + layer norm
        x = self.norm2(x + self.dropout(self.ff(x)))            # residual + layer norm
        return x


class HandTransformerEncoder(nn.Module):
    """Hand-made transformer encoder with the same interface as TransformerEncoder.

    Input:  input_ids (B, T) token ids from the tokenizer, attention_mask (B, T) with 1 = token, 0 = padding
    Output: (B, emb_dim), one vector per text
    """
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
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.pooling_type = pooling_type
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length                    # read by the trainer
        self.dropout = dropout

        # token id -> vector (lookup table)
        self.token_embedding = (
            token_embedding if token_embedding is not None else nn.Embedding(vocab_size, emb_dim)
        )
        self.pos_enc = PositionalEncoding(emb_dim, max_seq_length)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = self.setup_model()

    def setup_model(self) -> nn.ModuleList:
        # ModuleList (not a normal list) so PyTorch registers the parameters of every layer
        return nn.ModuleList([
            EncoderLayer(self.emb_dim, self.hidden_dim, self.n_heads, self.feed_forward_dim, self.dropout)
            for _ in range(self.n_layers)
        ])

    def pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, T, emb_dim) -> (B, emb_dim), ignoring padding tokens."""
        if self.pooling_type == 'cls':
            return x[:, 0, :]                                   # first token is [CLS]

        elif self.pooling_type == 'max':
            return x.masked_fill(mask.unsqueeze(-1) == 0, float('-inf')).max(dim=1).values

        elif self.pooling_type == 'mean':
            m = mask.unsqueeze(-1).to(x.dtype)
            return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)   # mean of non-padding tokens

        raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

    def get_attention_maps(self) -> list[torch.Tensor]:
        """Attention maps (B, H, T, T) of every layer from the last forward pass."""
        return [layer.last_attn for layer in self.layers]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        x = self.token_embedding(input_ids)                     # (B, T, emb_dim)
        x = self.emb_dropout(self.pos_enc(x))                   # add position information

        for layer in self.layers:
            x = layer(x, attention_mask)

        return self.pool(x, attention_mask)                     # (B, emb_dim)
