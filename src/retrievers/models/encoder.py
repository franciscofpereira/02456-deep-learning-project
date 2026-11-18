import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class PositionalEncoding(nn.Module): # notebook inspired function
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        assert d_model % 2 == 0, "emb_dim must be even for positional encoding"
        t = torch.arange(max_len).unsqueeze(1).float() # (max_len,1): positions 0, 1, 2, ...
        two_i = torch.arange(0, d_model, 2).float() # (d_model/2,): 0, 2, 4, ...
        freqs = 1.0 / torch.pow(10000.0, two_i / d_model) # one frequency per pair of dimensions


        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(t * freqs) # even dimensions
        pe[:, 1::2] = torch.cos(t * freqs) # odd dimensions
        # register_buffer: model is saved but not trained
        self.register_buffer("pe", pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model) -> add position to each token
        return x + self.pe[:, :x.size(1)]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_heads):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = hidden_dim // n_heads # size of each head

        self.q_proj = nn.Linear(emb_dim, hidden_dim)
        self.k_proj = nn.Linear(emb_dim, hidden_dim)
        self.v_proj = nn.Linear(emb_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, emb_dim)  # mixes the heads back together

    def forward(self, x, mask):
        # x: (B, T, emb_dim) mask: (B, T) con 1 = real token, 0 = padding
        B, T, _ = x.shape

        # (B, T, hidden) -> (B, T, H, d_head) -> (B, H, T, d_head)
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / sqrt(self.d_head)

        # Where the KEY is padding, score = -inf, and the softmax converts -inf to 0: no attention to the padding
        scores = scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1) # (B, H, T, T)
        out = attn @ v # (B, H, T, d_head)

        # mix the heads together: (B, H, T, d_head) -> (B, T, hidden)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out), attn

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_heads, feed_forward_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(emb_dim, hidden_dim, n_heads)
        self.ff = nn.Sequential( # MultiLayerPerceptron that proccesses each token independently
            nn.Linear(emb_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, emb_dim),
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x, mask):
        attn_out, attn = self.attn(x, mask)
        self.last_attn = attn # attention maps
        x = self.norm1(x + attn_out) # residual + layer norm
        x = self.norm2(x + self.ff(x)) # residual + layer norm
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, n_heads, emb_dim, hidden_dim, feed_forward_dim, pooling_type, vocab_size, max_seq_length, token_embedding=None):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim =    emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.pooling_type = pooling_type
        self.setup_model()
        self.max_seq_length = max_seq_length
        self.token_embedding = (token_embedding if token_embedding is not None
                                else nn.Embedding(vocab_size, emb_dim))

    def setup_model(self):
        """
        Instantiate the model.
        """
        self.pos_enc = PositionalEncoding(self.emb_dim, max_len=self.max_seq_length)
        # Module list (not using a normal list) for PyTorch to register the parameters
        self.model = nn.ModuleList([
            EncoderLayer(self.emb_dim, self.hidden_dim, self.n_heads, self.feed_forward_dim) 
            for _ in range(self.n_layers)
        ])

    def pool(self, x, mask):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension 
        Returns: A pooled vector of shape batch_size x emb_dim
        """
        if self.cfg.model.pooling_type == 'cls':
            return x[:, 0, :] # first token is the CLS token
        
        elif self.cfg.model.pooling_type == 'max':
            return x.masked_fill(mask.unsqueeze(-1)==0, float("-inf")).max(dim=1).values
        
        elif self.cfg.model.pooling_type == 'mean':
            m = mask.unsqueeze(-1).to(x.dtype)
            return (x * m).sum(dim=1) / m.sum(dim=1) # mean of non-padding tokens

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        x = self.token_embedding(input_ids)      # token embedding (ids instead of vectors)
        x = self.pos_enc(x)
        for layer in self.model:
            x = layer(x, attention_mask)
        return self.pool(x, attention_mask)
