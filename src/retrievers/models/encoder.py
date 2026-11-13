import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int, pooling_type: str):
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
                activation='relu',
                batch_first=True
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
        
    def forward(self, x: torch.tensor, attention_mask: torch.tensor = None) -> torch.tensor:
        """
        Run a forward pass on the model
        """
        # x is a tensor of dimensions batch_size x sequence_length x emb_dim
        # Pass the input through the transformer encoder
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.model(x, src_key_padding_mask=key_padding_mask)
        
        # Pool the output to get a single vector representation for each input sequence
        x = self.pool(x, attention_mask)
        
        return x
