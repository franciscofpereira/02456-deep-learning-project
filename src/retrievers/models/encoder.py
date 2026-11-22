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
            token_embedding if token_embedding is not None else nn.Embedding(self.vocab_size, hidden_dim)
        )
        self.pos_embedding = (
            pos_embedding if pos_embedding is not None else nn.Embedding(max_seq_length, hidden_dim)
        )
    
        self.proj = (
            nn.Linear(hidden_dim, emb_dim) if emb_dim is not None and emb_dim != hidden_dim
            else nn.Identity()
        )

        self.model = self.setup_model()

    def setup_model(self):
        """
        Instantiate the model.
        """
        
        model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.n_heads,
                dim_feedforward=self.feed_forward_dim,
                dropout=0.1,
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

        x = self.proj(x)

        return x
