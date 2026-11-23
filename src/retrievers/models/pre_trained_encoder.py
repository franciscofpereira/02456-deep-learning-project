from transformers import AutoModel
import torch.nn as nn
import torch

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