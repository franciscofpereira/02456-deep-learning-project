import torch
import torch.nn as nn


class DPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # loss = -log( exp(s_pos) / (exp(s_pos) + Σ exp(s_neg)) )
    
    def forward(self, queries: torch.tensor, documents: torch.tensor):
        """
        @param queries: batch_size x embedding_dim
        @param documents: batch_size x embedding_dim
        Returns: Loss fθ(Di)^T * (gϕ(Qi)) -> where f is document encoder and g is query encoder
        """
        scores = queries @ documents.T
        labels = torch.arange(scores.size(0), device=scores.device)
        return nn.functional.cross_entropy(scores, labels)