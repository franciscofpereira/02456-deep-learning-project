import torch
import torch.nn as nn
import torch.nn.functional as F


class DPRLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, queries: torch.Tensor, documents: torch.Tensor):
        queries = F.normalize(queries, p=2, dim=-1)
        documents = F.normalize(documents, p=2, dim=-1)
        scores = (queries @ documents.T) / self.temperature
        labels = torch.arange(scores.size(0), device=scores.device)

        loss_q2d = F.cross_entropy(scores, labels)
        loss_d2q = F.cross_entropy(scores.T, labels)
        return 0.5 * (loss_q2d + loss_d2q)