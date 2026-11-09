from __future__ import annotations

import torch
from torch import nn


MODEL_VERSION = "formal_osla_boundary_v1"
INPUT_SIZE = 7


class FormalOneShotGRU(nn.Module):
    """Actor-only GRUCell policy for the formal one-shot LossA path."""

    def __init__(self, input_size: int = INPUT_SIZE, hidden_size: int = 32, dropout: float = 0.0):
        super().__init__()
        if input_size != INPUT_SIZE:
            raise ValueError(f"formal model requires input_size={INPUT_SIZE}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = float(dropout)
        self.core = nn.GRUCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.actor = nn.Linear(hidden_size, 1)

    def initial_hidden(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def step(self, inputs: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_next = self.core(inputs, hidden)
        logits = self.actor(self.dropout(hidden_next)).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        return hidden_next, logits, probabilities

    def parameter_inventory(self) -> dict[str, object]:
        tensors = [
            {"name": name, "shape": list(param.shape), "count": int(param.numel())}
            for name, param in self.named_parameters()
        ]
        return {
            "model_version": MODEL_VERSION,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout_rate,
            "total_parameters": int(sum(item["count"] for item in tensors)),
            "trainable_parameters": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
            "tensors": tensors,
        }
