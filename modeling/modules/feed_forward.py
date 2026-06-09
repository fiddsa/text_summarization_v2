import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise feed-forward network from the original Transformer."""

    def __init__(self, dim: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)
