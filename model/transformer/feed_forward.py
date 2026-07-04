import torch
import torch.nn as nn

from utils.config import ModelConfig


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network.

    Architecture:
        Linear
            ↓
        GELU
            ↓
        Dropout
            ↓
        Linear
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                config.embedding_dim,
                config.feed_forward_dim
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(
                config.feed_forward_dim,
                config.embedding_dim
            )
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the Feed Forward Network.

        Input:
            (batch, sequence, embedding_dim)

        Output:
            (batch, sequence, embedding_dim)
        """

        return self.network(x)