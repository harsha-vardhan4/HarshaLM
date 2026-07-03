import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """
    Single-head self-attention.

    This module currently implements only the
    Query, Key and Value projection layers.

    Attention computation will be added
    in the next step.
    """

    def __init__(
        self,
        embedding_dim: int
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

    def forward(
        self,
        x: torch.Tensor
    ):

        """ Input shape: (batch_size, sequence_length, embedding_dim)"""

        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)

        # Compute attention scores
        attention_scores = Q @ K.transpose(-2, -1)

        # Scale scores
        attention_scores = attention_scores / math.sqrt(self.embedding_dim)

        return attention_scores, V