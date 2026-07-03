import torch
import torch.nn as nn


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

        """
        Input shape:

        (batch_size,
         sequence_length,
         embedding_dim)
        """

        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)

        return Q, K, V