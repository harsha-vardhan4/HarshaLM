import torch
import torch.nn as nn

from utils.config import ModelConfig
from model.embeddings.token_embedding import TokenEmbedding
from model.embeddings.positional_embedding import PositionalEmbedding


class EmbeddingLayer(nn.Module):
    """
    Combines token embeddings and positional embeddings.

    Architecture:

        Input Token IDs
               │
               ▼
        Token Embedding
               │
               ▼
     Positional Embedding
               │
               ▼
      Final Input Embeddings
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.token_embedding = TokenEmbedding(config)

        self.position_embedding = PositionalEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Input:
            (batch, sequence)

        Output:
            (batch, sequence, embedding_dim)
        """

        token_embeddings = self.token_embedding(
            input_ids
        )

        embeddings = self.position_embedding(
            token_embeddings
        )

        return embeddings