import torch
import torch.nn as nn

from utils.config import ModelConfig


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embedding.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.position_embedding = nn.Embedding(
            num_embeddings=config.context_length,
            embedding_dim=config.embedding_dim
        )

    def forward(
        self,
        token_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Input:
            (batch, sequence, embedding_dim)

        Output:
            (batch, sequence, embedding_dim)
        """

        _, sequence_length, _ = token_embeddings.shape

        positions = torch.arange(
            sequence_length,
            device=token_embeddings.device
        )

        position_embeddings = self.position_embedding(
            positions
        )

        position_embeddings = position_embeddings.unsqueeze(0)

        return token_embeddings + position_embeddings