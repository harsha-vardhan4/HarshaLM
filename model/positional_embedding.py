import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embedding.

    Every position in the sequence has its own trainable vector.
    """

    def __init__(
        self,
        context_length: int,
        embedding_dim: int
    ):
        super().__init__()

        self.position_embedding = nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=embedding_dim
        )

    def forward(self, token_embeddings: torch.Tensor):

        """
        token_embeddings shape:

        (batch_size, sequence_length, embedding_dim)
        """

        batch_size, sequence_length, embedding_dim = token_embeddings.shape

        positions = torch.arange(
            sequence_length,
            device=token_embeddings.device
        )

        position_vectors = self.position_embedding(
            positions
        )

        position_vectors = position_vectors.unsqueeze(0)

        return token_embeddings + position_vectors