import torch

from utils.config import ModelConfig
from model.embeddings.token_embedding import TokenEmbedding
from model.embeddings.positional_embedding import PositionalEmbedding


def test_positional_embedding():

    config = ModelConfig()

    token_embedding = TokenEmbedding(config)

    positional_embedding = PositionalEmbedding(config)

    token_ids = torch.tensor(
        [
            [1, 5, 9, 2],
            [7, 3, 8, 4]
        ]
    )

    token_vectors = token_embedding(token_ids)

    output = positional_embedding(token_vectors)

    print("Input Shape:")
    print(token_vectors.shape)

    print()

    print("Output Shape:")
    print(output.shape)

    assert output.shape == (
        2,
        4,
        config.embedding_dim
    )

    print("\n✓ PositionalEmbedding test passed")


if __name__ == "__main__":
    test_positional_embedding()