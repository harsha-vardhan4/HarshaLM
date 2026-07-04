import torch

from utils.config import ModelConfig
from model.embeddings.token_embedding import TokenEmbedding


def test_token_embedding():

    config = ModelConfig()

    embedding = TokenEmbedding(config)

    token_ids = torch.tensor(
        [
            [1, 5, 9, 2],
            [7, 3, 8, 4]
        ]
    )

    output = embedding(token_ids)

    print("Input Shape:")
    print(token_ids.shape)

    print()

    print("Output Shape:")
    print(output.shape)

    assert output.shape == (
        2,
        4,
        config.embedding_dim
    )

    print("\n✓ TokenEmbedding test passed")


if __name__ == "__main__":
    test_token_embedding()