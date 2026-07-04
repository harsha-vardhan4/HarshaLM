import torch

from utils.config import ModelConfig
from model.embeddings.embedding_layer import EmbeddingLayer


def test_embedding_layer():

    config = ModelConfig()

    embedding = EmbeddingLayer(config)

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 8)
    )

    output = embedding(input_ids)

    print("Input Shape:")
    print(input_ids.shape)

    print()

    print("Output Shape:")
    print(output.shape)

    assert output.shape == (
        2,
        8,
        config.embedding_dim
    )

    print("\n✓ EmbeddingLayer test passed")


if __name__ == "__main__":
    test_embedding_layer()