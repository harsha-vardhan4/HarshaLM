import torch

from utils.config import ModelConfig
from model.transformer.transformer_block import TransformerBlock


def test_transformer_block():

    config = ModelConfig()

    block = TransformerBlock(config)

    x = torch.randn(
        2,
        8,
        config.embedding_dim
    )

    output = block(x)

    print("Input Shape:")
    print(x.shape)

    print()

    print("Output Shape:")
    print(output.shape)

    assert output.shape == x.shape

    print("\n✓ TransformerBlock test passed")


if __name__ == "__main__":
    test_transformer_block()