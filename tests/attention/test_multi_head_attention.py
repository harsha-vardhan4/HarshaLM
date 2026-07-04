import torch

from utils.config import ModelConfig
from model.attention import MultiHeadAttention


def test_multi_head_attention():

    config = ModelConfig()

    mha = MultiHeadAttention(config)

    x = torch.randn(
        2,
        8,
        config.embedding_dim
    )

    output = mha(x)

    print("Input Shape:")
    print(x.shape)

    print()

    print("Output Shape:")
    print(output.shape)

    assert output.shape == (
        2,
        8,
        config.embedding_dim
    )

    print("\n✓ MultiHeadAttention test passed")


if __name__ == "__main__":
    test_multi_head_attention()