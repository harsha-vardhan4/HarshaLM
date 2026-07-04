import torch

from utils.config import ModelConfig
from model.transformer.feed_forward import FeedForward


def test_feed_forward():

    config = ModelConfig()

    ffn = FeedForward(config)

    x = torch.randn(
        2,
        8,
        config.embedding_dim
    )

    output = ffn(x)

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

    print("\n✓ FeedForward test passed")


if __name__ == "__main__":
    test_feed_forward()