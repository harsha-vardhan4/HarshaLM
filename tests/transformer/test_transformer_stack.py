import torch

from utils.config import ModelConfig
from model.transformer.transformer_stack import TransformerStack


def test_transformer_stack():

    config = ModelConfig()

    stack = TransformerStack(config)

    x = torch.randn(
        2,
        8,
        config.embedding_dim
    )

    output = stack(x)

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

    print(f"\n✓ TransformerStack ({config.num_layers} layers) test passed")


if __name__ == "__main__":
    test_transformer_stack()