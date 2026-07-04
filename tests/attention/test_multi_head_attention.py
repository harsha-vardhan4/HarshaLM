import torch.nn as nn

from model.attention.multi_head_attention import (
    MultiHeadAttention,
)
from utils.config import ModelConfig


def test_constructor():

    config = ModelConfig()

    mha = MultiHeadAttention(config)

    assert mha.embedding_dim == 256
    assert mha.num_heads == 4
    assert mha.head_dim == 64

    assert isinstance(
        mha.qkv_projection,
        nn.Linear
    )

    assert isinstance(
        mha.output_projection,
        nn.Linear
    )

    print("✓ Constructor test passed")

if __name__ == "__main__":
    test_constructor()