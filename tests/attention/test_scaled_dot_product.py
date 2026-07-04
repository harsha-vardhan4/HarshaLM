import torch

from model.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)

from utils.config import ModelConfig


def test_scaled_dot_product_attention():

    config = ModelConfig()
    config.attention_dropout = 0.0  # deterministic

    attention = ScaledDotProductAttention(config)
    attention.eval()

    head_dim = config.embedding_dim // config.num_heads

    q = torch.randn(2, config.num_heads, 8, head_dim)
    k = torch.randn(2, config.num_heads, 8, head_dim)
    v = torch.randn(2, config.num_heads, 8, head_dim)

    output = attention(q, k, v)

    assert output.shape == (
        2,
        config.num_heads,
        8,
        head_dim,
    )

    assert torch.isfinite(output).all()

    assert output.dtype == q.dtype

    print("✓ Shape test passed")
    print("✓ Finite values test passed")
    print("✓ Dtype test passed")

if __name__ == "__main__":
    test_scaled_dot_product_attention()