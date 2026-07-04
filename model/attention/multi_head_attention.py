import torch
import torch.nn as nn

from utils.config import ModelConfig
from model.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)


class MultiHeadAttention(nn.Module):
    """
    GPT-style Multi-Head Self-Attention.

    Responsibilities:
    - QKV Projection
    - Split into multiple heads
    - Run scaled dot-product attention
    - Merge heads
    - Output projection
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        assert (
            config.embedding_dim % config.num_heads == 0
        ), "embedding_dim must be divisible by num_heads"

        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = (
            config.embedding_dim //
            config.num_heads
        )

        # One projection for Q, K and V
        self.qkv_projection = nn.Linear(
            self.embedding_dim,
            self.embedding_dim * 3,
            bias=False
        )

        # Final projection
        self.output_projection = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=False
        )

        self.attention = ScaledDotProductAttention(
            config
        )