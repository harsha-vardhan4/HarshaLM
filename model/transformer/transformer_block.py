import torch
import torch.nn as nn

from utils.config import ModelConfig
from model.attention.multi_head_attention import MultiHeadAttention
from model.transformer.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """
    GPT-style Pre-LayerNorm Transformer Block.

    Architecture:

        Input
           │
        LayerNorm
           │
        Multi-Head Attention
           │
        Dropout
           │
        Residual Add
           │
        LayerNorm
           │
        Feed Forward
           │
        Dropout
           │
        Residual Add
           │
         Output
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.attention_norm = nn.LayerNorm(
            config.embedding_dim
        )

        self.attention = MultiHeadAttention(config)

        self.feed_forward_norm = nn.LayerNorm(
            config.embedding_dim
        )

        self.feed_forward = FeedForward(config)

        self.residual_dropout = nn.Dropout(
            config.dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass through a GPT-style Transformer block.

        Input:
            x:
                (batch, sequence, embedding_dim)

        Output:
            (batch, sequence, embedding_dim)
        """

        # -----------------------------
        # Multi-Head Self-Attention
        # -----------------------------

        residual = x

        attention_output = self.attention(
            self.attention_norm(x),
            attention_mask
        )

        attention_output = self.residual_dropout(
            attention_output
        )

        x = residual + attention_output

        # -----------------------------
        # Feed Forward Network
        # -----------------------------

        residual = x

        feed_forward_output = self.feed_forward(
            self.feed_forward_norm(x)
        )

        feed_forward_output = self.residual_dropout(
            feed_forward_output
        )

        x = residual + feed_forward_output

        return x