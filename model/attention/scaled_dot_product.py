import math
from typing import Optional

import torch
import torch.nn as nn

from utils.config import ModelConfig


class ScaledDotProductAttention(nn.Module):
    """
    Performs scaled dot-product attention.

    Expected tensor shape:

        (batch, heads, sequence_length, head_dimension)

    This module is responsible only for computing attention.
    It does not perform QKV projection or head splitting.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads

        self.dropout = nn.Dropout(
            config.attention_dropout
        )

        causal_mask = torch.triu(
            torch.ones(
                config.context_length,
                config.context_length,
                dtype=torch.bool
            ),
            diagonal=1
        )

        self.register_buffer(
            "causal_mask",
            causal_mask
        )

    def _compute_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor
    ) -> torch.Tensor:

        scores = (
            query @ key.transpose(-2, -1)
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        return scores

    def _apply_mask(
        self,
        scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        sequence_length = scores.size(-1)

        if attention_mask is None:

            attention_mask = self.causal_mask[
                :sequence_length,
                :sequence_length
            ]

        scores = scores.masked_fill(
            attention_mask,
            float("-inf")
        )

        return scores

    def _compute_attention_weights(
        self,
        scores: torch.Tensor
    ) -> torch.Tensor:

        weights = torch.softmax(
            scores,
            dim=-1
        )

        weights = self.dropout(
            weights
        )

        return weights

    def _compute_output(
        self,
        attention_weights: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:

        return attention_weights @ value

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        assert query.dim() == 4, "Query must be 4-dimensional."
        assert key.dim() == 4, "Key must be 4-dimensional."
        assert value.dim() == 4, "Value must be 4-dimensional."

        assert (
            query.shape == key.shape == value.shape
        ), "Q, K and V must have identical shapes."

        scores = self._compute_scores(
            query,
            key
        )

        scores = self._apply_mask(
            scores,
            attention_mask
        )

        attention_weights = (
            self._compute_attention_weights(
                scores
            )
        )

        output = self._compute_output(
            attention_weights,
            value
        )

        return output