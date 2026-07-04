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

        self.qkv_projection = nn.Linear(
            self.embedding_dim,
            self.embedding_dim * 3,
            bias=False
        )

        self.output_projection = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=False
        )

        self.attention = ScaledDotProductAttention(config)

    def _project_qkv(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Projects the input embeddings into a combined
        Query-Key-Value tensor.
        """
        return self.qkv_projection(x)

    def _split_qkv(
        self,
        qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Splits the combined QKV tensor into
        Query, Key and Value tensors.
        """

        query, key, value = torch.chunk(
            qkv,
            chunks=3,
            dim=-1
        )

        return query, key, value
    
    def _split_heads(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Splits the embedding dimension into multiple attention heads.

        Input:
            (batch, sequence, embedding_dim)

        Output:
            (batch, heads, sequence, head_dim)
        """

        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        x = x.transpose(1, 2)

        return x
    
    def _prepare_attention_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts Query, Key and Value tensors into
        multi-head format expected by the attention module.
        """

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        return query, key, value
    
    def _merge_heads(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Merges multiple attention heads back into a single embedding.

        Input:
            (batch, heads, sequence, head_dim)

        Output:
            (batch, sequence, embedding_dim)
        """

        batch_size, _, sequence_length, _ = x.shape

        x = x.transpose(1, 2)

        x = x.reshape(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        return x
    
    def _output_projection(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the final linear projection after
        merging all attention heads.

        Input:
            (batch, sequence, embedding_dim)

        Output:
            (batch, sequence, embedding_dim)
        """

        return self.output_projection(x)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:

        qkv = self._project_qkv(x)

        query, key, value = self._split_qkv(qkv)

        query, key, value = self._prepare_attention_inputs(
            query,
            key,
            value
        )

        attention_output = self.attention(
            query,
            key,
            value,
            attention_mask
        )

        attention_output = self._merge_heads(attention_output)

        output = self._output_projection(attention_output)

        return output