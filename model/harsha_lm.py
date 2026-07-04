import torch
import torch.nn as nn

from utils.config import ModelConfig
from model.embeddings.token_embedding import TokenEmbedding
from model.embeddings.positional_embedding import PositionalEmbedding
from model.transformer.transformer_stack import TransformerStack


class HarshaLM(nn.Module):
    """
    GPT-style Decoder Only Language Model.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # Token Embedding
        self.token_embedding = TokenEmbedding(config)

        # Positional Embedding
        self.position_embedding = PositionalEmbedding(config)

        # Transformer Stack
        self.transformer = TransformerStack(config)

        # Final LayerNorm
        self.final_layer_norm = nn.LayerNorm(
            config.embedding_dim
        )

        # Language Modeling Head
        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass of HarshaLM.

        Input:
            input_ids:
                (batch, sequence)

        Output:
            logits:
                (batch, sequence, vocab_size)
        """

        token_embeddings = self.token_embedding(
            input_ids
        )

        embeddings = self.position_embedding(
            token_embeddings
        )

        hidden_states = self.transformer(
            embeddings,
            attention_mask
        )

        hidden_states = self.final_layer_norm(
            hidden_states
        )

        logits = self.lm_head(
            hidden_states
        )

        return logits