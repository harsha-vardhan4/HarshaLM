import torch
import torch.nn as nn

from utils.config import ModelConfig

from model.embeddings.embedding_layer import EmbeddingLayer
from model.heads.lm_head import LMHead
from model.transformer.transformer_stack import TransformerStack


class HarshaLM(nn.Module):
    """
    GPT-style Decoder Only Language Model.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        #Embedding
        self.embedding_layer = EmbeddingLayer(config)
        
        # Transformer Stack
        self.transformer = TransformerStack(config)

        # Final LayerNorm
        self.final_layer_norm = nn.LayerNorm(
            config.embedding_dim
        )

        # Language Modeling Head
        self.lm_head = LMHead(config)

        # Tie token embedding and LM head weights
        self.lm_head.linear.weight = (
            self.embedding_layer
                .token_embedding
                .embedding
                .weight
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

        embeddings = self.embedding_layer(
            input_ids
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