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

    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()

        self.config = config

        #
        # Embedding Layer
        #

        self.embedding_layer = EmbeddingLayer(
            config
        )

        #
        # Transformer Stack
        #

        self.transformer = TransformerStack(
            config
        )

        #
        # Final LayerNorm
        #

        self.final_layer_norm = nn.LayerNorm(
            config.embedding_dim
        )

        #
        # Language Modeling Head
        #

        self.lm_head = LMHead(
            config
        )

        #
        # Weight Tying
        #

        self.tie_weights()

    def tie_weights(
        self,
    ):
        """
        Shares the token embedding matrix with the
        language modeling head.

        This reduces the number of trainable
        parameters and matches GPT-2/GPT-3.
        """

        self.lm_head.linear.weight = (
            self.embedding_layer
                .token_embedding
                .embedding
                .weight
        )

    def verify_weight_tying(
        self,
    ) -> bool:
        """
        Verifies that the embedding layer and
        LM head share the same weights.
        """

        return (

            self.embedding_layer
                .token_embedding
                .embedding
                .weight
                .data_ptr()

            ==

            self.lm_head
                .linear
                .weight
                .data_ptr()

        )

    def num_parameters(
        self,
    ) -> int:
        """
        Returns the total number of parameters.
        """

        return sum(

            parameter.numel()

            for parameter in self.parameters()

        )

    def trainable_parameters(
        self,
    ) -> int:
        """
        Returns the number of trainable parameters.
        """

        return sum(

            parameter.numel()

            for parameter in self.parameters()

            if parameter.requires_grad

        )

    def model_size_mb(
        self,
    ) -> float:
        """
        Approximate model size in MB.
        """

        total_bytes = sum(

            parameter.numel()
            * parameter.element_size()

            for parameter in self.parameters()

        )

        return total_bytes / (1024 ** 2)

    def summary(
        self,
    ):
        """
        Prints a summary of the model.
        """

        print()

        print("=" * 60)

        print("HarshaLM Summary")

        print("=" * 60)

        print(
            f"Vocabulary Size     : {self.config.vocab_size:,}"
        )

        print(
            f"Context Length      : {self.config.context_length}"
        )

        print(
            f"Embedding Dimension : {self.config.embedding_dim}"
        )

        print(
            f"Transformer Layers  : {self.config.num_layers}"
        )

        print(
            f"Attention Heads     : {self.config.num_heads}"
        )

        print()

        print(
            f"Parameters          : {self.num_parameters():,}"
        )

        print(
            f"Trainable Params    : {self.trainable_parameters():,}"
        )

        print(
            f"Model Size          : {self.model_size_mb():.2f} MB"
        )

        print()

        print(
            f"Weight Tied         : {self.verify_weight_tying()}"
        )

        print("=" * 60)

        print()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Input:
            input_ids:
                (batch_size, sequence_length)

        Output:
            logits:
                (batch_size, sequence_length, vocab_size)
        """

        embeddings = self.embedding_layer(
            input_ids
        )

        hidden_states = self.transformer(
            embeddings,
            attention_mask,
        )

        hidden_states = self.final_layer_norm(
            hidden_states
        )

        logits = self.lm_head(
            hidden_states
        )

        return logits