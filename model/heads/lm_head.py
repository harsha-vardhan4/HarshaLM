import torch
import torch.nn as nn

from utils.config import ModelConfig


class LMHead(nn.Module):
    """
    Language Modeling Head.

    Projects the transformer hidden states into
    vocabulary logits.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.output_projection = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Input:
            (batch, sequence, embedding_dim)

        Output:
            (batch, sequence, vocab_size)
        """

        return self.output_projection(hidden_states)