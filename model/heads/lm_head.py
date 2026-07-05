import torch
import torch.nn as nn

from utils.config import ModelConfig


class LMHead(nn.Module):
    """
    Projects hidden states to vocabulary logits.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.linear = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:

        return self.linear(hidden_states)