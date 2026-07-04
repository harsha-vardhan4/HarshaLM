import torch
import torch.nn as nn

from utils.config import ModelConfig
from model.transformer.transformer_block import TransformerBlock


class TransformerStack(nn.Module):
    """
    Stack of GPT-style Transformer Blocks.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass through all Transformer blocks.
        """

        for layer in self.layers:
            x = layer(
                x,
                attention_mask
            )

        return x