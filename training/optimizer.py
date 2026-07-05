import torch

from utils.config import ModelConfig


def create_optimizer(
    model: torch.nn.Module,
    config: ModelConfig
) -> torch.optim.Optimizer:
    """
    Creates the optimizer for HarshaLM.
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    return optimizer