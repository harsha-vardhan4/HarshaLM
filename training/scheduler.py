import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from utils.config import ModelConfig


def create_scheduler(
    optimizer: Optimizer,
    config: ModelConfig
):
    """
    Creates a learning rate scheduler with:

    - Linear warmup
    - Cosine decay
    """

    def lr_lambda(current_step: int):

        # Warmup
        if current_step < config.warmup_steps:
            return current_step / max(1, config.warmup_steps)

        # Cosine decay
        progress = (
            current_step - config.warmup_steps
        ) / max(
            1,
            config.max_training_steps - config.warmup_steps
        )

        return 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return LambdaLR(
        optimizer,
        lr_lambda
    )