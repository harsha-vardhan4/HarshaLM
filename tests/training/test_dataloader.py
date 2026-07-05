import torch

from utils.config import ModelConfig
from training.dataloader import create_dataloader


def test_dataloader():

    config = ModelConfig()

    token_ids = list(range(1000))

    dataloader = create_dataloader(
        token_ids,
        config,
    )

    inputs, targets = next(iter(dataloader))

    print("Input Shape:")
    print(inputs.shape)

    print()

    print("Target Shape:")
    print(targets.shape)

    assert inputs.shape == (
        config.batch_size,
        config.context_length,
    )

    assert targets.shape == (
        config.batch_size,
        config.context_length,
    )

    print("\n✓ DataLoader test passed")


if __name__ == "__main__":
    test_dataloader()