from pathlib import Path

import torch

from model.harsha_lm import HarshaLM
from training.checkpoint import CheckpointManager
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from utils.config import ModelConfig


def test_checkpoint():

    config = ModelConfig()

    model = HarshaLM(config)

    optimizer = create_optimizer(
        model,
        config,
    )

    scheduler = create_scheduler(
        optimizer,
        config,
    )

    manager = CheckpointManager(
        config
    )

    #
    # Save first checkpoint
    #

    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        step=10,
        train_loss=5.2,
        validation_loss=4.8,
    )

    checkpoint_file = (
        Path(config.checkpoint_dir)
        / "checkpoint_epoch_1.pt"
    )

    best_file = (
        Path(config.checkpoint_dir)
        / "best_model.pt"
    )

    assert checkpoint_file.exists()

    assert best_file.exists()

    #
    # Load checkpoint
    #

    checkpoint = manager.load(
        checkpoint_file,
        model,
        optimizer,
        scheduler,
    )

    assert checkpoint["epoch"] == 1

    assert checkpoint["step"] == 10

    assert checkpoint["train_loss"] == 5.2

    assert checkpoint["validation_loss"] == 4.8

    print()

    print("✓ Checkpoint test passed")


if __name__ == "__main__":
    test_checkpoint()