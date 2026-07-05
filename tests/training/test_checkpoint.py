from pathlib import Path

from utils.config import ModelConfig

from model.harsha_lm import HarshaLM

from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from training.checkpoint import CheckpointManager


def test_checkpoint():

    config = ModelConfig()

    model = HarshaLM(config)

    optimizer = create_optimizer(
        model,
        config
    )

    scheduler = create_scheduler(
        optimizer,
        config
    )

    manager = CheckpointManager(config)

    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        step=25,
        loss=6.42
    )

    checkpoint_path = (
        Path(config.checkpoint_dir)
        / "checkpoint_epoch_1.pt"
    )

    checkpoint = manager.load(
        checkpoint_path,
        model,
        optimizer,
        scheduler
    )

    print()

    print("Epoch:", checkpoint["epoch"])
    print("Step:", checkpoint["step"])
    print("Loss:", checkpoint["loss"])

    print("\n✓ Checkpoint test passed")


if __name__ == "__main__":
    test_checkpoint()