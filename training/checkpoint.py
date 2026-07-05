from pathlib import Path
from dataclasses import asdict

import torch

from utils.config import ModelConfig


class CheckpointManager:
    """
    Handles saving and loading HarshaLM checkpoints.
    """

    def __init__(self, config: ModelConfig):

        self.config = config

        self.checkpoint_dir = Path(
            config.checkpoint_dir
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def save(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        step: int | None = None,
        loss: float | None = None,
    ):

        checkpoint = {

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "epoch":
                epoch,

            "step":
                step,

            "loss":
                loss,

            "config":
                asdict(self.config),

            "model_name":
                self.config.model_name,

            "model_version":
                self.config.model_version
        }

        filename = (
            self.checkpoint_dir /
            f"checkpoint_epoch_{epoch}.pt"
        )

        torch.save(
            checkpoint,
            filename
        )

        print(
            f"✓ Saved checkpoint: {filename}"
        )

    def load(
        self,
        filename,
        model,
        optimizer=None,
        scheduler=None
    ):

        checkpoint = torch.load(
            filename,
            map_location=self.config.device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if optimizer is not None:

            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if scheduler is not None:

            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        print(
            f"✓ Loaded checkpoint: {filename}"
        )

        return checkpoint
    