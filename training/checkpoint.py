from pathlib import Path
from dataclasses import asdict

import torch

from utils.config import ModelConfig


class CheckpointManager:
    """
    Handles saving and loading HarshaLM checkpoints.
    """

    def __init__(
        self,
        config: ModelConfig,
    ):

        self.config = config

        self.checkpoint_dir = Path(
            config.checkpoint_dir
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_validation_loss = (
            self._load_best_validation_loss()
        )

    def _load_best_validation_loss(
        self,
    ) -> float:
        """
        Loads the validation loss of the current
        best model if it exists.
        """

        best_checkpoint = (
            self.checkpoint_dir /
            self.config.best_checkpoint_name
        )

        if not best_checkpoint.exists():

            return float("inf")

        try:

            checkpoint = torch.load(
                best_checkpoint,
                map_location=self.config.device,
                weights_only=False,
            )

        except Exception:

            print(
                "Old checkpoint detected. "
                "Ignoring previous best model."
            )

            return float("inf")

        validation_loss = checkpoint.get(
            "validation_loss"
        )

        if validation_loss is None:

            return float("inf")

        print(
            "✓ Loaded best validation loss:",
            validation_loss,
        )

        return validation_loss


    def _create_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        step: int | None,
        train_loss: float | None,
        validation_loss: float | None,
    ):

        return {

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

            "train_loss":
                train_loss,

            "validation_loss":
                validation_loss,

            "config":
                self.config.to_dict(),

            "model_name":
                self.config.model_name,

            "model_version":
                self.config.model_version,
        }


    def save(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        step: int | None = None,
        train_loss: float | None = None,
        validation_loss: float | None = None,
    ):

        checkpoint = self._create_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            step,
            train_loss,
            validation_loss,
        )


        # Save normal epoch checkpoint

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

        best_model_saved = False

        # Save best model

        if (
            validation_loss is not None
            and
            validation_loss <
            self.best_validation_loss
        ):

            self.best_validation_loss = (
                validation_loss
            )


            best_filename = (
                self.checkpoint_dir /
                self.config.best_checkpoint_name
            )


            torch.save(
                checkpoint,
                best_filename
            )


            print(
                f"✓ Saved best model: "
                f"{best_filename}"
            )

            best_model_saved = True

        return best_model_saved


    def load(
        self,
        filename,
        model,
        optimizer=None,
        scheduler=None,
    ):

        checkpoint = torch.load(
            filename,
            map_location=self.config.device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )


        if optimizer is not None:

            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )


        if scheduler is not None:

            scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )


        print(
            f"✓ Loaded checkpoint: {filename}"
        )


        return checkpoint