import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

from training.logger import TrainingLogger
from model.harsha_lm import HarshaLM
from training.loss import LanguageModelLoss
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from training.checkpoint import CheckpointManager
from training.metrics import TrainingMetrics
from training.early_stopping import EarlyStopping

from utils.config import ModelConfig



class Trainer:
    """
    Handles the complete training pipeline for HarshaLM.
    """

    
    def __init__(
        self,
        model: HarshaLM,
        config: ModelConfig,
    ):

        self.config = config

        self.device = torch.device(
            config.device
        )

        self.model = model.to(
            self.device
        )

        self.loss_fn = LanguageModelLoss()

        self.optimizer = create_optimizer(
            self.model,
            config
        )

        self.scheduler = create_scheduler(
            self.optimizer,
            config
        )

        self.checkpoint_manager = (
            CheckpointManager(config)
        )

        self.logger = (
            TrainingLogger(
                Path(
                    config.log_dir
                )
                /
                config.training_log_name
            )
        )

        self.early_stopping = (
            EarlyStopping(
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
            )
        )

    def _train_batch(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> float:
        """
        Performs one optimization step.
        """

        self.model.train()

        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        self.optimizer.zero_grad()

        logits = self.model(
            input_ids
        )

        loss = self.loss_fn(
            logits,
            target_ids
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        self.optimizer.step()

        self.scheduler.step()

        return loss.item()

    def _train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> float:
        """
        Trains one epoch.
        """

        total_loss = 0.0

        for batch_index, (input_ids, target_ids) in enumerate(
            dataloader,
            start=1,
        ):

            loss = self._train_batch(
                input_ids,
                target_ids,
            )

            total_loss += loss

            if batch_index % self.config.log_every == 0:

                current_lr = self.optimizer.param_groups[0]["lr"]

                print(
                    f"Epoch [{epoch}] "
                    f"Batch [{batch_index}/{len(dataloader)}] "
                    f"Loss: {loss:.4f} "
                    f"LR: {current_lr:.6f}"
                )

        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def _validate_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> float:
        """
        Evaluates the model on the validation dataset.
        """

        self.model.eval()

        total_loss = 0.0

        if len(dataloader) == 0:
            raise ValueError(
                "Validation dataloader is empty."
            )


        for input_ids, target_ids in dataloader:

            input_ids = input_ids.to(
                self.device
            )

            target_ids = target_ids.to(
                self.device
            )


            logits = self.model(
                input_ids
            )


            loss = self.loss_fn(
                logits,
                target_ids,
            )


            total_loss += loss.item()


        average_loss = (
            total_loss /
            len(dataloader)
        )


        print(
            f"Validation Loss: "
            f"{average_loss:.4f}"
        )


        return average_loss
    
    def resume(
        self,
        checkpoint_path: str,
    ) -> int:
        """
        Resumes training from a checkpoint.

        Returns
        -------
        int
            Next epoch to train.
        """

        checkpoint = (
            self.checkpoint_manager.load(
                checkpoint_path,
                self.model,
                self.optimizer,
                self.scheduler,
            )
        )

        start_epoch = (
            checkpoint["epoch"] + 1
        )

        print()

        print("=" * 60)

        print(
            f"Resuming training from epoch "
            f"{start_epoch}"
        )

        print("=" * 60)

        print()

        return start_epoch

    def train(
        self,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
    ) -> dict[str, list[float]]:
        """
        Runs the complete training process.
        """

        history = {
            "train_loss": [],
            "validation_loss": [],
            "train_perplexity": [],
            "validation_perplexity": [],
        }

        if "--resume" in sys.argv:

            self.config.resume_training = True

        start_epoch = 1

        if self.config.resume_training:

            start_epoch = self.resume(
                self.config.resume_checkpoint
            )

        for epoch in range(
            start_epoch,
            self.config.num_epochs + 1,
        ):

            train_loss = self._train_epoch(
                train_dataloader,
                epoch,
            )

            validation_loss = self._validate_epoch(
                validation_dataloader,
                epoch,
            )

            train_perplexity = (
                TrainingMetrics.perplexity(
                    train_loss
                )
            )

            validation_perplexity = (
                TrainingMetrics.perplexity(
                    validation_loss
                )
            )

            history["train_loss"].append(
                train_loss
            )

            history["validation_loss"].append(
                validation_loss
            )

            history["train_perplexity"].append(
                train_perplexity
            )

            history["validation_perplexity"].append(
                validation_perplexity
            )

            print()

            print("=" * 60)

            print(
                f"Epoch "
                f"{epoch}/{self.config.num_epochs}"
            )

            print("-" * 60)

            print(
                f"Train Loss            : "
                f"{train_loss:.4f}"
            )

            print(
                f"Validation Loss       : "
                f"{validation_loss:.4f}"
            )

            print(
                f"Train Perplexity      : "
                f"{train_perplexity:.4f}"
            )

            print(
                f"Validation Perplexity : "
                f"{validation_perplexity:.4f}"
            )

            print(
                f"Learning Rate         : "
                f"{self.optimizer.param_groups[0]['lr']:.8f}"
            )

            print("=" * 60)

            print()

            stop_training = (
                self.early_stopping.step(
                    validation_loss
                )
            )

            is_best_model = (
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    step=len(train_dataloader),
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                )
            )

            if self.early_stopping.improved:

                print(
                    "✓ Validation loss improved."
                )

            else:

                print(
                    f"No improvement "
                    f"({self.early_stopping.counter}/"
                    f"{self.early_stopping.patience})"
                )

            if stop_training:

                print()

                print("=" * 60)

                print(
                    "Early stopping triggered."
                )

                print(
                    f"Best Validation Loss : "
                    f"{self.early_stopping.best_loss:.4f}"
                )

                print("=" * 60)

                print()

                break

            self.logger.log(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                train_perplexity=train_perplexity,
                validation_perplexity=validation_perplexity,
                learning_rate=self.optimizer.param_groups[0]["lr"],
                is_best_model=is_best_model,
                validation_improved=(
                    self.early_stopping.improved
                ),
            )

        return history