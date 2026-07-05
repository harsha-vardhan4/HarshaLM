import torch
from torch.utils.data import DataLoader

from model.harsha_lm import HarshaLM
from training.loss import LanguageModelLoss
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from training.checkpoint import CheckpointManager

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

    def train(
        self,
        dataloader: DataLoader,
    ) -> list[float]:
        """
        Runs the complete training process.
        """

        history = []

        for epoch in range(
            1,
            self.config.num_epochs + 1,
        ):

            average_loss = self._train_epoch(
                dataloader,
                epoch,
            )

            history.append(
                average_loss
            )

            print(
                f"\nEpoch "
                f"[{epoch}/{self.config.num_epochs}] "
                f"Average Loss: "
                f"{average_loss:.4f}\n"
            )

            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                step=len(dataloader),
                loss=average_loss,
            )

        return history