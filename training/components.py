from dataclasses import dataclass

from torch.utils.data import DataLoader

from model.harsha_lm import HarshaLM
from training.trainer import Trainer


@dataclass
class TrainingComponents:
    """
    Contains all objects required for training.
    """

    model: HarshaLM

    trainer: Trainer

    train_dataloader: DataLoader

    validation_dataloader: DataLoader

    tokenizer: object