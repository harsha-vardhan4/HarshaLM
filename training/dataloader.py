from torch.utils.data import DataLoader
from torch.utils.data import random_split

from data.training_sample import TrainingSample
from training.conversation_dataset import ConversationDataset
from utils.config import ModelConfig


def create_dataloader(
    token_ids,
    config: ModelConfig,
    shuffle: bool = True,
) -> DataLoader:
    """
    Creates a DataLoader for GPT-style language modeling.
    """

    dataset = ConversationDataset(
        token_ids=token_ids,
        context_length=config.context_length,
        stride=config.stride,
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )


def create_train_validation_dataloaders(
    training_samples: list[TrainingSample],
    config: ModelConfig,
):
    """
    Splits the dataset into training and validation sets
    and creates DataLoaders for both.
    """

    dataset = ConversationDataset(
        samples=training_samples,
        context_length=config.context_length,
        stride=config.stride,
    )

    dataset_size = len(dataset)

    print()
    print("=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Total windows      : {dataset_size}")
    # print(f"Training windows   : {train_size}")
    # print(f"Validation windows : {validation_size}")
    print()

    validation_size = int(
        dataset_size * config.validation_split
    )

    train_size = dataset_size - validation_size

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return (
        train_dataloader,
        validation_dataloader,
    )