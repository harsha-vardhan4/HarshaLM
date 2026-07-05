from torch.utils.data import DataLoader

from training.dataset import ConversationDataset
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
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return dataloader