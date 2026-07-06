import torch

from model.harsha_lm import HarshaLM
from utils.config import ModelConfig
from training.trainer import Trainer
from training.dataloader import create_dataloader


def test_train_epoch():

    config = ModelConfig()

    model = HarshaLM(config)

    trainer = Trainer(
        model=model,
        config=config,
    )

    token_ids = torch.randint(
        0,
        config.vocab_size,
        (5000,),
    ).tolist()

    dataloader = create_dataloader(
        token_ids,
        config,
    )

    average_loss = trainer._train_epoch(
        dataloader,
        epoch=1,
    )

    print()

    print(f"Average Loss: {average_loss:.4f}")

    print("\n✓ Train epoch test passed")


if __name__ == "__main__":
    test_train_epoch()