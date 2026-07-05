from model.harsha_lm import HarshaLM
from utils.config import ModelConfig
from training.trainer import Trainer
import torch

from training.dataloader import create_dataloader


def test_trainer():

    config = ModelConfig(
        num_epochs=2,
        batch_size=8,
        context_length=32,
    )

    token_ids = torch.randint(
        0,
        config.vocab_size,
        (5000,)
    ).tolist()

    dataloader = create_dataloader(
        token_ids,
        config,
    )

    model = HarshaLM(config)

    trainer = Trainer(
        model,
        config,
    )

    history = trainer.train(
        dataloader
    )

    print("\nTraining History:")
    print(history)

    assert len(history) == config.num_epochs

    print("\n✓ Trainer test passed")


if __name__ == "__main__":
    test_trainer()