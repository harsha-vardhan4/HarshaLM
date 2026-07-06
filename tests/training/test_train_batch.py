import torch

from model.harsha_lm import HarshaLM
from utils.config import ModelConfig
from training.trainer import Trainer


def test_train_batch():

    config = ModelConfig()

    model = HarshaLM(config)

    trainer = Trainer(
        model=model,
        config=config,
    )

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (
            config.batch_size,
            config.context_length,
        ),
    )

    target_ids = torch.randint(
        0,
        config.vocab_size,
        (
            config.batch_size,
            config.context_length,
        ),
    )

    loss = trainer._train_batch(
        input_ids,
        target_ids,
    )

    print("Loss:")
    print(loss)

    assert isinstance(loss, float)

    print("\n✓ Train batch test passed")


if __name__ == "__main__":
    test_train_batch()