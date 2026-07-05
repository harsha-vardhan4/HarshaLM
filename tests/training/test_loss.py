import torch

from utils.config import ModelConfig
from training.loss import LanguageModelLoss


def test_language_model_loss():

    config = ModelConfig()

    loss_fn = LanguageModelLoss()

    logits = torch.randn(
        2,
        8,
        config.vocab_size
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (2, 8)
    )

    loss = loss_fn(
        logits,
        targets
    )

    print("Logits Shape:")
    print(logits.shape)

    print()

    print("Targets Shape:")
    print(targets.shape)

    print()

    print("Loss:")

    print(loss)

    assert loss.dim() == 0

    print("\n✓ LanguageModelLoss test passed")


if __name__ == "__main__":
    test_language_model_loss()