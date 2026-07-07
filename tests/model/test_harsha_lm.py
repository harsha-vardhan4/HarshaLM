import torch

from utils.config import ModelConfig
from model.harsha_lm import HarshaLM


def test_harsha_lm():

    config = ModelConfig()

    model = HarshaLM(config)

    model.summary()

    assert model.verify_weight_tying()

    print(
        f"Parameters: {model.num_parameters():,}"
    )

    print(
        f"Trainable: {model.trainable_parameters():,}"
    )

    print(
        f"Size: {model.model_size_mb():.2f} MB"
    )

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 8)
    )

    logits = model(input_ids)

    print("Input Shape:")
    print(input_ids.shape)

    print()

    print("Output Shape:")
    print(logits.shape)

    assert logits.shape == (
        2,
        8,
        config.vocab_size
    )

    print("\n✓ HarshaLM test passed")


if __name__ == "__main__":
    test_harsha_lm()