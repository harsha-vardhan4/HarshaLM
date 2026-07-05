import torch

from inference.sampling import TokenSampler


def test_temperature_sampling():

    logits = torch.randn(
        1,
        50257,
    )

    token = TokenSampler.temperature(
        logits,
        temperature=0.8,
    )

    print(token.shape)

    print(token)

    print("\n✓ Temperature sampling test passed")


if __name__ == "__main__":
    test_temperature_sampling()