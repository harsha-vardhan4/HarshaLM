import torch

from inference.sampling import TokenSampler


def test_top_p():

    logits = torch.randn(
        1,
        50257,
    )

    token = TokenSampler.top_p(
        logits,
        p=0.9,
        temperature=0.8,
    )

    print(token)

    print(token.shape)

    print("\n✓ Top-p sampling test passed")


if __name__ == "__main__":
    test_top_p()