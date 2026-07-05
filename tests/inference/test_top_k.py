import torch

from inference.sampling import TokenSampler


def test_top_k():

    logits = torch.randn(
        1,
        50257,
    )

    token = TokenSampler.top_k(
        logits,
        k=40,
        temperature=0.8,
    )

    print(token)

    print(token.shape)

    print("\n✓ Top-k sampling test passed")


if __name__ == "__main__":
    test_top_k()