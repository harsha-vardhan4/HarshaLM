import torch

from utils.config import ModelConfig
from model.heads.lm_head import LMHead


def test_lm_head():

    config = ModelConfig()

    lm_head = LMHead(config)

    hidden_states = torch.randn(
        2,
        8,
        config.embedding_dim
    )

    logits = lm_head(hidden_states)

    print("Input Shape:")
    print(hidden_states.shape)

    print()

    print("Output Shape:")
    print(logits.shape)

    assert logits.shape == (
        2,
        8,
        config.vocab_size
    )

    print("\n✓ LMHead test passed")


if __name__ == "__main__":
    test_lm_head()