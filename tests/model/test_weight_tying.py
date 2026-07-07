from model.harsha_lm import HarshaLM

from utils.config import ModelConfig


def test_weight_tying():

    config = ModelConfig()

    model = HarshaLM(config)

    embedding_weight = (
        model.embedding_layer
        .token_embedding
        .embedding
        .weight
    )

    lm_head_weight = (
        model.lm_head
        .linear
        .weight
    )

    assert (
        embedding_weight.data_ptr()
        ==
        lm_head_weight.data_ptr()
    )

    print()

    print("✓ Weight tying test passed")


if __name__ == "__main__":
    test_weight_tying()