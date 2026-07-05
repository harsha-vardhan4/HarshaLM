from utils.config import ModelConfig
from model.harsha_lm import HarshaLM
from training.optimizer import create_optimizer


def test_optimizer():

    config = ModelConfig()

    model = HarshaLM(config)

    optimizer = create_optimizer(
        model,
        config
    )

    print(type(optimizer))

    assert optimizer is not None

    print("\n✓ Optimizer test passed")


if __name__ == "__main__":
    test_optimizer()