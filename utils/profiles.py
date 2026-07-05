from utils.config import ModelConfig


def development_config() -> ModelConfig:
    """
    Small configuration for quick local testing.
    """

    return ModelConfig(
        context_length=16,
        batch_size=4,
        num_epochs=2,
    )


def production_config() -> ModelConfig:
    """
    Configuration for full-scale training.
    """

    return ModelConfig()