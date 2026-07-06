from model.harsha_lm import HarshaLM

from training.components import TrainingComponents
from training.trainer import Trainer
from training.dataloader import create_dataloader

from utils.config import ModelConfig


def test_components():

    config = ModelConfig()

    model = HarshaLM(config)

    trainer = Trainer(
        model=model,
        config=config,
    )

    token_ids = list(
        range(
            config.context_length * 3
        )
    )

    dataloader = create_dataloader(
        token_ids,
        config,
    )

    components = TrainingComponents(
        model=model,
        trainer=trainer,
        dataloader=dataloader,
        tokenizer=None,
    )

    assert components.model is model
    assert components.trainer is trainer
    assert components.dataloader is dataloader

    print()

    print("✓ TrainingComponents test passed")


if __name__ == "__main__":
    test_components()